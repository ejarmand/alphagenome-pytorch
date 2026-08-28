#!/usr/bin/env python
"""Fine-tune frozen AlphaGenome heads from Baskerville TFRecords."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
# Preload Dynamo/Triton before TensorFlow is imported lazily for TFRecord I/O.
# Loading these native runtimes in the opposite order can segfault on Cuica.
import torch._dynamo
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from tqdm import tqdm

from alphagenome_pytorch import AlphaGenome
from alphagenome_pytorch.config import DtypePolicy
from alphagenome_pytorch.extensions.finetuning.heads import create_finetuning_head
from alphagenome_pytorch.extensions.finetuning.tfrecord_dataset import (
    BaskervilleTFRecordDataset,
    BaskervilleMultiTFRecordDataset,
    collate_tfr_multimodal,
)
from alphagenome_pytorch.extensions.finetuning.training import create_lr_scheduler
from alphagenome_pytorch.extensions.finetuning.transfer import load_trunk, remove_all_heads
from alphagenome_pytorch.heads import EMBEDDING_128BP_DIM
from alphagenome_pytorch.losses import multinomial_loss
from alphagenome_pytorch.utils.sequence import reverse_complement_onehot_tensor


DEFAULT_HF_MODEL_ID = "gtca/alphagenome_pytorch"
DEFAULT_HF_FILENAME = "model_fold_1.safetensors"
ENCODER_SKIP_32BP_KEY = "encoder_skip_32bp"
ENCODER_SKIP_64BP_KEY = "encoder_skip_64bp"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a new AlphaGenome head from Baskerville TFRecords.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--pretrained-weights",
        type=Path,
        help=(
            "Local PyTorch weights (.pth/.safetensors). If omitted, downloads "
            "--hf-filename from --hf-model-id."
        ),
    )
    parser.add_argument(
        "--hf-model-id",
        default=DEFAULT_HF_MODEL_ID,
        help=(
            "Hugging Face AlphaGenome PyTorch checkpoint repo to download from "
            "when --pretrained-weights is omitted."
        ),
    )
    parser.add_argument(
        "--hf-filename",
        default=DEFAULT_HF_FILENAME,
        help="Checkpoint filename in --hf-model-id to download.",
    )
    parser.add_argument(
        "--hf-revision",
        default=None,
        help="Optional Hugging Face revision for --hf-model-id.",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help=(
            "Optional Hugging Face token. If omitted, huggingface_hub uses the "
            "cached login token or HF_TOKEN/HUGGING_FACE_HUB_TOKEN."
        ),
    )
    parser.add_argument(
        "--hf-cache-dir",
        type=Path,
        default=None,
        help="Optional Hugging Face cache directory for the downloaded safetensors file.",
    )
    parser.add_argument(
        "--modality",
        action="append",
        default=None,
        help=(
            "Modality to train. Repeat for multiple heads, or omit/use 'all' "
            "to train all modalities in targets.txt."
        ),
    )
    parser.add_argument("--pooling", choices=["mean", "sum"], default="mean")
    parser.add_argument(
        "--target-resolution",
        type=int,
        choices=[32, 128],
        default=128,
        help=(
            "Target bin size to train against. 128 pools four raw 32 bp TFRecord "
            "bins; 32 trains on raw TFRecord targets using a 4x U-Net upsampler."
        ),
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for head initialization and cached-data augmentation.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--val-num-workers",
        type=int,
        default=0,
        help=(
            "Validation DataLoader workers per rank. Defaults to 0 to avoid "
            "forking TFRecord workers after CUDA/DDP initialization."
        ),
    )
    parser.add_argument(
        "--prefetch-n",
        type=int,
        default=1,
        help=(
            "DataLoader prefetch_factor when workers are enabled. Ignored when "
            "the effective worker count is 0."
        ),
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=500,
        help=(
            "Warmup optimizer steps. Set 0 to disable warmup."
        ),
    )
    parser.add_argument(
        "--warmup-fraction",
        type=float,
        default=None,
        help=(
            "Warmup as a fraction of estimated optimizer steps. When set, "
            "this overrides --warmup-steps."
        ),
    )
    parser.add_argument(
        "--lr-schedule",
        choices=["constant", "cosine"],
        default="cosine",
        help="Learning rate schedule after optional warmup.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Accumulate gradients over this many local batches before optimizer.step().",
    )
    parser.add_argument(
        "--loss",
        choices=["poisson-multinomial", "multinomial", "poisson", "mse"],
        default="poisson-multinomial",
        help=(
            "Training loss. 'poisson-multinomial' is the AlphaGenome-style "
            "combined Poisson total-count and multinomial profile loss; "
            "'multinomial' is kept as an alias."
        ),
    )
    parser.add_argument("--positional-weight", type=float, default=5.0)
    parser.add_argument("--count-weight", type=float, default=1.0)
    parser.add_argument("--track-means-samples", type=int, default=16)
    parser.add_argument(
        "--cell-embedding-dim",
        type=int,
        default=16,
        help=(
            "Bottleneck dimension for shared cell-type embeddings before the "
            "per-modality 1-output layer."
        ),
    )
    parser.add_argument("--max-train-steps", type=int)
    parser.add_argument("--max-val-steps", type=int)
    parser.add_argument(
        "--tfr-num-parallel-reads",
        type=int,
        default=1,
        help=(
            "Number of parallel TFRecord reads per DataLoader worker. Defaults "
            "to 1 to avoid TensorFlow thread oversubscription under torchrun."
        ),
    )
    parser.add_argument(
        "--augment-rc",
        action="store_true",
        help=(
            "Apply Baskerville-style stochastic reverse-complement augmentation "
            "during training and switch predictions back with strand_pair."
        ),
    )
    parser.add_argument(
        "--augment-shift",
        type=int,
        default=0,
        help=(
            "Apply Baskerville-style stochastic sequence shifts in [-N, N] bp "
            "during training. Targets are left unchanged."
        ),
    )
    parser.add_argument(
        "--precompute-embeddings",
        action="store_true",
        help=(
            "Precompute/cache original and reverse-complement frozen trunk "
            "embeddings for train/valid TFRecords, then train heads from the "
            "cached embeddings instead of forwarding the trunk each step."
        ),
    )
    parser.add_argument(
        "--embedding-cache-dir",
        type=Path,
        default=None,
        help=(
            "Directory for cached embedding chunks. Defaults to a settings-keyed "
            "subdirectory under --output-dir/embedding_cache."
        ),
    )
    parser.add_argument(
        "--refresh-embedding-cache",
        action="store_true",
        help="Recompute cached embeddings even when matching cache manifests exist.",
    )
    parser.add_argument(
        "--embedding-cache-dtype",
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
        help="Dtype used when storing cached 128 bp embeddings.",
    )
    parser.add_argument(
        "--embedding-cache-chunk-size",
        type=int,
        default=16,
        help="Number of examples per cached embedding chunk file.",
    )
    parser.add_argument(
        "--grad-checkpoint",
        action="store_true",
        help=(
            "Recompute the 32 bp upsampler's ConvNormGelu blocks during backward "
            "instead of storing their activations. Cuts head activation memory "
            "(~37%% at 32 bp) for ~15-30%% more compute; no effect at 128 bp."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("finetuning_output/tfr_heads"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', 'cuda', or e.g. 'cuda:0'")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument(
        "--organism-idx",
        type=int,
        default=0,
        help="AlphaGenome organism index used for trunk embeddings. 0=human, 1=mouse.",
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging on rank 0.")
    parser.add_argument("--wandb-project", default="alphagenome-tfr-finetune")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-tags", default=None, help="Comma-separated W&B tags.")
    parser.add_argument(
        "--debug-ranks",
        action="store_true",
        help="Print per-rank progress around DataLoader and distributed synchronization.",
    )
    parser.add_argument("--loader-only", action="store_true", help="Decode one batch and exit.")
    return parser.parse_args()


def create_tfr_dataset(
    data_dir: Path,
    split: str,
    modalities: list[str],
    pooling: str,
    target_resolution: int,
    *,
    repeat: bool = False,
    shuffle_files: bool = False,
    num_parallel_reads: int | None = 1,
    rank: int = 0,
    world_size: int = 1,
    augment_rc: bool = False,
    augment_shift: int = 0,
) -> BaskervilleMultiTFRecordDataset:
    return BaskervilleMultiTFRecordDataset(
        data_dir,
        split=split,  # type: ignore[arg-type]
        modalities=modalities,
        pooling=pooling,  # type: ignore[arg-type]
        target_resolution=target_resolution,  # type: ignore[arg-type]
        repeat=repeat,
        shuffle_files=shuffle_files,
        num_parallel_reads=num_parallel_reads,
        rank=rank,
        world_size=world_size,
        augment_rc=augment_rc,
        augment_shift=augment_shift,
    )


def create_loader(
    dataset: BaskervilleMultiTFRecordDataset,
    batch_size: int,
    num_workers: int,
    *,
    prefetch_n: int = 1,
    persistent_workers: bool = False,
) -> DataLoader:
    if prefetch_n < 1:
        raise ValueError(f"prefetch_n must be >= 1, got {prefetch_n}")
    local_files = dataset._worker_files()
    effective_workers = min(num_workers, len(local_files)) if num_workers > 0 else 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=effective_workers,
        collate_fn=collate_tfr_multimodal,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=prefetch_n if effective_workers > 0 else None,
        persistent_workers=persistent_workers and effective_workers > 0,
    )


def collate_tfr_cached_embeddings(
    batch: list[
        tuple[dict[int | str, torch.Tensor], dict[str, dict[int, torch.Tensor]]]
        | tuple[
            dict[int | str, torch.Tensor],
            dict[str, dict[int, torch.Tensor]],
            dict[str, bool | int],
        ]
    ],
) -> (
    tuple[dict[int | str, torch.Tensor], dict[str, dict[int, torch.Tensor]]]
    | tuple[
        dict[int | str, torch.Tensor],
        dict[str, dict[int, torch.Tensor]],
        dict[str, torch.Tensor],
    ]
):
    first_embeddings = batch[0][0]
    embeddings = {
        key: torch.stack([item[0][key] for item in batch], dim=0)
        for key in first_embeddings
    }
    first_targets = batch[0][1]
    targets = {
        modality: {
            resolution: torch.stack(
                [item[1][modality][resolution] for item in batch],
                dim=0,
            )
            for resolution in modality_targets
        }
        for modality, modality_targets in first_targets.items()
    }
    if len(batch[0]) == 3:
        augmentation = {
            "reverse_complement": torch.tensor(
                [bool(item[2]["reverse_complement"]) for item in batch],
                dtype=torch.bool,
            ),
            "shift": torch.tensor(
                [int(item[2]["shift"]) for item in batch],
                dtype=torch.long,
            ),
        }
        if "input_reverse_complement" in batch[0][2]:
            augmentation["input_reverse_complement"] = torch.tensor(
                [bool(item[2]["input_reverse_complement"]) for item in batch],
                dtype=torch.bool,
            )
        return embeddings, targets, augmentation
    return embeddings, targets


def reverse_complement_targets(
    targets: torch.Tensor,
    strand_pair: torch.Tensor | np.ndarray,
) -> torch.Tensor:
    """Reverse target bins and swap stranded channels into RC target order."""
    pair = torch.as_tensor(strand_pair, dtype=torch.long)
    return torch.flip(targets, dims=[1]).index_select(-1, pair)


class CachedEmbeddingTFRecordDataset(IterableDataset):
    """Iterable dataset backed by precomputed original/RC trunk embeddings."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        split: str,
        source_dataset: BaskervilleMultiTFRecordDataset,
        rank: int = 0,
        world_size: int = 1,
        augment_rc: bool = False,
        shuffle_files: bool = False,
        seed: int = 0,
    ):
        super().__init__()
        self.cache_dir = Path(cache_dir)
        self.split = split
        self.rank = rank
        self.world_size = world_size
        self.augment_rc = augment_rc
        self.shuffle_files = shuffle_files
        self.seed = seed
        self.target_resolution = source_dataset.target_resolution
        self._iteration = 0

        self.metadata = source_dataset.metadata
        self.modalities = list(source_dataset.modalities)
        self.target_indices_by_modality = source_dataset.target_indices_by_modality
        self.target_rows_by_modality = source_dataset.target_rows_by_modality
        self.strand_pair_by_modality = source_dataset.strand_pair_by_modality
        self.assay_type_by_modality = source_dataset.assay_type_by_modality
        self._global_length = len(source_dataset)

        file_index_by_path = {
            path.resolve(): file_idx
            for file_idx, path in enumerate(source_dataset.files)
        }
        manifests = []
        for source_file in source_dataset._worker_files():
            file_idx = file_index_by_path[source_file.resolve()]
            manifest_path = embedding_cache_file_manifest_path(
                self.cache_dir,
                split,
                file_idx,
            )
            if not manifest_path.exists():
                raise FileNotFoundError(
                    f"Cached embedding file manifest not found: {manifest_path}"
                )
            manifests.append(json.loads(manifest_path.read_text()))

        self.manifests = manifests
        self.chunk_files = [
            self.cache_dir / chunk["path"]
            for manifest in self.manifests
            for chunk in manifest.get("chunks", [])
        ]
        missing = [path for path in self.chunk_files if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Cached embedding chunks are missing: "
                + ", ".join(str(path) for path in missing[:3])
            )

    def __len__(self) -> int:
        return self._global_length

    @property
    def prediction_crop_128bp(self) -> int:
        return self.metadata.prediction_crop_128bp

    @property
    def output_length_128bp(self) -> int:
        return self.metadata.output_length_128bp

    @property
    def prediction_crop_bins(self) -> int:
        return self.metadata.prediction_crop_bins(self.target_resolution)

    @property
    def output_length(self) -> int:
        return self.metadata.output_length(self.target_resolution)

    def _next_iteration(self) -> int:
        iteration = self._iteration
        self._iteration += 1
        return iteration

    def _rng(self, iteration: int) -> random.Random:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        return random.Random(
            self.seed
            + self.rank * 1009
            + worker_id * 9176
            + iteration * 104729
        )

    def _worker_chunk_files(self, iteration: int) -> list[Path]:
        files = list(self.chunk_files)
        worker = get_worker_info()
        if worker is not None:
            files = files[worker.id::worker.num_workers]
        if self.shuffle_files:
            rng = self._rng(iteration)
            rng.shuffle(files)
        return files

    def __iter__(self):
        iteration = self._next_iteration()
        rng = self._rng(iteration)
        target_resolution = self.target_resolution
        for chunk_path in self._worker_chunk_files(iteration):
            chunk = torch.load(chunk_path, map_location="cpu")
            embeddings_forward = chunk["embeddings_forward_128bp"]
            embeddings_reverse = chunk["embeddings_reverse_128bp"]
            skip_64_forward = chunk.get("encoder_skip_forward_64bp")
            skip_64_reverse = chunk.get("encoder_skip_reverse_64bp")
            skip_32_forward = chunk.get("encoder_skip_forward_32bp")
            skip_32_reverse = chunk.get("encoder_skip_reverse_32bp")
            targets_by_modality = chunk["targets"]
            reverse_targets_by_modality = chunk.get("targets_reverse")
            for sample_idx in range(embeddings_forward.shape[0]):
                reverse_complement = self.augment_rc and rng.random() > 0.5
                embeddings = {
                    128: (
                        embeddings_reverse[sample_idx]
                        if reverse_complement
                        else embeddings_forward[sample_idx]
                    )
                }
                if skip_64_forward is not None and skip_64_reverse is not None:
                    embeddings[ENCODER_SKIP_64BP_KEY] = (
                        skip_64_reverse[sample_idx]
                        if reverse_complement
                        else skip_64_forward[sample_idx]
                    )
                if skip_32_forward is not None and skip_32_reverse is not None:
                    embeddings[ENCODER_SKIP_32BP_KEY] = (
                        skip_32_reverse[sample_idx]
                        if reverse_complement
                        else skip_32_forward[sample_idx]
                    )
                selected_targets_by_modality = targets_by_modality
                switch_predictions = reverse_complement
                if reverse_complement and reverse_targets_by_modality is not None:
                    selected_targets_by_modality = reverse_targets_by_modality
                    switch_predictions = False
                targets = {
                    modality: {target_resolution: targets[target_resolution][sample_idx]}
                    for modality, targets in selected_targets_by_modality.items()
                }
                if self.augment_rc:
                    yield (
                        embeddings,
                        targets,
                        {
                            "reverse_complement": switch_predictions,
                            "input_reverse_complement": reverse_complement,
                            "shift": 0,
                        },
                    )
                else:
                    yield embeddings, targets


def create_cached_embedding_loader(
    dataset: CachedEmbeddingTFRecordDataset,
    batch_size: int,
    num_workers: int,
    *,
    prefetch_n: int = 1,
    persistent_workers: bool = False,
) -> DataLoader:
    if prefetch_n < 1:
        raise ValueError(f"prefetch_n must be >= 1, got {prefetch_n}")
    effective_workers = min(num_workers, len(dataset.chunk_files)) if num_workers > 0 else 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=effective_workers,
        collate_fn=collate_tfr_cached_embeddings,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=prefetch_n if effective_workers > 0 else None,
        persistent_workers=persistent_workers and effective_workers > 0,
    )


def estimate_local_batches(
    dataset: BaskervilleMultiTFRecordDataset | CachedEmbeddingTFRecordDataset,
    batch_size: int,
    max_steps: int | None,
) -> int:
    if max_steps is not None:
        return max_steps
    local_examples = math.ceil(len(dataset) / dataset.world_size)
    return max(1, math.ceil(local_examples / batch_size))


def estimate_total_optimizer_steps(
    dataset: BaskervilleMultiTFRecordDataset | CachedEmbeddingTFRecordDataset,
    batch_size: int,
    epochs: int,
    accumulation_steps: int,
    max_train_steps: int | None,
) -> int:
    local_batches = estimate_local_batches(dataset, batch_size, max_train_steps)
    return max(1, epochs * math.ceil(local_batches / accumulation_steps))


def resolve_warmup_steps(warmup_steps: int | None, total_steps: int) -> int:
    if warmup_steps is not None:
        return warmup_steps
    return max(1, math.ceil(total_steps * 0.01))


def _module_key(name: str, used: set[str]) -> str:
    key = re.sub(r"[^A-Za-z0-9_]", "_", name).strip("_") or "cell"
    if key[0].isdigit():
        key = f"cell_{key}"
    base = key
    suffix = 1
    while key in used:
        suffix += 1
        key = f"{base}_{suffix}"
    used.add(key)
    return key


def create_cell_type_key_map(cell_types_by_modality: dict[str, list[str]]) -> dict[str, str]:
    used: set[str] = set()
    key_by_cell_type = {}
    for cell_types in cell_types_by_modality.values():
        for cell_type in cell_types:
            if cell_type not in key_by_cell_type:
                key_by_cell_type[cell_type] = _module_key(cell_type, used)
    return key_by_cell_type


def create_modality_type_key_map(
    modality_types_by_modality: dict[str, list[str]],
) -> dict[str, str]:
    used: set[str] = set()
    key_by_modality_type = {}
    for modality_types in modality_types_by_modality.values():
        for modality_type in modality_types:
            if modality_type not in key_by_modality_type:
                safe_name = modality_type.replace("+", "_plus").replace("-", "_minus")
                key_by_modality_type[modality_type] = _module_key(safe_name, used)
    return key_by_modality_type


def target_row_strand(row: dict[str, str], idx: int) -> str:
    strand = row.get("strand") if row.get("strand") in ("+", "-") else ""
    if strand:
        return strand
    try:
        row_index = int(row.get("index", idx))
        strand_pair = int(row.get("strand_pair", row_index))
    except (TypeError, ValueError):
        return ""
    identifier = row.get("identifier", "")
    if strand_pair != row_index and identifier.endswith(("+", "-")):
        return identifier[-1]
    return ""


def cell_types_from_target_rows(
    target_rows_by_modality: dict[str, list[dict[str, str]]],
) -> dict[str, list[str]]:
    cell_types_by_modality = {}
    for modality, rows in target_rows_by_modality.items():
        cell_types = []
        for idx, row in enumerate(rows):
            cell_type = row.get("ct") or row.get("identifier") or f"{modality}_{idx}"
            cell_types.append(cell_type)
        cell_types_by_modality[modality] = cell_types
    return cell_types_by_modality


def modality_types_from_target_rows(
    target_rows_by_modality: dict[str, list[dict[str, str]]],
) -> dict[str, list[str]]:
    modality_types_by_modality = {}
    for modality, rows in target_rows_by_modality.items():
        modality_types = []
        for idx, row in enumerate(rows):
            strand = target_row_strand(row, idx)
            modality_types.append(f"{modality}{strand}" if strand else modality)
        modality_types_by_modality[modality] = modality_types
    return modality_types_by_modality


class CellTypeTrackGroups(nn.Module):
    """Track-index buffers for one modality, grouped by cell and decoder type."""

    def __init__(self, cell_keys: list[str], modality_keys: list[str]):
        super().__init__()
        if len(cell_keys) != len(modality_keys):
            raise ValueError(
                "cell_keys and modality_keys must have the same number of tracks"
            )
        self.cell_keys = []
        cell_index_by_key = {}
        for cell_key in dict.fromkeys(cell_keys):
            indices = [
                track_idx
                for track_idx, track_cell_key in enumerate(cell_keys)
                if track_cell_key == cell_key
            ]
            cell_index_by_key[cell_key] = len(self.cell_keys)
            self.cell_keys.append(cell_key)
            self.register_buffer(
                cell_key,
                torch.tensor(indices, dtype=torch.long),
                persistent=False,
            )
        self.modality_keys = list(dict.fromkeys(modality_keys))
        modality_index_by_key = {
            modality_key: idx for idx, modality_key in enumerate(self.modality_keys)
        }
        self.register_buffer(
            "track_cell_indices",
            torch.tensor(
                [cell_index_by_key[cell_key] for cell_key in cell_keys],
                dtype=torch.long,
            ),
            persistent=False,
        )
        self.register_buffer(
            "track_modality_indices",
            torch.tensor(
                [modality_index_by_key[modality_key] for modality_key in modality_keys],
                dtype=torch.long,
            ),
            persistent=False,
        )


class CellTypeEmbedding(nn.Module):
    """Per-cell bottleneck projection from trunk embeddings to cell embeddings."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.proj = nn.Conv1d(input_dim, output_dim, kernel_size=1)
        stdv = 1.0 / math.sqrt(input_dim)
        nn.init.trunc_normal_(self.proj.weight, std=stdv)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class ModalityDecoder(nn.Module):
    """Shared per-modality decoder from cell embeddings to one track value."""

    def __init__(
        self,
        embedding_dim: int,
        target_resolution: int = 128,
        grad_checkpoint: bool = False,
    ):
        super().__init__()
        self.target_resolution = target_resolution
        if target_resolution == 32:
            self.upscaler = EncoderSkip32bpUpsampler(embedding_dim, grad_checkpoint)
        elif target_resolution == 128:
            self.upscaler = nn.Identity()
        else:
            raise ValueError(
                f"target_resolution must be 32 or 128, got {target_resolution}"
            )
        self.layers = nn.Sequential(
            nn.Conv1d(embedding_dim, embedding_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(embedding_dim, 1, kernel_size=1),
        )
        for layer in self.layers:
            if isinstance(layer, nn.Conv1d):
                stdv = 1.0 / math.sqrt(layer.in_channels)
                nn.init.trunc_normal_(layer.weight, std=stdv)
                nn.init.zeros_(layer.bias)

    def forward(
        self,
        x: torch.Tensor,
        *,
        encoder_skips: dict[str, torch.Tensor] | None = None,
        n_cells: int = 1,
    ) -> torch.Tensor:
        if self.target_resolution == 32:
            x = self.upscaler(x, encoder_skips=encoder_skips, n_cells=n_cells)
        else:
            x = self.upscaler(x)
        return self.layers(x)


class ConvNormGelu(nn.Module):
    """Small NCL convolution block used by the 32 bp fine-tuning upsampler."""

    def __init__(self, channels: int, kernel_size: int = 5):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        stdv = 1.0 / math.sqrt(channels * kernel_size)
        nn.init.trunc_normal_(self.conv.weight, std=stdv)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.gelu(self.norm(x)))


class EncoderSkip32bpUpsampler(nn.Module):
    """Upscale 128 bp cell embeddings to 32 bp bins using encoder skip features."""

    def __init__(self, channels: int, grad_checkpoint: bool = False):
        super().__init__()
        self.grad_checkpoint = grad_checkpoint
        self.stem = ConvNormGelu(channels)
        self.skip_64bp = nn.Conv1d(1536, channels, kernel_size=1)
        self.skip_32bp = nn.Conv1d(1408, channels, kernel_size=1)
        self.up_refine_64bp = ConvNormGelu(channels)
        self.up_refine_32bp = ConvNormGelu(channels)
        for layer in (self.skip_64bp, self.skip_32bp):
            stdv = 1.0 / math.sqrt(layer.in_channels)
            nn.init.trunc_normal_(layer.weight, std=stdv)
            nn.init.zeros_(layer.bias)

    def _refine(self, block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        # Recompute the block's norm/gelu/conv activations in backward instead of
        # storing them; these fp32 GroupNorm tensors at 8192/16384 length dominate
        # the 32 bp head's activation memory.
        if self.grad_checkpoint and torch.is_grad_enabled() and x.requires_grad:
            return checkpoint(block, x, use_reentrant=False)
        return block(x)

    def forward(
        self,
        x: torch.Tensor,
        *,
        encoder_skips: dict[str, torch.Tensor] | None,
        n_cells: int,
    ) -> torch.Tensor:
        if encoder_skips is None:
            raise KeyError("32 bp upsampling requires cached/runtime encoder skip features")
        skip_64bp = encoder_skips.get(ENCODER_SKIP_64BP_KEY)
        skip_32bp = encoder_skips.get(ENCODER_SKIP_32BP_KEY)
        if skip_64bp is None or skip_32bp is None:
            raise KeyError(
                "32 bp upsampling requires encoder_skip_64bp and encoder_skip_32bp"
            )

        x_128bp = x + self._refine(self.stem, x)

        projected_64bp = self.skip_64bp(skip_64bp).repeat_interleave(n_cells, dim=0)
        up_64bp = F.interpolate(x_128bp, size=projected_64bp.shape[-1], mode="nearest")
        up_64bp = up_64bp + projected_64bp
        up_64bp = up_64bp + self._refine(self.up_refine_64bp, up_64bp)

        projected_32bp = self.skip_32bp(skip_32bp).repeat_interleave(n_cells, dim=0)
        up_32bp = F.interpolate(up_64bp, size=projected_32bp.shape[-1], mode="nearest")
        up_32bp = up_32bp + projected_32bp
        return up_32bp + self._refine(self.up_refine_32bp, up_32bp)


class TFRHeads(nn.Module):
    """DDP-friendly wrapper with shared cell-type layers and modality heads."""

    def __init__(
        self,
        heads: dict[str, nn.Module],
        cell_types_by_modality: dict[str, list[str]] | None = None,
        modality_types_by_modality: dict[str, list[str]] | None = None,
        embedding_dim: int = EMBEDDING_128BP_DIM,
        cell_embedding_dim: int = 16,
        target_resolution: int = 128,
        grad_checkpoint: bool = False,
    ):
        super().__init__()
        if cell_embedding_dim < 1:
            raise ValueError(
                f"cell_embedding_dim must be >= 1, got {cell_embedding_dim}"
            )
        if target_resolution not in (32, 128):
            raise ValueError(
                f"target_resolution must be 32 or 128, got {target_resolution}"
            )
        self.target_resolution = target_resolution
        self.heads = nn.ModuleDict(heads)
        if modality_types_by_modality is None:
            modality_types_by_modality = {}
        else:
            modality_types_by_modality = dict(modality_types_by_modality)
        for modality, head in heads.items():
            modality_types_by_modality.setdefault(
                modality,
                [modality] * head.num_tracks,
            )
        self.modality_key_by_type = create_modality_type_key_map(
            modality_types_by_modality
        )
        self.modality_layers = nn.ModuleDict({
            modality_key: self._create_modality_layer(
                cell_embedding_dim, target_resolution, grad_checkpoint
            )
            for modality_key in self.modality_key_by_type.values()
        })
        self.cell_key_by_type = (
            create_cell_type_key_map(cell_types_by_modality)
            if cell_types_by_modality is not None
            else {}
        )
        self.cell_layers = nn.ModuleDict({
            cell_key: self._create_cell_layer(embedding_dim, cell_embedding_dim)
            for cell_key in self.cell_key_by_type.values()
        })
        self.track_groups = nn.ModuleDict()
        if cell_types_by_modality is not None:
            for modality, cell_types in cell_types_by_modality.items():
                if modality not in self.heads:
                    continue
                for param in self.heads[modality].convs.parameters():
                    param.requires_grad = False
                n_tracks = self.heads[modality].num_tracks
                if len(cell_types) != n_tracks:
                    raise ValueError(
                        f"{modality}: got {len(cell_types)} cell types for "
                        f"{n_tracks} tracks"
                    )
                modality_types = modality_types_by_modality.get(modality)
                if modality_types is None:
                    modality_types = [modality] * n_tracks
                if len(modality_types) != n_tracks:
                    raise ValueError(
                        f"{modality}: got {len(modality_types)} modality types for "
                        f"{n_tracks} tracks"
                    )
                cell_keys = [self.cell_key_by_type[cell_type] for cell_type in cell_types]
                modality_keys = [
                    self.modality_key_by_type[modality_type]
                    for modality_type in modality_types
                ]
                self.track_groups[modality] = CellTypeTrackGroups(
                    cell_keys, modality_keys
                )

    @staticmethod
    def _create_cell_layer(
        embedding_dim: int,
        cell_embedding_dim: int,
    ) -> CellTypeEmbedding:
        return CellTypeEmbedding(embedding_dim, cell_embedding_dim)

    @staticmethod
    def _create_modality_layer(
        embedding_dim: int,
        target_resolution: int,
        grad_checkpoint: bool = False,
    ) -> ModalityDecoder:
        return ModalityDecoder(embedding_dim, target_resolution, grad_checkpoint)

    def _forward_cell_modality_head(
        self,
        modality: str,
        embeddings: dict[int | str, torch.Tensor],
        organism_idx: torch.Tensor,
        *,
        return_scaled: bool,
        channels_last: bool,
    ) -> torch.Tensor:
        head = self.heads[modality]
        if 128 not in embeddings:
            raise KeyError("TFR fine-tuning heads require 128 bp embeddings")
        if modality not in self.track_groups:
            return head(
                embeddings,
                organism_idx,
                return_scaled=return_scaled,
                channels_last=channels_last,
            )[128]

        emb = embeddings[128]
        groups = self.track_groups[modality]

        cell_weights = torch.stack([
            self.cell_layers[cell_key].proj.weight.squeeze(-1)
            for cell_key in groups.cell_keys
        ])
        cell_biases = torch.stack([
            self.cell_layers[cell_key].proj.bias
            for cell_key in groups.cell_keys
        ])
        cell_embeddings = torch.einsum(
            "bcs,nhc->bnhs",
            emb,
            cell_weights.to(dtype=emb.dtype),
        )
        cell_embeddings = cell_embeddings + cell_biases.to(dtype=emb.dtype)[None, :, :, None]
        batch_size, n_cells, cell_dim, seq_len = cell_embeddings.shape
        encoder_skips = None
        if self.target_resolution == 32:
            encoder_skips = {
                ENCODER_SKIP_64BP_KEY: embeddings[ENCODER_SKIP_64BP_KEY],
                ENCODER_SKIP_32BP_KEY: embeddings[ENCODER_SKIP_32BP_KEY],
            }

        scaled_pred = None
        for modality_index, modality_key in enumerate(groups.modality_keys):
            track_indices = torch.nonzero(
                groups.track_modality_indices == modality_index,
                as_tuple=False,
            ).flatten()
            if track_indices.numel() == 0:
                continue
            modality_layer = self.modality_layers[modality_key]
            pred_by_cell = modality_layer(
                cell_embeddings.reshape(batch_size * n_cells, cell_dim, seq_len),
                encoder_skips=encoder_skips,
                n_cells=n_cells,
            )
            output_len = pred_by_cell.shape[-1]
            pred_by_cell = pred_by_cell.reshape(batch_size, n_cells, output_len)
            if scaled_pred is None:
                scaled_pred = pred_by_cell.new_empty(
                    batch_size,
                    head.num_tracks,
                    output_len,
                )
            scaled_pred[:, track_indices, :] = pred_by_cell[
                :, groups.track_cell_indices[track_indices], :
            ]

        if scaled_pred is None:
            raise RuntimeError(f"{modality}: no tracks assigned to modality decoders")
        residual_scale = head.residual_scales["128"][organism_idx]
        scaled_pred = F.softplus(scaled_pred) * F.softplus(residual_scale.unsqueeze(2))

        if channels_last:
            scaled_pred = scaled_pred.transpose(1, 2)
        if return_scaled:
            return scaled_pred
        return head.unscale(
            scaled_pred,
            organism_idx,
            self.target_resolution,
            channels_last,
        )

    def forward(
        self,
        embeddings: dict[int | str, torch.Tensor],
        organism_idx: torch.Tensor,
        *,
        return_scaled: bool,
    ) -> dict[str, torch.Tensor]:
        return {
            modality: self._forward_cell_modality_head(
                modality,
                embeddings,
                organism_idx,
                return_scaled=return_scaled,
                channels_last=True,
            )
            for modality in self.heads
        }


def setup_torchrun(device_arg: str) -> tuple[int, int, int, torch.device]:
    launched_with_torchrun = "RANK" in os.environ
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if device_arg == "auto":
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_arg)

    if device.type == "cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            raise RuntimeError(
                f"Requested {device}, but CUDA is not available to PyTorch. "
                "Use --device cpu or run on a GPU-visible node."
            )
        index = local_rank if device.index is None and world_size > 1 else (device.index or 0)
        if index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested cuda:{index}, but only {torch.cuda.device_count()} CUDA "
                "device(s) are visible."
            )
        torch.cuda.set_device(index)
        device = torch.device(f"cuda:{index}")

    if dist.is_available() and not dist.is_initialized() and launched_with_torchrun:
        backend = (
            "nccl"
            if device.type == "cuda" and torch.cuda.is_available()
            else "gloo"
        )
        if backend == "nccl":
            try:
                dist.init_process_group(backend=backend, device_id=device)
            except TypeError:
                dist.init_process_group(backend=backend)
        else:
            dist.init_process_group(backend=backend)

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()

    return rank, world_size, local_rank, device


def cleanup_torchrun() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def print_rank0(message: str, rank: int) -> None:
    if is_main_process(rank):
        print(message)


def broadcast_object(obj: Any, src: int = 0) -> Any:
    if not (dist.is_available() and dist.is_initialized()):
        return obj
    objects = [obj]
    dist.broadcast_object_list(objects, src=src)
    return objects[0]


def embedding_cache_file_manifest_path(
    cache_dir: Path,
    split: str,
    file_index: int,
) -> Path:
    return cache_dir / f"{split}.file{file_index:05d}.json"


def resolve_embedding_cache_dir(
    args: argparse.Namespace,
    modalities: list[str],
    world_size: int,
) -> Path:
    if args.embedding_cache_dir is not None:
        return args.embedding_cache_dir
    key_payload = {
        "data_dir": str(args.data_dir.resolve()),
        "pretrained_weights": (
            str(args.pretrained_weights.resolve())
            if args.pretrained_weights is not None
            else None
        ),
        "modalities": modalities,
        "pooling": args.pooling,
        "target_resolution": args.target_resolution,
        "dtype": args.dtype,
        "cache_dtype": args.embedding_cache_dtype,
    }
    cache_key = hashlib.sha1(
        json.dumps(key_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    return args.output_dir / "embedding_cache" / cache_key


def embedding_cache_expected_metadata(
    args: argparse.Namespace,
    dataset: BaskervilleMultiTFRecordDataset,
    split: str,
) -> dict[str, Any]:
    return {
        "format_version": 3,
        "split": split,
        "data_dir": str(args.data_dir.resolve()),
        "pretrained_weights": (
            str(args.pretrained_weights.resolve())
            if args.pretrained_weights is not None
            else None
        ),
        "modalities": list(dataset.modalities),
        "pooling": args.pooling,
        "target_resolution": args.target_resolution,
        "encoder_skips": args.target_resolution == 32,
        "model_dtype": args.dtype,
        "embedding_cache_dtype": args.embedding_cache_dtype,
        "target_cache": "forward_and_reverse_complement",
        "seq_length": dataset.metadata.seq_length,
        "target_length": dataset.metadata.target_length,
        "output_length": dataset.output_length,
        "crop_bins": dataset.prediction_crop_bins,
        "output_length_128bp": dataset.output_length_128bp,
        "crop_bins_128bp": dataset.prediction_crop_128bp,
    }


def embedding_cache_file_ready(
    manifest_path: Path,
    expected_metadata: dict[str, Any],
    *,
    source_file: Path,
    refresh: bool,
) -> bool:
    if refresh or not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text())
    manifest_metadata = {
        key: manifest.get(key)
        for key in expected_metadata
    }
    if manifest_metadata != expected_metadata:
        raise ValueError(
            f"Embedding cache manifest {manifest_path} does not match the "
            "requested data/model settings. Use --refresh-embedding-cache or "
            "choose a different --embedding-cache-dir."
        )
    if manifest.get("file_name") != source_file.name:
        raise ValueError(
            f"Embedding cache manifest {manifest_path} is for "
            f"{manifest.get('file_name')!r}, expected {source_file.name!r}."
        )
    chunks = manifest.get("chunks", [])
    return bool(chunks) and all(
        (manifest_path.parent / chunk["path"]).exists()
        for chunk in chunks
    )


def embedding_cache_ready(
    cache_dir: Path,
    expected_metadata: dict[str, Any],
    source_dataset: BaskervilleMultiTFRecordDataset,
    *,
    refresh: bool,
) -> bool:
    file_index_by_path = {
        path.resolve(): file_idx
        for file_idx, path in enumerate(source_dataset.files)
    }
    for source_file in source_dataset._worker_files():
        file_idx = file_index_by_path[source_file.resolve()]
        manifest_path = embedding_cache_file_manifest_path(
            cache_dir,
            source_dataset.split,
            file_idx,
        )
        if not embedding_cache_file_ready(
            manifest_path,
            expected_metadata,
            source_file=source_file,
            refresh=refresh,
        ):
            return False
    return True


def cache_storage_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported embedding cache dtype: {name}")


@torch.no_grad()
def compute_embedding_inputs(
    model: torch.nn.Module,
    sequences: torch.Tensor,
    organism_idx: torch.Tensor,
    *,
    use_amp: bool,
    include_encoder_skips: bool,
) -> dict[int | str, torch.Tensor]:
    amp_context = autocast_context(sequences.device, use_amp)
    with amp_context:
        if include_encoder_skips:
            sequences = model.dtype_policy.cast_to_compute(sequences)
            trunk, intermediates = model.encoder(sequences)
            skip_64bp = intermediates["bin_size_64"].detach()
            skip_32bp = intermediates["bin_size_32"].detach()
            trunk = trunk.transpose(1, 2)
            organism_embedding = model.organism_embed(organism_idx).unsqueeze(1)
            trunk = trunk + organism_embedding
            trunk, _pair_activations = model.tower(
                trunk,
                compute_dtype=model.dtype_policy.compute_dtype,
            )
            trunk_ncl = trunk.transpose(1, 2)
            embeddings_128bp = model.embedder_128bp(
                trunk_ncl,
                organism_idx,
                channels_last=False,
            )
            return {
                128: embeddings_128bp.detach(),
                ENCODER_SKIP_64BP_KEY: skip_64bp,
                ENCODER_SKIP_32BP_KEY: skip_32bp,
            }
        outputs = model(
            sequences,
            organism_idx,
            return_embeddings=True,
            resolutions=(128,),
            channels_last=False,
            embeddings_only=True,
        )
    return {128: outputs["embeddings_128bp"].detach()}


@torch.no_grad()
def precompute_embedding_cache(
    model: torch.nn.Module,
    dataset: BaskervilleMultiTFRecordDataset,
    args: argparse.Namespace,
    *,
    split: str,
    cache_dir: Path,
    device: torch.device,
    rank: int,
    world_size: int,
    num_workers: int,
) -> None:
    if args.embedding_cache_chunk_size < 1:
        raise ValueError(
            f"--embedding-cache-chunk-size must be >= 1, got "
            f"{args.embedding_cache_chunk_size}"
        )
    cache_dir.mkdir(parents=True, exist_ok=True)
    expected_metadata = embedding_cache_expected_metadata(
        args,
        dataset,
        split,
    )
    if embedding_cache_ready(
        cache_dir,
        expected_metadata,
        dataset,
        refresh=args.refresh_embedding_cache,
    ):
        print_rank0(
            f"Using existing {split} embedding cache at {cache_dir}",
            rank,
        )
        return

    storage_dtype = cache_storage_dtype(args.embedding_cache_dtype)
    target_resolution = args.target_resolution
    include_encoder_skips = target_resolution == 32

    file_index_by_path = {
        path.resolve(): file_idx
        for file_idx, path in enumerate(dataset.files)
    }
    source_files = dataset._worker_files()

    pbar = tqdm(
        desc=f"cache-{split}",
        disable=not is_main_process(rank),
        total=None,
    )

    def write_file_cache(source_file: Path, file_idx: int) -> None:
        manifest_path = embedding_cache_file_manifest_path(cache_dir, split, file_idx)
        if embedding_cache_file_ready(
            manifest_path,
            expected_metadata,
            source_file=source_file,
            refresh=args.refresh_embedding_cache,
        ):
            return

        chunks: list[dict[str, Any]] = []
        chunk_idx = 0
        examples_in_cache = 0
        pending_forward: dict[int | str, list[torch.Tensor]] = {128: []}
        pending_reverse: dict[int | str, list[torch.Tensor]] = {128: []}
        if include_encoder_skips:
            pending_forward[ENCODER_SKIP_64BP_KEY] = []
            pending_forward[ENCODER_SKIP_32BP_KEY] = []
            pending_reverse[ENCODER_SKIP_64BP_KEY] = []
            pending_reverse[ENCODER_SKIP_32BP_KEY] = []
        pending_targets: dict[str, list[torch.Tensor]] = {
            modality: []
            for modality in dataset.modalities
        }
        pending_targets_reverse: dict[str, list[torch.Tensor]] = {
            modality: []
            for modality in dataset.modalities
        }
        sequence_batch: list[torch.Tensor] = []
        target_batch: dict[str, list[torch.Tensor]] = {
            modality: []
            for modality in dataset.modalities
        }

        def flush_chunk() -> None:
            nonlocal chunk_idx, examples_in_cache
            if not pending_forward[128]:
                return
            forward_inputs = {
                key: torch.cat(parts, dim=0)
                for key, parts in pending_forward.items()
            }
            reverse_inputs = {
                key: torch.cat(parts, dim=0)
                for key, parts in pending_reverse.items()
            }
            embeddings_forward = forward_inputs[128]
            embeddings_reverse = reverse_inputs[128]
            targets = {
                modality: {target_resolution: torch.cat(parts, dim=0)}
                for modality, parts in pending_targets.items()
            }
            targets_reverse = {
                modality: {target_resolution: torch.cat(parts, dim=0)}
                for modality, parts in pending_targets_reverse.items()
            }
            chunk_name = (
                f"{split}.file{file_idx:05d}.chunk{chunk_idx:06d}.pt"
            )
            chunk_payload = {
                "embeddings_forward_128bp": embeddings_forward,
                "embeddings_reverse_128bp": embeddings_reverse,
                "targets": targets,
                "targets_reverse": targets_reverse,
            }
            if include_encoder_skips:
                chunk_payload.update({
                    "encoder_skip_forward_64bp": forward_inputs[ENCODER_SKIP_64BP_KEY],
                    "encoder_skip_reverse_64bp": reverse_inputs[ENCODER_SKIP_64BP_KEY],
                    "encoder_skip_forward_32bp": forward_inputs[ENCODER_SKIP_32BP_KEY],
                    "encoder_skip_reverse_32bp": reverse_inputs[ENCODER_SKIP_32BP_KEY],
                })
            torch.save(chunk_payload, cache_dir / chunk_name)
            chunks.append({
                "path": chunk_name,
                "num_examples": int(embeddings_forward.shape[0]),
                "embedding_shape": list(embeddings_forward.shape[1:]),
                "embedding_dtype": str(embeddings_forward.dtype).replace("torch.", ""),
                "encoder_skip_shapes": (
                    {
                        "64bp": list(forward_inputs[ENCODER_SKIP_64BP_KEY].shape[1:]),
                        "32bp": list(forward_inputs[ENCODER_SKIP_32BP_KEY].shape[1:]),
                    }
                    if include_encoder_skips
                    else None
                ),
                "target_resolution": target_resolution,
            })
            examples_in_cache += int(embeddings_forward.shape[0])
            chunk_idx += 1
            for parts in pending_forward.values():
                parts.clear()
            for parts in pending_reverse.values():
                parts.clear()
            for parts in pending_targets.values():
                parts.clear()
            for parts in pending_targets_reverse.values():
                parts.clear()

        def flush_model_batch() -> None:
            if not sequence_batch:
                return
            sequences = torch.stack(sequence_batch, dim=0).to(device, non_blocking=True)
            batch_size = sequences.shape[0]
            organism_idx = torch.full(
                (batch_size,),
                args.organism_idx,
                dtype=torch.long,
                device=device,
            )
            forward_inputs = compute_embedding_inputs(
                model,
                sequences,
                organism_idx,
                use_amp=not args.no_amp,
                include_encoder_skips=include_encoder_skips,
            )
            reverse_sequences = reverse_complement_onehot_tensor(sequences)
            reverse_inputs = compute_embedding_inputs(
                model,
                reverse_sequences,
                organism_idx,
                use_amp=not args.no_amp,
                include_encoder_skips=include_encoder_skips,
            )
            for key, tensor in forward_inputs.items():
                pending_forward[key].append(tensor.to("cpu", dtype=storage_dtype))
            for key, tensor in reverse_inputs.items():
                pending_reverse[key].append(tensor.to("cpu", dtype=storage_dtype))
            for modality in dataset.modalities:
                target = torch.stack(target_batch[modality], dim=0).detach().cpu()
                pending_targets[modality].append(target)
                pending_targets_reverse[modality].append(
                    reverse_complement_targets(
                        target,
                        dataset.strand_pair_by_modality[modality],
                    )
                )
                target_batch[modality].clear()
            sequence_batch.clear()
            if (
                sum(part.shape[0] for part in pending_forward[128])
                >= args.embedding_cache_chunk_size
            ):
                flush_chunk()
            pbar.update(batch_size)

        tf_dataset = dataset._tf_dataset([source_file])
        if tf_dataset is None:
            raise FileNotFoundError(f"No TFRecords found for cache file {source_file}")
        tf = dataset._ensure_tensorflow_cpu()
        feature_spec = {
            "sequence": tf.io.FixedLenFeature([], tf.string),
            "target": tf.io.FixedLenFeature([], tf.string),
        }
        for record in tf_dataset:
            parsed = tf.io.parse_single_example(record, feature_spec)
            sequence = dataset._decode_sequence(
                parsed["sequence"].numpy(),
                dataset.metadata.seq_length,
            )
            full_target = dataset._decode_full_target(parsed["target"].numpy())
            sequence_batch.append(torch.from_numpy(sequence).float())
            for modality, indices in dataset.target_indices_by_modality.items():
                target = full_target[:, indices].astype("float32", copy=False)
                target = dataset._target_for_resolution(target)
                target_batch[modality].append(torch.from_numpy(target).float())
            if len(sequence_batch) >= args.batch_size:
                flush_model_batch()
        flush_model_batch()
        flush_chunk()

        if examples_in_cache == 0:
            raise RuntimeError(f"No examples were cached for {source_file}")

        manifest = {
            **expected_metadata,
            "file_index": file_idx,
            "file_name": source_file.name,
            "num_examples": examples_in_cache,
            "chunks": chunks,
        }
        tmp_manifest_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
        tmp_manifest_path.write_text(json.dumps(manifest, indent=2))
        os.replace(tmp_manifest_path, manifest_path)

    for source_file in source_files:
        file_idx = file_index_by_path[source_file.resolve()]
        write_file_cache(source_file, file_idx)
    pbar.close()

    if not embedding_cache_ready(
        cache_dir,
        expected_metadata,
        dataset,
        refresh=False,
    ):
        raise RuntimeError(
            f"Embedding cache for split={split!r} was not fully written by rank {rank}"
        )


def unwrap_heads(heads_model: nn.Module) -> TFRHeads:
    if isinstance(heads_model, DDP):
        return heads_model.module  # type: ignore[return-value]
    return heads_model  # type: ignore[return-value]


@torch.no_grad()
def estimate_track_means(
    data_dir: Path,
    modalities: list[str],
    pooling: str,
    target_resolution: int,
    max_samples: int,
    batch_size: int,
    num_workers: int,
) -> dict[str, torch.Tensor]:
    dataset = create_tfr_dataset(
        data_dir,
        "train",
        modalities,
        pooling,
        target_resolution,
    )
    if max_samples <= 0:
        return {
            modality: torch.ones(1, len(dataset.target_indices_by_modality[modality]))
            for modality in dataset.modalities
        }

    loader = create_loader(dataset, batch_size=batch_size, num_workers=num_workers)
    sums = {
        modality: torch.zeros(len(indices), dtype=torch.float64)
        for modality, indices in dataset.target_indices_by_modality.items()
    }
    position_counts = {modality: 0 for modality in dataset.modalities}
    samples_seen = 0

    for sequences, modality_targets in loader:
        del sequences
        for modality, targets in modality_targets.items():
            target = targets[target_resolution].double()
            sums[modality] += target.sum(dim=(0, 1))
            position_counts[modality] += target.shape[0] * target.shape[1]
        samples_seen += next(iter(modality_targets.values()))[target_resolution].shape[0]
        if samples_seen >= max_samples:
            break

    return {
        modality: (
            sums[modality] / max(1, position_counts[modality])
        ).float().clamp_min(1e-6).unsqueeze(0)
        for modality in dataset.modalities
    }


def crop_predictions(pred: torch.Tensor, crop_bins: int) -> torch.Tensor:
    if crop_bins == 0:
        return pred
    if pred.shape[1] <= crop_bins * 2:
        raise ValueError(
            f"Cannot crop {crop_bins} bins from prediction length {pred.shape[1]}"
        )
    return pred[:, crop_bins:-crop_bins, :]


def unpack_tfr_batch(
    batch: tuple[torch.Tensor, dict[str, dict[int, torch.Tensor]]]
    | tuple[torch.Tensor, dict[str, dict[int, torch.Tensor]], dict[str, torch.Tensor]]
    | tuple[dict[int, torch.Tensor], dict[str, dict[int, torch.Tensor]]]
    | tuple[
        dict[int, torch.Tensor],
        dict[str, dict[int, torch.Tensor]],
        dict[str, torch.Tensor],
    ],
) -> tuple[
    torch.Tensor | dict[int, torch.Tensor],
    dict[str, dict[int, torch.Tensor]],
    dict[str, torch.Tensor] | None,
]:
    if len(batch) == 3:
        sequences, modality_targets, augmentation = batch
        return sequences, modality_targets, augmentation
    sequences, modality_targets = batch
    return sequences, modality_targets, None


def switch_reverse_predictions(
    pred: torch.Tensor,
    reverse_complement: torch.Tensor | None,
    strand_pair: torch.Tensor | None,
) -> torch.Tensor:
    """Match Baskerville SwitchReverse for channels-last 1D predictions."""
    if reverse_complement is None or not bool(reverse_complement.any()):
        return pred
    reversed_pred = torch.flip(pred, dims=[1])
    if strand_pair is not None:
        reversed_pred = reversed_pred.index_select(-1, strand_pair.to(pred.device))
    return torch.where(
        reverse_complement.to(pred.device).view(-1, 1, 1),
        reversed_pred,
        pred,
    )


def autocast_context(device: torch.device, use_amp: bool):
    if not use_amp or device.type != "cuda":
        return nullcontext()
    index = 0 if device.index is None else device.index
    major = torch.cuda.get_device_properties(index).major
    if major >= 8:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type="cuda", dtype=torch.float16)


def compute_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    loss_name: str,
    head: torch.nn.Module,
    organism_idx: torch.Tensor,
    target_resolution: int,
    positional_weight: float,
    count_weight: float,
) -> torch.Tensor:
    if loss_name == "mse":
        return F.mse_loss(pred, target)
    if loss_name == "poisson":
        return F.poisson_nll_loss(
            pred.clamp_min(1e-8),
            target,
            log_input=False,
            full=False,
            reduction="mean",
        )
    if loss_name in ("poisson-multinomial", "multinomial"):
        target_scaled = head.scale(
            target,
            organism_idx,
            resolution=target_resolution,
            channels_last=True,
        )
        mask = torch.ones(
            pred.shape[0],
            1,
            pred.shape[-1],
            dtype=torch.bool,
            device=pred.device,
        )
        loss_dict = multinomial_loss(
            y_pred=pred,
            y_true=target_scaled,
            mask=mask,
            multinomial_resolution=max(1, pred.shape[1] // 8),
            positional_weight=positional_weight,
            count_weight=count_weight,
            channels_last=True,
        )
        return loss_dict["loss"]
    raise ValueError(f"Unknown loss: {loss_name}")


def new_metric_stats(device: torch.device) -> torch.Tensor:
    # n, sum_pred, sum_true, sum_pred2, sum_true2, sum_pred_true, sum_squared_error.
    # The 7-element running accumulator stays float64 (56 bytes, free) so the
    # epoch-long sum-of-squares used for Pearson/R2 does not lose precision to
    # catastrophic cancellation; the per-step element math below runs in float32
    # to match the fp32 loss and avoid materializing full-size fp64 tensors.
    return torch.zeros(7, dtype=torch.float64, device=device)


@torch.no_grad()
def update_metric_stats(
    stats: torch.Tensor,
    pred: torch.Tensor,
    target: torch.Tensor,
) -> None:
    pred = pred.detach().float()
    target = target.detach().float()

    stats[0] += pred.numel()
    stats[1] += pred.sum(dtype=torch.float64)
    stats[2] += target.sum(dtype=torch.float64)
    stats[3] += pred.square().sum(dtype=torch.float64)
    stats[4] += target.square().sum(dtype=torch.float64)
    stats[5] += (pred * target).sum(dtype=torch.float64)
    stats[6] += (pred - target).square().sum(dtype=torch.float64)


def compute_regression_metrics(stats: torch.Tensor) -> dict[str, float]:
    n = stats[0].clamp_min(1.0)
    pred_var_sum = stats[3] - stats[1].square() / n
    true_var_sum = stats[4] - stats[2].square() / n
    covariance_sum = stats[5] - stats[1] * stats[2] / n

    denominator = torch.sqrt(pred_var_sum.clamp_min(0.0) * true_var_sum.clamp_min(0.0))
    if denominator <= 0:
        pearson = torch.tensor(float("nan"), device=stats.device)
    else:
        pearson = covariance_sum / denominator

    if true_var_sum <= 0:
        r2 = torch.tensor(float("nan"), device=stats.device)
    else:
        r2 = 1.0 - stats[6] / true_var_sum

    return {
        "pearson_r": pearson.item(),
        "r2": r2.item(),
    }


@torch.no_grad()
def compute_gradient_norm(module: nn.Module) -> torch.Tensor:
    total = None
    for param in module.parameters():
        if param.grad is None:
            continue
        grad_norm = param.grad.detach().float().norm(2).square()
        total = grad_norm if total is None else total + grad_norm
    if total is None:
        device = next(module.parameters()).device
        return torch.tensor(0.0, device=device)
    return total.sqrt()


def forward_heads(
    model: torch.nn.Module,
    heads_model: nn.Module,
    inputs: torch.Tensor | dict[int | str, torch.Tensor],
    organism_idx: torch.Tensor,
    crop_bins: int,
    use_amp: bool,
    return_scaled: bool,
    requires_grad: bool,
    reverse_complement: torch.Tensor | None = None,
    strand_pair_by_modality: dict[str, torch.Tensor] | None = None,
    head_organism_idx: torch.Tensor | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    if head_organism_idx is None:
        head_organism_idx = organism_idx
    if isinstance(inputs, dict):
        input_device = next(iter(inputs.values())).device
        embeddings = {resolution: tensor.detach() for resolution, tensor in inputs.items()}
    else:
        input_device = inputs.device
        include_encoder_skips = unwrap_heads(heads_model).target_resolution == 32
        with torch.no_grad():
            embeddings = compute_embedding_inputs(
                model,
                inputs,
                organism_idx,
                use_amp=use_amp,
                include_encoder_skips=include_encoder_skips,
            )
    amp_context = autocast_context(input_device, use_amp)
    with torch.set_grad_enabled(requires_grad):
        with amp_context:
            loss_predictions = heads_model(
                embeddings,
                head_organism_idx,
                return_scaled=return_scaled,
            )

    unwrapped_heads = unwrap_heads(heads_model)
    if return_scaled:
        experimental_predictions = {
            modality: unwrapped_heads.heads[modality].unscale(
                prediction,
                head_organism_idx,
                unwrapped_heads.target_resolution,
                channels_last=True,
            )
            for modality, prediction in loss_predictions.items()
        }
    else:
        experimental_predictions = loss_predictions

    switched_loss_predictions = {}
    switched_metric_predictions = {}
    for modality, prediction in experimental_predictions.items():
        strand_pair = None
        if strand_pair_by_modality is not None:
            strand_pair = strand_pair_by_modality.get(modality)
        switched_experimental = switch_reverse_predictions(
            prediction,
            reverse_complement,
            strand_pair,
        )
        switched_metric_predictions[modality] = switched_experimental.detach()
        if return_scaled:
            switched_loss_predictions[modality] = unwrapped_heads.heads[modality].scale(
                switched_experimental,
                head_organism_idx,
                unwrapped_heads.target_resolution,
                channels_last=True,
            )
        else:
            switched_loss_predictions[modality] = switched_experimental

    cropped_loss_predictions = {
        modality: crop_predictions(prediction, crop_bins)
        for modality, prediction in switched_loss_predictions.items()
    }
    cropped_metric_predictions = {
        modality: crop_predictions(prediction, crop_bins)
        for modality, prediction in switched_metric_predictions.items()
    }
    return cropped_loss_predictions, cropped_metric_predictions


def run_epoch(
    model: torch.nn.Module,
    heads_model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scheduler,
    args: argparse.Namespace,
    crop_bins: int,
    max_steps: int | None,
    rank: int,
    world_size: int,
) -> tuple[float, dict[str, float], dict[str, float]]:
    training = optimizer is not None
    heads_model.train(training)
    model.eval()
    if training and args.gradient_accumulation_steps < 1:
        raise ValueError("--gradient-accumulation-steps must be >= 1")
    total_loss = torch.tensor(0.0, device=device)
    total_steps = torch.tensor(0.0, device=device)
    grad_norm_sum = torch.tensor(0.0, device=device)
    grad_norm_count = torch.tensor(0.0, device=device)
    grad_norm_max = torch.tensor(0.0, device=device)
    metric_stats: dict[str, torch.Tensor] = {}
    modality_loss_sums: dict[str, torch.Tensor] = {}
    strand_pair_by_modality = {
        modality: torch.as_tensor(indices, dtype=torch.long, device=device)
        for modality, indices in loader.dataset.strand_pair_by_modality.items()
    }
    steps = 0
    accumulated_steps = 0
    pbar = tqdm(
        total=max_steps,
        desc="train" if training else "valid",
        disable=not is_main_process(rank),
    )
    heads = unwrap_heads(heads_model).heads
    data_iter = iter(loader)

    while max_steps is None or steps < max_steps:
        if args.debug_ranks and steps < 3:
            print(f"[rank{rank}] {pbar.desc} step={steps} next_batch start", flush=True)
        try:
            batch = next(data_iter)
            sequences, modality_targets, augmentation = unpack_tfr_batch(batch)
            has_batch = torch.tensor(1, device=device)
        except StopIteration:
            sequences = None
            modality_targets = None
            augmentation = None
            has_batch = torch.tensor(0, device=device)
        if args.debug_ranks and steps < 3:
            print(
                f"[rank{rank}] {pbar.desc} step={steps} "
                f"next_batch done has_batch={int(has_batch.item())}",
                flush=True,
            )

        if world_size > 1:
            if args.debug_ranks and steps < 3:
                print(f"[rank{rank}] {pbar.desc} step={steps} all_reduce start", flush=True)
            dist.all_reduce(has_batch, op=dist.ReduceOp.MIN)
            if args.debug_ranks and steps < 3:
                print(
                    f"[rank{rank}] {pbar.desc} step={steps} "
                    f"all_reduce done has_batch={int(has_batch.item())}",
                    flush=True,
                )
        if has_batch.item() == 0:
            break
        if sequences is None or modality_targets is None:
            raise RuntimeError("Local loader is exhausted but distributed batch check passed")

        if isinstance(sequences, dict):
            model_inputs: torch.Tensor | dict[int | str, torch.Tensor] = {
                key: tensor.to(device, non_blocking=True)
                for key, tensor in sequences.items()
            }
            batch_size = next(iter(model_inputs.values())).shape[0]
        else:
            model_inputs = sequences.to(device, non_blocking=True)
            batch_size = model_inputs.shape[0]
        organism_idx = torch.full(
            (batch_size,),
            args.organism_idx,
            dtype=torch.long,
            device=device,
        )
        head_organism_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
        reverse_complement = None
        if augmentation is not None:
            reverse_complement = augmentation["reverse_complement"].to(
                device,
                non_blocking=True,
            )

        predictions_by_modality, metric_predictions_by_modality = forward_heads(
            model,
            heads_model,
            model_inputs,
            organism_idx,
            crop_bins,
            use_amp=not args.no_amp,
            return_scaled=args.loss in ("poisson-multinomial", "multinomial"),
            requires_grad=training,
            reverse_complement=reverse_complement,
            strand_pair_by_modality=strand_pair_by_modality,
            head_organism_idx=head_organism_idx,
        )
        loss = torch.tensor(0.0, device=device)
        loss_by_modality = {}
        for modality, pred in predictions_by_modality.items():
            target = modality_targets[modality][args.target_resolution].to(
                device,
                non_blocking=True,
            )
            if pred.shape != target.shape:
                raise ValueError(
                    f"{modality}: prediction shape {tuple(pred.shape)} does not "
                    f"match target shape {tuple(target.shape)} after crop_bins={crop_bins}"
                )
            modality_loss = compute_loss(
                pred,
                target,
                loss_name=args.loss,
                head=heads[modality],
                organism_idx=head_organism_idx,
                target_resolution=args.target_resolution,
                positional_weight=args.positional_weight,
                count_weight=args.count_weight,
            )
            loss = loss + modality_loss
            loss_by_modality[modality] = float(modality_loss.detach())
            if modality not in modality_loss_sums:
                modality_loss_sums[modality] = torch.tensor(0.0, device=device)
            modality_loss_sums[modality] += modality_loss.detach()
            if modality not in metric_stats:
                metric_stats[modality] = new_metric_stats(device)
            update_metric_stats(
                metric_stats[modality],
                metric_predictions_by_modality[modality],
                target,
            )

        if training:
            if accumulated_steps == 0:
                optimizer.zero_grad(set_to_none=True)
            (loss / args.gradient_accumulation_steps).backward()
            accumulated_steps += 1
            if accumulated_steps == args.gradient_accumulation_steps:
                grad_norm = compute_gradient_norm(heads_model)
                grad_norm_sum += grad_norm
                grad_norm_count += 1
                grad_norm_max = torch.maximum(grad_norm_max, grad_norm)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                accumulated_steps = 0

        total_loss += loss.detach()
        total_steps += 1
        steps += 1
        postfix = {"loss": f"{loss.item():.4f}"}
        if optimizer is not None:
            postfix["lr"] = f"{optimizer.param_groups[0]['lr']:.2e}"
        postfix.update({
            modality: f"{modality_loss:.4f}"
            for modality, modality_loss in loss_by_modality.items()
        })
        pbar.set_postfix(postfix)
        pbar.update(1)

    pbar.close()

    if training and accumulated_steps > 0:
        grad_norm = compute_gradient_norm(heads_model)
        grad_norm_sum += grad_norm
        grad_norm_count += 1
        grad_norm_max = torch.maximum(grad_norm_max, grad_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    if world_size > 1:
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_steps, op=dist.ReduceOp.SUM)
        dist.all_reduce(grad_norm_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(grad_norm_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(grad_norm_max, op=dist.ReduceOp.MAX)
        for modality_loss_sum in modality_loss_sums.values():
            dist.all_reduce(modality_loss_sum, op=dist.ReduceOp.SUM)
        for stats in metric_stats.values():
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)

    metrics: dict[str, float] = {}
    grad_metrics: dict[str, float] = {}
    if training and grad_norm_count.item() > 0:
        grad_metrics["grad_norm"] = (grad_norm_sum / grad_norm_count).item()
        grad_metrics["grad_norm_max"] = grad_norm_max.item()
    for modality, modality_loss_sum in modality_loss_sums.items():
        metrics[f"{modality}_loss"] = (
            modality_loss_sum / total_steps.clamp_min(1)
        ).item()
    for modality, stats in metric_stats.items():
        modality_metrics = compute_regression_metrics(stats)
        metrics[f"{modality}_pearson_r"] = modality_metrics["pearson_r"]
        metrics[f"{modality}_r2"] = modality_metrics["r2"]
    pearson_values = [
        value for key, value in metrics.items()
        if key.endswith("_pearson_r") and not math.isnan(value)
    ]
    r2_values = [
        value for key, value in metrics.items()
        if key.endswith("_r2") and not math.isnan(value)
    ]
    if pearson_values:
        metrics["mean_pearson_r"] = sum(pearson_values) / len(pearson_values)
    if r2_values:
        metrics["mean_r2"] = sum(r2_values) / len(r2_values)

    return (total_loss / total_steps.clamp_min(1)).item(), metrics, grad_metrics


def save_checkpoint(
    path: Path,
    heads_model: nn.Module,
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    unwrapped = unwrap_heads(heads_model)
    torch.save(
        {
            "heads_state_dict": unwrapped.state_dict(),
            "modality_heads_state_dict": unwrapped.heads.state_dict(),
            "heads_model_state_dict": unwrapped.state_dict(),
            "cell_layers_state_dict": unwrapped.cell_layers.state_dict(),
            "modality_layers_state_dict": unwrapped.modality_layers.state_dict(),
            "metadata": metadata,
        },
        path,
    )


def format_metrics(metrics: dict[str, float], prefix: str) -> str:
    if not metrics:
        return ""
    return " ".join(
        f"{prefix}_{key}={value:.6f}"
        for key, value in sorted(metrics.items())
    )


def create_wandb_run(
    args: argparse.Namespace,
    rank: int,
    run_name: str,
    config: dict[str, Any],
):
    if not args.wandb or not is_main_process(rank):
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "W&B logging requires the `wandb` package. Install it or omit --wandb."
        ) from exc

    tags = None
    if args.wandb_tags:
        tags = [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]

    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name or run_name,
        config=config,
        tags=tags,
    )


def log_wandb(run, metrics: dict[str, float], step: int) -> None:
    if run is not None:
        run.log(metrics, step=step)


def finish_wandb(run) -> None:
    if run is not None:
        run.finish()


def resolve_modalities(args: argparse.Namespace) -> list[str]:
    requested = args.modality or ["all"]
    if len(requested) == 1 and requested[0].lower() == "all":
        return BaskervilleTFRecordDataset.available_modalities(args.data_dir)
    if any(modality.lower() == "all" for modality in requested):
        raise ValueError("Use either --modality all or explicit repeated --modality values")
    return requested


def download_hf_weights(args: argparse.Namespace) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "Downloading AlphaGenome weights from Hugging Face requires "
            "`huggingface_hub`. Install it or pass --pretrained-weights."
        ) from exc

    try:
        weights_path = hf_hub_download(
            repo_id=args.hf_model_id,
            filename=args.hf_filename,
            revision=args.hf_revision,
            repo_type="model",
            cache_dir=str(args.hf_cache_dir) if args.hf_cache_dir is not None else None,
            token=args.hf_token,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to download Hugging Face weights "
            f"{args.hf_model_id!r}/{args.hf_filename!r}. "
            "Pass --hf-token if the repo requires authentication, or pass "
            "--pretrained-weights with a local checkpoint."
        ) from exc
    return Path(weights_path)


def resolve_pretrained_weights(
    args: argparse.Namespace,
    rank: int,
) -> Path:
    if args.pretrained_weights is not None:
        return args.pretrained_weights

    resolved: Path | None = None
    if is_main_process(rank):
        print(
            "No --pretrained-weights supplied; downloading base checkpoint "
            f"{args.hf_model_id}/{args.hf_filename}."
        )
        resolved = download_hf_weights(args)
        print(f"Using Hugging Face PyTorch weights: {resolved}")

    resolved = broadcast_object(str(resolved), src=0)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    return Path(resolved)


def main() -> None:
    args = parse_args()
    if args.precompute_embeddings and args.augment_shift != 0:
        raise SystemExit(
            "--precompute-embeddings caches original and reverse-complement "
            "embeddings only; use --augment-shift 0."
        )
    torch.backends.cuda.matmul.allow_tf32 = True
    rank, world_size, local_rank, device = setup_torchrun(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    wandb_run = None
    try:
        modalities = resolve_modalities(args)
        print_rank0(f"Distributed: rank={rank} world_size={world_size}", rank)
        print_rank0(f"Seed: {args.seed}", rank)

        train_dataset = create_tfr_dataset(
            args.data_dir,
            "train",
            modalities,
            args.pooling,
            args.target_resolution,
            repeat=False,
            shuffle_files=True,
            num_parallel_reads=args.tfr_num_parallel_reads,
            rank=rank,
            world_size=world_size,
            augment_rc=args.augment_rc,
            augment_shift=args.augment_shift,
        )
        val_dataset = create_tfr_dataset(
            args.data_dir,
            "valid",
            modalities,
            args.pooling,
            args.target_resolution,
            num_parallel_reads=args.tfr_num_parallel_reads,
            rank=rank,
            world_size=world_size,
        )

        train_loader = create_loader(
            train_dataset,
            args.batch_size,
            args.num_workers,
            prefetch_n=args.prefetch_n,
            persistent_workers=not args.precompute_embeddings,
        )
        val_loader = create_loader(
            val_dataset,
            args.batch_size,
            args.val_num_workers,
            prefetch_n=args.prefetch_n,
            persistent_workers=False,
        )
        print_rank0(
            " ".join(
                (
                    f"TFRecord files/rank: train={len(train_dataset._worker_files())}",
                    f"valid={len(val_dataset._worker_files())}",
                    f"DataLoader workers: train={train_loader.num_workers}",
                    f"valid={val_loader.num_workers}",
                    f"prefetch_n={args.prefetch_n}",
                    f"tf_parallel_reads={args.tfr_num_parallel_reads}",
                )
            ),
            rank,
        )

        sample_sequences, sample_targets, sample_augmentation = unpack_tfr_batch(
            next(iter(train_loader))
        )
        print_rank0(
            "Loader OK: "
            f"seq={tuple(sample_sequences.shape)} "
            + "targets="
            + ",".join(
                f"{modality}:{tuple(targets[args.target_resolution].shape)}"
                for modality, targets in sample_targets.items()
            )
            + f" crop_bins={train_dataset.prediction_crop_bins}"
            + f" target_resolution={args.target_resolution}"
            + f" modalities={modalities}",
            rank,
        )
        if sample_augmentation is not None:
            print_rank0(
                "Augmentation: "
                f"rc={args.augment_rc} "
                f"shift={args.augment_shift} "
                f"sample_rc={sample_augmentation['reverse_complement'].tolist()} "
                f"sample_shift={sample_augmentation['shift'].tolist()}",
                rank,
            )
        if args.loader_only:
            return

        pretrained_weights_arg = args.pretrained_weights
        pretrained_weights = resolve_pretrained_weights(args, rank)
        args.pretrained_weights = pretrained_weights

        print_rank0(f"Device: {device} local_rank={local_rank}", rank)
        dtype_policy = (
            DtypePolicy.full_float32()
            if args.dtype == "float32"
            else DtypePolicy.mixed_precision()
        )

        track_means = None
        if is_main_process(rank):
            print("Estimating track means...")
            track_means = estimate_track_means(
                args.data_dir,
                modalities,
                args.pooling,
                args.target_resolution,
                args.track_means_samples,
                args.batch_size,
                args.num_workers,
            )
            print(
                "Track means:",
                ", ".join(
                    f"{modality}={means.mean().item():.6g}"
                    for modality, means in track_means.items()
                ),
            )
        track_means = broadcast_object(track_means, src=0)

        model = AlphaGenome(dtype_policy=dtype_policy)
        model = load_trunk(model, str(pretrained_weights), exclude_heads=True)
        model = remove_all_heads(model).to(device)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        embedding_cache_dir = None
        if args.precompute_embeddings:
            embedding_cache_dir = resolve_embedding_cache_dir(args, modalities, world_size)
            print_rank0(f"Embedding cache: {embedding_cache_dir}", rank)
            cache_train_source = create_tfr_dataset(
                args.data_dir,
                "train",
                modalities,
                args.pooling,
                args.target_resolution,
                repeat=False,
                shuffle_files=False,
                num_parallel_reads=args.tfr_num_parallel_reads,
                rank=rank,
                world_size=world_size,
                augment_rc=False,
                augment_shift=0,
            )
            cache_val_source = create_tfr_dataset(
                args.data_dir,
                "valid",
                modalities,
                args.pooling,
                args.target_resolution,
                num_parallel_reads=args.tfr_num_parallel_reads,
                rank=rank,
                world_size=world_size,
                augment_rc=False,
                augment_shift=0,
            )
            precompute_embedding_cache(
                model,
                cache_train_source,
                args,
                split="train",
                cache_dir=embedding_cache_dir,
                device=device,
                rank=rank,
                world_size=world_size,
                num_workers=args.num_workers,
            )
            precompute_embedding_cache(
                model,
                cache_val_source,
                args,
                split="valid",
                cache_dir=embedding_cache_dir,
                device=device,
                rank=rank,
                world_size=world_size,
                num_workers=args.val_num_workers,
            )
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
            train_dataset = CachedEmbeddingTFRecordDataset(
                embedding_cache_dir,
                split="train",
                source_dataset=cache_train_source,
                rank=rank,
                world_size=world_size,
                augment_rc=args.augment_rc,
                shuffle_files=True,
                seed=args.seed,
            )
            val_dataset = CachedEmbeddingTFRecordDataset(
                embedding_cache_dir,
                split="valid",
                source_dataset=cache_val_source,
                rank=rank,
                world_size=world_size,
                augment_rc=False,
                shuffle_files=False,
            )
            train_loader = create_cached_embedding_loader(
                train_dataset,
                args.batch_size,
                args.num_workers,
                prefetch_n=args.prefetch_n,
                persistent_workers=True,
            )
            val_loader = create_cached_embedding_loader(
                val_dataset,
                args.batch_size,
                args.val_num_workers,
                prefetch_n=args.prefetch_n,
                persistent_workers=False,
            )
            print_rank0(
                " ".join(
                    (
                        "Cached embedding loaders:",
                        f"train_chunks={len(train_dataset.chunk_files)}",
                        f"valid_chunks={len(val_dataset.chunk_files)}",
                        f"train_workers={train_loader.num_workers}",
                        f"valid_workers={val_loader.num_workers}",
                    )
                ),
                rank,
            )
            if device.type == "cuda":
                model.to("cpu")
                torch.cuda.empty_cache()

        heads = {}
        for modality in modalities:
            heads[modality] = create_finetuning_head(
                assay_type=train_dataset.assay_type_by_modality[modality],
                n_tracks=len(train_dataset.target_indices_by_modality[modality]),
                resolutions=(128,),
                num_organisms=1,
                track_means=track_means[modality],
            )
        cell_types_by_modality = cell_types_from_target_rows({
            modality: train_dataset.target_rows_by_modality[modality]
            for modality in modalities
        })
        modality_types_by_modality = modality_types_from_target_rows({
            modality: train_dataset.target_rows_by_modality[modality]
            for modality in modalities
        })
        unique_ct_values = {
            row["ct"]
            for rows in train_dataset.target_rows_by_modality.values()
            for row in rows
            if row.get("ct")
        }
        strand_specific_modality_types = {
            modality_type
            for modality_types in modality_types_by_modality.values()
            for modality_type in modality_types
            if modality_type.endswith(("+", "-"))
        }
        unique_modality_types = {
            modality_type
            for modality_types in modality_types_by_modality.values()
            for modality_type in modality_types
        }
        heads_model: nn.Module = TFRHeads(
            heads,
            cell_types_by_modality,
            modality_types_by_modality,
            cell_embedding_dim=args.cell_embedding_dim,
            target_resolution=args.target_resolution,
            grad_checkpoint=args.grad_checkpoint,
        ).to(device)
        print_rank0(
            "Cell layers: "
            f"{len(unwrap_heads(heads_model).cell_layers)} shared ct transforms "
            f"({len(unique_ct_values)} unique ct values); "
            f"modality layers={len(unwrap_heads(heads_model).modality_layers)} "
            f"({len(unique_modality_types)} track modality types; "
            f"{len(strand_specific_modality_types)} strand-specific); "
            f"cell_embedding_dim={args.cell_embedding_dim}",
            rank,
        )
        if world_size > 1:
            device_ids = [device.index] if device.type == "cuda" else None
            heads_model = DDP(
                heads_model,
                device_ids=device_ids,
                gradient_as_bucket_view=True,
                static_graph=True,
            )

        optimizer = torch.optim.AdamW(
            heads_model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        total_optimizer_steps = estimate_total_optimizer_steps(
            train_dataset,
            args.batch_size,
            args.epochs,
            args.gradient_accumulation_steps,
            args.max_train_steps,
        )
        if args.warmup_fraction is not None:
            if not 0.0 <= args.warmup_fraction <= 1.0:
                raise ValueError(
                    "warmup_fraction must be between 0 and 1, got "
                    f"{args.warmup_fraction}"
                )
            warmup_steps = math.ceil(
                total_optimizer_steps * args.warmup_fraction
            )
        else:
            warmup_steps = resolve_warmup_steps(
                args.warmup_steps, total_optimizer_steps
            )
        scheduler = create_lr_scheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_optimizer_steps,
            schedule=args.lr_schedule,
        )
        print_rank0(
            " ".join(
                (
                    f"LR schedule: {args.lr_schedule}",
                    f"warmup_steps={warmup_steps}",
                    f"estimated_optimizer_steps={total_optimizer_steps}",
                    f"gradient_accumulation_steps={args.gradient_accumulation_steps}",
                    f"weight_decay={args.weight_decay}",
                )
            ),
            rank,
        )

        if args.run_name is not None:
            run_name = args.run_name
        else:
            generated_run_name = (
                datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                if is_main_process(rank)
                else None
            )
            run_name = broadcast_object(generated_run_name, src=0)
        output_dir = args.output_dir / run_name
        if is_main_process(rank):
            output_dir.mkdir(parents=True, exist_ok=True)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        metadata = {
            "data_dir": str(args.data_dir),
            "modalities": modalities,
            "assay_types": train_dataset.assay_type_by_modality,
            "pooling": args.pooling,
            "target_resolution": args.target_resolution,
            "n_tracks": {
                modality: len(train_dataset.target_indices_by_modality[modality])
                for modality in modalities
            },
            "track_names": {
                modality: [
                    row.get("identifier", "")
                    for row in train_dataset.target_rows_by_modality[modality]
                ]
                for modality in modalities
            },
            "strand_pair": {
                modality: train_dataset.strand_pair_by_modality[modality].tolist()
                for modality in modalities
            },
            "cell_types": cell_types_by_modality,
            "modality_types": modality_types_by_modality,
            "n_cell_types": len(unwrap_heads(heads_model).cell_layers),
            "cell_embedding_dim": args.cell_embedding_dim,
            "n_modality_layers": len(unwrap_heads(heads_model).modality_layers),
            "head_factorization": (
                "cell_bottleneck_linear_then_encoder_skip_unet32bp_modality_mlp"
                if args.target_resolution == 32
                else "cell_bottleneck_linear_then_modality_mlp"
            ),
            "crop_bins": train_dataset.prediction_crop_bins,
            "target_length": train_dataset.output_length,
            "crop_bins_128bp": train_dataset.prediction_crop_128bp,
            "target_length_128bp": train_dataset.output_length_128bp,
            "loss": args.loss,
            "organism_idx": args.organism_idx,
            "pretrained_weights": str(pretrained_weights),
            "pretrained_weights_source": (
                "local" if pretrained_weights_arg is not None else "huggingface"
            ),
            "hf_model_id": None if pretrained_weights_arg is not None else args.hf_model_id,
            "hf_filename": None if pretrained_weights_arg is not None else args.hf_filename,
            "hf_revision": None if pretrained_weights_arg is not None else args.hf_revision,
            "world_size": world_size,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "val_num_workers": args.val_num_workers,
            "prefetch_n": args.prefetch_n,
            "epochs": args.epochs,
            "seed": args.seed,
            "lr": args.lr,
            "lr_schedule": args.lr_schedule,
            "warmup_steps": warmup_steps,
            "warmup_steps_arg": args.warmup_steps,
            "warmup_fraction": args.warmup_fraction,
            "estimated_optimizer_steps": total_optimizer_steps,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "weight_decay": args.weight_decay,
            "dtype": args.dtype,
            "use_amp": not args.no_amp,
            "grad_checkpoint": args.grad_checkpoint,
            "augment_rc": args.augment_rc,
            "augment_shift": args.augment_shift,
            "precompute_embeddings": args.precompute_embeddings,
            "embedding_cache_dir": (
                str(embedding_cache_dir)
                if embedding_cache_dir is not None
                else None
            ),
            "embedding_cache_dtype": args.embedding_cache_dtype,
            "embedding_cache_chunk_size": args.embedding_cache_chunk_size,
            "best_checkpoint_metric": "val/mean_pearson_r",
        }
        if is_main_process(rank):
            (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        wandb_run = create_wandb_run(args, rank, run_name, metadata)

        best_val_mean_pearson_r = float("-inf")
        best_epoch = None
        for epoch in range(1, args.epochs + 1):
            print_rank0(f"Epoch {epoch}/{args.epochs}", rank)
            train_loss, train_metrics, train_grad_metrics = run_epoch(
                model,
                heads_model,
                train_loader,
                device,
                optimizer,
                scheduler,
                args,
                train_dataset.prediction_crop_bins,
                args.max_train_steps,
                rank,
                world_size,
            )
            val_loss, val_metrics, _ = run_epoch(
                model,
                heads_model,
                val_loader,
                device,
                None,
                None,
                args,
                val_dataset.prediction_crop_bins,
                args.max_val_steps,
                rank,
                world_size,
            )
            epoch_log = {
                "epoch": epoch,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "lr": optimizer.param_groups[0]["lr"],
            }
            epoch_log.update({
                f"train/{key}": value
                for key, value in train_metrics.items()
            })
            epoch_log.update({
                f"train/{key}": value
                for key, value in train_grad_metrics.items()
            })
            epoch_log.update({
                f"val/{key}": value
                for key, value in val_metrics.items()
            })
            current_val_mean_pearson_r = val_metrics.get("mean_pearson_r", float("nan"))
            is_best = (
                math.isfinite(current_val_mean_pearson_r)
                and current_val_mean_pearson_r > best_val_mean_pearson_r
            )
            if is_best:
                best_val_mean_pearson_r = current_val_mean_pearson_r
                best_epoch = epoch
            if best_epoch is not None:
                epoch_log["best/val_mean_pearson_r"] = best_val_mean_pearson_r
                epoch_log["best/epoch"] = best_epoch
            log_wandb(wandb_run, epoch_log, step=epoch)
            print_rank0(
                " ".join(
                    part for part in (
                        f"epoch={epoch}",
                        f"train_loss={train_loss:.6f}",
                        format_metrics(train_metrics, "train"),
                        f"val_loss={val_loss:.6f}",
                        format_metrics(val_metrics, "val"),
                    )
                    if part
                ),
                rank,
            )
            if is_main_process(rank):
                epoch_metadata = {
                    **metadata,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "best_checkpoint_metric": "val/mean_pearson_r",
                    "best_val_mean_pearson_r": best_val_mean_pearson_r,
                    "best_epoch": best_epoch,
                }
                save_checkpoint(
                    output_dir / "last_heads.pt",
                    heads_model,
                    {**epoch_metadata, "checkpoint_kind": "last"},
                )
                if is_best:
                    save_checkpoint(
                        output_dir / "best_heads.pt",
                        heads_model,
                        {**epoch_metadata, "checkpoint_kind": "best"},
                    )
                    print_rank0(
                        (
                            "Saved best_heads.pt "
                            f"epoch={epoch} "
                            f"val_mean_pearson_r={best_val_mean_pearson_r:.6f}"
                        ),
                        rank,
                    )
    finally:
        finish_wandb(wandb_run)
        cleanup_torchrun()


if __name__ == "__main__":
    main()
