#!/usr/bin/env python
"""Fine-tune frozen AlphaGenome 128 bp heads from Baskerville TFRecords."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a new 128 bp AlphaGenome head from Baskerville TFRecords.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--pretrained-weights", type=Path)
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
    parser.add_argument("--epochs", type=int, default=1)
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
    parser.add_argument("--output-dir", type=Path, default=Path("finetuning_output/tfr_heads"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', 'cuda', or e.g. 'cuda:0'")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
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
    *,
    repeat: bool = False,
    shuffle_files: bool = False,
    num_parallel_reads: int | None = 1,
    rank: int = 0,
    world_size: int = 1,
) -> BaskervilleMultiTFRecordDataset:
    return BaskervilleMultiTFRecordDataset(
        data_dir,
        split=split,  # type: ignore[arg-type]
        modalities=modalities,
        pooling=pooling,  # type: ignore[arg-type]
        repeat=repeat,
        shuffle_files=shuffle_files,
        num_parallel_reads=num_parallel_reads,
        rank=rank,
        world_size=world_size,
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


def estimate_local_batches(
    dataset: BaskervilleMultiTFRecordDataset,
    batch_size: int,
    max_steps: int | None,
) -> int:
    if max_steps is not None:
        return max_steps
    local_examples = math.ceil(len(dataset) / dataset.world_size)
    return max(1, math.ceil(local_examples / batch_size))


def estimate_total_optimizer_steps(
    dataset: BaskervilleMultiTFRecordDataset,
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


def cell_types_from_target_rows(
    target_rows_by_modality: dict[str, list[dict[str, str]]],
) -> dict[str, list[str]]:
    cell_types_by_modality = {}
    for modality, rows in target_rows_by_modality.items():
        cell_types_by_modality[modality] = [
            row.get("ct") or row.get("identifier") or f"{modality}_{idx}"
            for idx, row in enumerate(rows)
        ]
    return cell_types_by_modality


class CellTypeTrackGroups(nn.Module):
    """Track-index buffers for one modality, grouped by shared cell type."""

    def __init__(self, cell_keys: list[str]):
        super().__init__()
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
        self.register_buffer(
            "track_cell_indices",
            torch.tensor(
                [cell_index_by_key[cell_key] for cell_key in cell_keys],
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

    def __init__(self, embedding_dim: int):
        super().__init__()
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TFRHeads(nn.Module):
    """DDP-friendly wrapper with shared cell-type layers and modality heads."""

    def __init__(
        self,
        heads: dict[str, nn.Module],
        cell_types_by_modality: dict[str, list[str]] | None = None,
        embedding_dim: int = EMBEDDING_128BP_DIM,
        cell_embedding_dim: int = 16,
    ):
        super().__init__()
        if cell_embedding_dim < 1:
            raise ValueError(
                f"cell_embedding_dim must be >= 1, got {cell_embedding_dim}"
            )
        self.heads = nn.ModuleDict(heads)
        self.modality_layers = nn.ModuleDict({
            modality: self._create_modality_layer(cell_embedding_dim)
            for modality in heads
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
                cell_keys = [self.cell_key_by_type[cell_type] for cell_type in cell_types]
                self.track_groups[modality] = CellTypeTrackGroups(cell_keys)

    @staticmethod
    def _create_cell_layer(
        embedding_dim: int,
        cell_embedding_dim: int,
    ) -> CellTypeEmbedding:
        return CellTypeEmbedding(embedding_dim, cell_embedding_dim)

    @staticmethod
    def _create_modality_layer(embedding_dim: int) -> ModalityDecoder:
        return ModalityDecoder(embedding_dim)

    def _forward_cell_modality_head(
        self,
        modality: str,
        embeddings: dict[int, torch.Tensor],
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
        modality_layer = self.modality_layers[modality]

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
        pred_by_cell = modality_layer(
            cell_embeddings.reshape(batch_size * n_cells, cell_dim, seq_len)
        )
        pred_by_cell = pred_by_cell.reshape(batch_size, n_cells, seq_len)

        scaled_pred = pred_by_cell[:, groups.track_cell_indices, :]
        residual_scale = head.residual_scales["128"][organism_idx]
        scaled_pred = F.softplus(scaled_pred) * F.softplus(residual_scale.unsqueeze(2))

        if channels_last:
            scaled_pred = scaled_pred.transpose(1, 2)
        if return_scaled:
            return scaled_pred
        return head.unscale(scaled_pred, organism_idx, 128, channels_last)

    def forward(
        self,
        embeddings: dict[int, torch.Tensor],
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


def unwrap_heads(heads_model: nn.Module) -> TFRHeads:
    if isinstance(heads_model, DDP):
        return heads_model.module  # type: ignore[return-value]
    return heads_model  # type: ignore[return-value]


@torch.no_grad()
def estimate_track_means(
    data_dir: Path,
    modalities: list[str],
    pooling: str,
    max_samples: int,
    batch_size: int,
    num_workers: int,
) -> dict[str, torch.Tensor]:
    dataset = create_tfr_dataset(data_dir, "train", modalities, pooling)
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
            target = targets[128].double()
            sums[modality] += target.sum(dim=(0, 1))
            position_counts[modality] += target.shape[0] * target.shape[1]
        samples_seen += next(iter(modality_targets.values()))[128].shape[0]
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
        target_scaled = head.scale(target, organism_idx, resolution=128, channels_last=True)
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
    # n, sum_pred, sum_true, sum_pred2, sum_true2, sum_pred_true, sum_squared_error
    return torch.zeros(7, dtype=torch.float64, device=device)


@torch.no_grad()
def update_metric_stats(
    stats: torch.Tensor,
    pred: torch.Tensor,
    target: torch.Tensor,
) -> None:
    pred = pred.detach().double()
    target = target.detach().double()
    diff = pred - target

    stats[0] += pred.numel()
    stats[1] += pred.sum()
    stats[2] += target.sum()
    stats[3] += pred.square().sum()
    stats[4] += target.square().sum()
    stats[5] += (pred * target).sum()
    stats[6] += diff.square().sum()


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
    sequences: torch.Tensor,
    organism_idx: torch.Tensor,
    crop_bins: int,
    use_amp: bool,
    return_scaled: bool,
    requires_grad: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    amp_context = autocast_context(sequences.device, use_amp)
    with torch.no_grad():
        with amp_context:
            outputs = model(
                sequences,
                organism_idx,
                return_embeddings=True,
                resolutions=(128,),
                channels_last=False,
                embeddings_only=True,
            )
            embeddings = {128: outputs["embeddings_128bp"].detach()}
    with torch.set_grad_enabled(requires_grad):
        with amp_context:
            loss_predictions = heads_model(
                embeddings,
                organism_idx,
                return_scaled=return_scaled,
            )

    with torch.no_grad():
        if not return_scaled:
            metric_predictions = loss_predictions
        else:
            with amp_context:
                metric_predictions = unwrap_heads(heads_model)(
                    embeddings,
                    organism_idx,
                    return_scaled=False,
                )

    cropped_loss_predictions = {
        modality: crop_predictions(prediction, crop_bins)
        for modality, prediction in loss_predictions.items()
    }
    cropped_metric_predictions = {
        modality: crop_predictions(prediction, crop_bins)
        for modality, prediction in metric_predictions.items()
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
            sequences, modality_targets = next(data_iter)
            has_batch = torch.tensor(1, device=device)
        except StopIteration:
            sequences = None
            modality_targets = None
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

        sequences = sequences.to(device, non_blocking=True)
        organism_idx = torch.zeros(sequences.shape[0], dtype=torch.long, device=device)

        predictions_by_modality, metric_predictions_by_modality = forward_heads(
            model,
            heads_model,
            sequences,
            organism_idx,
            crop_bins,
            use_amp=not args.no_amp,
            return_scaled=args.loss in ("poisson-multinomial", "multinomial"),
            requires_grad=training,
        )
        loss = torch.tensor(0.0, device=device)
        loss_by_modality = {}
        for modality, pred in predictions_by_modality.items():
            target = modality_targets[modality][128].to(device, non_blocking=True)
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
                organism_idx=organism_idx,
                positional_weight=args.positional_weight,
                count_weight=args.count_weight,
            )
            loss = loss + modality_loss
            loss_by_modality[modality] = float(modality_loss.detach())
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
        for stats in metric_stats.values():
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)

    metrics: dict[str, float] = {}
    grad_metrics: dict[str, float] = {}
    if training and grad_norm_count.item() > 0:
        grad_metrics["grad_norm"] = (grad_norm_sum / grad_norm_count).item()
        grad_metrics["grad_norm_max"] = grad_norm_max.item()
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


def main() -> None:
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    rank, world_size, local_rank, device = setup_torchrun(args.device)
    wandb_run = None
    try:
        modalities = resolve_modalities(args)
        print_rank0(f"Distributed: rank={rank} world_size={world_size}", rank)

        train_dataset = create_tfr_dataset(
            args.data_dir,
            "train",
            modalities,
            args.pooling,
            repeat=False,
            shuffle_files=True,
            num_parallel_reads=args.tfr_num_parallel_reads,
            rank=rank,
            world_size=world_size,
        )
        val_dataset = create_tfr_dataset(
            args.data_dir,
            "valid",
            modalities,
            args.pooling,
            num_parallel_reads=args.tfr_num_parallel_reads,
            rank=rank,
            world_size=world_size,
        )

        train_loader = create_loader(
            train_dataset,
            args.batch_size,
            args.num_workers,
            prefetch_n=args.prefetch_n,
            persistent_workers=True,
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

        sample_sequences, sample_targets = next(iter(train_loader))
        print_rank0(
            "Loader OK: "
            f"seq={tuple(sample_sequences.shape)} "
            + "targets="
            + ",".join(
                f"{modality}:{tuple(targets[128].shape)}"
                for modality, targets in sample_targets.items()
            )
            + f" crop_bins={train_dataset.prediction_crop_128bp}"
            + f" modalities={modalities}",
            rank,
        )
        if args.loader_only:
            return

        if args.pretrained_weights is None:
            raise SystemExit("--pretrained-weights is required unless --loader-only is set")

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
        model = load_trunk(model, str(args.pretrained_weights), exclude_heads=True)
        model = remove_all_heads(model).to(device)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

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
        heads_model: nn.Module = TFRHeads(
            heads,
            cell_types_by_modality,
            cell_embedding_dim=args.cell_embedding_dim,
        ).to(device)
        print_rank0(
            "Cell layers: "
            f"{len(unwrap_heads(heads_model).cell_layers)} shared ct transforms; "
            f"cell_embedding_dim={args.cell_embedding_dim}; "
            f"modality layers={len(unwrap_heads(heads_model).modality_layers)}",
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
        warmup_steps = resolve_warmup_steps(args.warmup_steps, total_optimizer_steps)
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

        run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
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
            "cell_types": cell_types_by_modality,
            "n_cell_types": len(unwrap_heads(heads_model).cell_layers),
            "cell_embedding_dim": args.cell_embedding_dim,
            "n_modality_layers": len(unwrap_heads(heads_model).modality_layers),
            "head_factorization": "cell_bottleneck_linear_then_modality_mlp",
            "crop_bins_128bp": train_dataset.prediction_crop_128bp,
            "target_length_128bp": train_dataset.output_length_128bp,
            "loss": args.loss,
            "pretrained_weights": str(args.pretrained_weights),
            "world_size": world_size,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "val_num_workers": args.val_num_workers,
            "prefetch_n": args.prefetch_n,
            "epochs": args.epochs,
            "lr": args.lr,
            "lr_schedule": args.lr_schedule,
            "warmup_steps": warmup_steps,
            "warmup_steps_arg": args.warmup_steps,
            "estimated_optimizer_steps": total_optimizer_steps,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "weight_decay": args.weight_decay,
            "dtype": args.dtype,
            "use_amp": not args.no_amp,
        }
        if is_main_process(rank):
            (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        wandb_run = create_wandb_run(args, rank, run_name, metadata)

        best_val = float("inf")
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
                train_dataset.prediction_crop_128bp,
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
                val_dataset.prediction_crop_128bp,
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
                }
                save_checkpoint(
                    output_dir / "last_heads.pt",
                    heads_model,
                    epoch_metadata,
                )
                if val_loss < best_val:
                    best_val = val_loss
                    save_checkpoint(
                        output_dir / "best_heads.pt",
                        heads_model,
                        epoch_metadata,
                    )
    finally:
        finish_wandb(wandb_run)
        cleanup_torchrun()


if __name__ == "__main__":
    main()
