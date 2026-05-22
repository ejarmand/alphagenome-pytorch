"""TFRecord datasets for AlphaGenome fine-tuning.

This module reads Baskerville/Borzoi TFRecord datasets with TensorFlow on CPU
and yields PyTorch tensors for training AlphaGenome heads.
"""

from __future__ import annotations

import csv
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

_Pooling = Literal["mean", "sum"]
_Split = Literal["train", "valid", "test"]

__all__ = [
    "BaskervilleTFRecordDataset",
    "BaskervilleMultiTFRecordDataset",
    "collate_tfr_genomic",
    "collate_tfr_multimodal",
]


_MODALITY_TO_ASSAY_TYPE = {
    "ATAC": "atac",
    "CGN": "chip_histone",
    "CHN": "chip_histone",
    "H3K27AC": "chip_histone",
    "H3K27ME3": "chip_histone",
    "H3K4ME1": "chip_histone",
    "H3K9ME3": "chip_histone",
    "RNA": "rna_seq",
}


@dataclass(frozen=True)
class _TFRecordDatasetMetadata:
    """Metadata loaded from a Baskerville TFRecord dataset directory."""

    seq_length: int
    target_length: int
    num_targets: int
    pool_width: int
    crop_bp: int
    split_counts: dict[str, int]

    @property
    def output_length_128bp(self) -> int:
        if self.target_length % 4 != 0:
            raise ValueError(
                f"target_length={self.target_length} is not divisible by 4, "
                "so 32 bp targets cannot be pooled to 128 bp bins."
            )
        return self.target_length // 4

    @property
    def prediction_crop_128bp(self) -> int:
        if self.crop_bp % 128 != 0:
            raise ValueError(
                f"crop_bp={self.crop_bp} is not divisible by 128, so prediction "
                "crop cannot be represented at 128 bp resolution."
            )
        return self.crop_bp // 128


def _natural_sort_key(path: Path) -> tuple:
    parts = re.split(r"(\d+)", path.name)
    return tuple(int(p) if p.isdigit() else p for p in parts)


def collate_tfr_genomic(
    batch: list[tuple[torch.Tensor, dict[int, torch.Tensor]]],
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """Collate samples yielded by :class:`BaskervilleTFRecordDataset`."""
    sequences = torch.stack([item[0] for item in batch], dim=0)
    targets = {128: torch.stack([item[1][128] for item in batch], dim=0)}
    return sequences, targets


def collate_tfr_multimodal(
    batch: list[tuple[torch.Tensor, dict[str, dict[int, torch.Tensor]]]],
) -> tuple[torch.Tensor, dict[str, dict[int, torch.Tensor]]]:
    """Collate samples yielded by :class:`BaskervilleMultiTFRecordDataset`."""
    sequences = torch.stack([item[0] for item in batch], dim=0)
    first_targets = batch[0][1]
    targets = {
        modality: {
            128: torch.stack([item[1][modality][128] for item in batch], dim=0)
        }
        for modality in first_targets
    }
    return sequences, targets


class BaskervilleTFRecordDataset(IterableDataset):
    """Iterable PyTorch dataset for Baskerville/Borzoi TFRecords.

    TensorFlow is imported lazily inside iteration and is configured to avoid
    GPU visibility before any ``tf.data`` pipeline is constructed. This keeps
    TFRecord parsing on CPU and prevents TensorFlow from pre-allocating GPU VRAM
    in PyTorch training jobs.
    """

    def __init__(
        self,
        data_dir: str | Path,
        *,
        split: _Split = "train",
        modality: str = "ATAC",
        pooling: _Pooling = "mean",
        shuffle_files: bool = False,
        repeat: bool = False,
        seed: int = 0,
        num_parallel_reads: int | None = None,
    ):
        super().__init__()
        if pooling not in ("mean", "sum"):
            raise ValueError(f"pooling must be 'mean' or 'sum', got {pooling!r}")

        self.data_dir = Path(data_dir)
        self.split = split
        self.modality = modality
        self.pooling = pooling
        self.shuffle_files = shuffle_files
        self.repeat = repeat
        self.seed = seed
        self.num_parallel_reads = num_parallel_reads

        self.metadata = self._load_metadata(self.data_dir)
        self.files = self._get_tfrecord_files(self.data_dir, split)
        indices, target_rows = self._modality_indices(self.data_dir, modality)
        self.target_indices = np.asarray(indices, dtype=np.int64)
        self.target_rows = target_rows
        self.n_tracks = len(indices)

    def __len__(self) -> int:
        if self.split not in self.metadata.split_counts:
            raise TypeError(f"No sequence count found for split={self.split!r}")
        return self.metadata.split_counts[self.split]

    @property
    def assay_type(self) -> str:
        key = self.modality.upper()
        if key not in _MODALITY_TO_ASSAY_TYPE:
            available = ", ".join(sorted(_MODALITY_TO_ASSAY_TYPE))
            raise ValueError(
                f"Unknown TFRecord modality {self.modality!r}. "
                f"Known modalities: {available}"
            )
        return _MODALITY_TO_ASSAY_TYPE[key]

    @property
    def prediction_crop_128bp(self) -> int:
        return self.metadata.prediction_crop_128bp

    @property
    def output_length_128bp(self) -> int:
        return self.metadata.output_length_128bp

    @staticmethod
    def _load_metadata(data_dir: Path) -> _TFRecordDatasetMetadata:
        with (data_dir / "statistics.json").open() as f:
            stats = json.load(f)

        split_counts = {}
        for split in ("train", "valid", "test"):
            key = f"{split}_seqs"
            if key in stats:
                split_counts[split] = int(stats[key])

        return _TFRecordDatasetMetadata(
            seq_length=int(stats["seq_length"]),
            target_length=int(stats["target_length"]),
            num_targets=int(stats["num_targets"]),
            pool_width=int(stats.get("pool_width", 32)),
            crop_bp=int(stats.get("crop_bp", 0)),
            split_counts=split_counts,
        )

    @staticmethod
    def _get_tfrecord_files(data_dir: Path, split: _Split) -> list[Path]:
        tfrecord_dir = data_dir / "tfrecords"
        files = sorted(tfrecord_dir.glob(f"{split}-*.tfr"), key=_natural_sort_key)
        if not files:
            raise FileNotFoundError(
                f"No TFRecord files found for split={split!r} in {tfrecord_dir}"
            )
        return files

    @staticmethod
    def _load_targets_table(data_dir: Path) -> list[dict[str, str]]:
        with (data_dir / "targets.txt").open(newline="") as f:
            return list(csv.DictReader(f, delimiter="\t"))

    @classmethod
    def _modality_indices(
        cls,
        data_dir: Path,
        modality: str,
    ) -> tuple[list[int], list[dict[str, str]]]:
        requested = modality.upper()
        rows = cls._load_targets_table(data_dir)
        selected = [
            row for row in rows
            if row.get("modality", "").upper() == requested
        ]
        if not selected:
            available = sorted({
                row.get("modality", "") for row in rows if row.get("modality")
            })
            raise ValueError(
                f"No targets found for modality={modality!r}. "
                f"Available modalities: {available}"
            )

        # Some Baskerville dataset builds keep original/global target ``index``
        # values in targets.txt after filtering tracks, while TFRecord targets
        # are packed densely in targets.txt row order. Use explicit indices only
        # when they fit the decoded target matrix; otherwise use row positions.
        use_row_positions = False
        if len(rows) == cls._load_metadata(data_dir).num_targets:
            index_values = [int(row["index"]) for row in rows]
            use_row_positions = max(index_values, default=-1) >= len(rows)

        if use_row_positions:
            indices = [
                row_idx for row_idx, row in enumerate(rows)
                if row.get("modality", "").upper() == requested
            ]
        else:
            indices = [int(row["index"]) for row in selected]
        return indices, selected

    @classmethod
    def available_modalities(cls, data_dir: str | Path) -> list[str]:
        """Return modality labels present in targets.txt in first-seen order."""
        rows = cls._load_targets_table(Path(data_dir))
        modalities = []
        seen = set()
        for row in rows:
            modality = row.get("modality", "")
            if modality and modality.upper() not in seen:
                modalities.append(modality)
                seen.add(modality.upper())
        return modalities

    @staticmethod
    def _decode_sequence(sequence_bytes: bytes, seq_length: int) -> np.ndarray:
        encoded = np.frombuffer(sequence_bytes, dtype=np.uint8)

        if encoded.size == seq_length * 4:
            return encoded.reshape(seq_length, 4).astype(np.float32, copy=False)

        if encoded.size == seq_length:
            one_hot = np.zeros((seq_length, 4), dtype=np.float32)
            valid = encoded < 4
            one_hot[np.arange(seq_length)[valid], encoded[valid]] = 1.0
            return one_hot

        raise ValueError(
            f"Unexpected sequence byte length: decoded {encoded.size} uint8 values; "
            f"expected {seq_length} base indices or {seq_length * 4} one-hot values."
        )

    def _decode_target(self, target_bytes: bytes) -> np.ndarray:
        target = self._decode_full_target(target_bytes)
        target = target[:, self.target_indices]
        return target.astype(np.float32, copy=False)

    def _decode_full_target(self, target_bytes: bytes) -> np.ndarray:
        target = np.frombuffer(target_bytes, dtype=np.float16)
        expected = self.metadata.target_length * self.metadata.num_targets
        if target.size != expected:
            raise ValueError(
                f"Unexpected target byte length: decoded {target.size} float16 values; "
                f"expected {expected} for shape "
                f"({self.metadata.target_length}, {self.metadata.num_targets})."
            )
        return target.reshape(self.metadata.target_length, self.metadata.num_targets)

    def _pool_target_128bp(self, target: np.ndarray) -> np.ndarray:
        if target.shape[0] % 4 != 0:
            raise ValueError(
                f"Target length {target.shape[0]} is not divisible by 4; "
                "cannot pool 32 bp targets to 128 bp."
            )
        reshaped = target.reshape(target.shape[0] // 4, 4, target.shape[1])
        if self.pooling == "mean":
            return reshaped.mean(axis=1)
        if self.pooling == "sum":
            return reshaped.sum(axis=1)
        raise ValueError(f"Unknown pooling={self.pooling!r}; expected 'mean' or 'sum'.")

    def _worker_files(self) -> list[Path]:
        files = list(self.files)
        worker = get_worker_info()
        if worker is not None:
            files = files[worker.id::worker.num_workers]
        if self.shuffle_files:
            shuffle_seed = self.seed + (worker.id if worker is not None else 0)
            rng = random.Random(shuffle_seed)
            rng.shuffle(files)
        return files

    @staticmethod
    def _ensure_tensorflow_cpu():
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        try:
            import tensorflow as tf  # pylint: disable=import-outside-toplevel
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "BaskervilleTFRecordDataset requires TensorFlow for TFRecord "
                "parsing. Install tensorflow in the training environment."
            ) from exc

        try:
            tf.config.set_visible_devices([], "GPU")
        except RuntimeError:
            # TensorFlow devices may already be initialized in this process.
            # Keep going; the environment variable still protects worker processes.
            pass
        return tf

    def _tf_dataset(self, files: Iterable[Path]):
        tf = self._ensure_tensorflow_cpu()
        file_names = [str(path) for path in files]
        if not file_names:
            return None

        parallel_reads = self.num_parallel_reads
        if parallel_reads is None:
            parallel_reads = tf.data.AUTOTUNE

        dataset = tf.data.TFRecordDataset(
            file_names,
            compression_type="ZLIB",
            num_parallel_reads=parallel_reads,
        )
        options = tf.data.Options()
        options.experimental_deterministic = not self.shuffle_files
        return dataset.with_options(options)

    def __iter__(self):
        files = self._worker_files()
        if not files:
            return

        while True:
            dataset = self._tf_dataset(files)
            if dataset is None:
                return

            tf = self._ensure_tensorflow_cpu()
            feature_spec = {
                "sequence": tf.io.FixedLenFeature([], tf.string),
                "target": tf.io.FixedLenFeature([], tf.string),
            }

            for record in dataset:
                parsed = tf.io.parse_single_example(record, feature_spec)
                sequence = self._decode_sequence(
                    parsed["sequence"].numpy(),
                    self.metadata.seq_length,
                )
                target = self._decode_target(parsed["target"].numpy())
                target_128bp = self._pool_target_128bp(target)

                yield (
                    torch.from_numpy(sequence).float(),
                    {128: torch.from_numpy(target_128bp).float()},
                )

            if not self.repeat:
                break


class BaskervilleMultiTFRecordDataset(BaskervilleTFRecordDataset):
    """Iterable dataset that yields targets for multiple modalities per sequence."""

    def __init__(
        self,
        data_dir: str | Path,
        *,
        split: _Split = "train",
        modalities: Iterable[str] | None = None,
        pooling: _Pooling = "mean",
        shuffle_files: bool = False,
        repeat: bool = False,
        seed: int = 0,
        num_parallel_reads: int | None = None,
    ):
        if modalities is None:
            modalities = self.available_modalities(data_dir)
        self.modalities = list(modalities)
        if not self.modalities:
            raise ValueError("At least one modality is required")

        super().__init__(
            data_dir,
            split=split,
            modality=self.modalities[0],
            pooling=pooling,
            shuffle_files=shuffle_files,
            repeat=repeat,
            seed=seed,
            num_parallel_reads=num_parallel_reads,
        )

        self.target_indices_by_modality: dict[str, np.ndarray] = {}
        self.target_rows_by_modality: dict[str, list[dict[str, str]]] = {}
        self.assay_type_by_modality: dict[str, str] = {}
        for modality in self.modalities:
            indices, rows = self._modality_indices(self.data_dir, modality)
            self.target_indices_by_modality[modality] = np.asarray(indices, dtype=np.int64)
            self.target_rows_by_modality[modality] = rows
            key = modality.upper()
            if key not in _MODALITY_TO_ASSAY_TYPE:
                available = ", ".join(sorted(_MODALITY_TO_ASSAY_TYPE))
                raise ValueError(
                    f"Unknown TFRecord modality {modality!r}. "
                    f"Known modalities: {available}"
                )
            self.assay_type_by_modality[modality] = _MODALITY_TO_ASSAY_TYPE[key]

    def __iter__(self):
        files = self._worker_files()
        if not files:
            return

        while True:
            dataset = self._tf_dataset(files)
            if dataset is None:
                return

            tf = self._ensure_tensorflow_cpu()
            feature_spec = {
                "sequence": tf.io.FixedLenFeature([], tf.string),
                "target": tf.io.FixedLenFeature([], tf.string),
            }

            for record in dataset:
                parsed = tf.io.parse_single_example(record, feature_spec)
                sequence = self._decode_sequence(
                    parsed["sequence"].numpy(),
                    self.metadata.seq_length,
                )
                full_target = self._decode_full_target(parsed["target"].numpy())
                modality_targets = {}
                for modality, indices in self.target_indices_by_modality.items():
                    target = full_target[:, indices].astype(np.float32, copy=False)
                    target_128bp = self._pool_target_128bp(target)
                    modality_targets[modality] = {
                        128: torch.from_numpy(target_128bp).float()
                    }

                yield torch.from_numpy(sequence).float(), modality_targets

            if not self.repeat:
                break
