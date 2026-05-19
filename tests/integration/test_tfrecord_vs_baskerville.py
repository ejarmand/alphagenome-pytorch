"""Compare the PyTorch TFRecord dataset against Baskerville's TF dataset."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from alphagenome_pytorch.extensions.finetuning.tfrecord_dataset import (  # noqa: E402
    BaskervilleTFRecordDataset,
)


DATA_DIR = (
    Path(__file__).resolve().parents[4]
    / "baskerville_dnn/out/datasets/mouse_fixed_borzoi_contig"
)


def _load_baskerville_first_example(data_dir: Path):
    pytest.importorskip("tensorflow")
    baskerville_dataset = pytest.importorskip("baskerville.dataset")

    ds = baskerville_dataset.SeqDataset(
        str(data_dir),
        split_label="train",
        batch_size=1,
        mode="eval",
        tfr_pattern="train-0.tfr",
    )
    sequence_batch, target_batch = next(iter(ds.dataset))
    return sequence_batch.numpy()[0], target_batch.numpy()[0]


def _expected_pooled_targets(
    raw_targets: np.ndarray,
    target_indices: np.ndarray,
    pooling: str,
) -> np.ndarray:
    selected = raw_targets[:, target_indices]
    grouped = selected.reshape(selected.shape[0] // 4, 4, selected.shape[1])
    if pooling == "mean":
        return grouped.mean(axis=1)
    if pooling == "sum":
        return grouped.sum(axis=1)
    raise AssertionError(f"Unexpected pooling={pooling!r}")


@pytest.fixture(scope="module")
def baskerville_first_example():
    if not DATA_DIR.exists():
        pytest.skip(f"Baskerville dataset not found: {DATA_DIR}")
    return _load_baskerville_first_example(DATA_DIR)


@pytest.mark.integration
def test_tfrecord_dataset_preserves_baskerville_data_properties(
    baskerville_first_example,
):
    baskerville_sequence, baskerville_targets = baskerville_first_example

    assert tuple(baskerville_sequence.shape) == (524_288, 4)
    assert tuple(baskerville_targets.shape) == (16_320, 1_594)

    for modality, expected_tracks in [("ATAC", 250), ("RNA", 326)]:
        for pooling in ["mean", "sum"]:
            torch_dataset = BaskervilleTFRecordDataset(
                DATA_DIR,
                split="train",
                modality=modality,
                pooling=pooling,
            )
            torch_sequence, torch_targets = next(iter(torch_dataset))

            torch_sequence_np = torch_sequence.numpy()
            torch_target_np = torch_targets[128].numpy()
            expected_target = _expected_pooled_targets(
                baskerville_targets,
                torch_dataset.target_indices,
                pooling,
            )

            assert torch_dataset.n_tracks == expected_tracks
            assert tuple(torch_sequence_np.shape) == (524_288, 4)
            assert tuple(torch_target_np.shape) == (4_080, expected_tracks)

            np.testing.assert_array_equal(torch_sequence_np, baskerville_sequence)
            np.testing.assert_allclose(torch_target_np, expected_target, rtol=0, atol=0)

            assert torch_sequence_np.dtype == np.float32
            assert torch_target_np.dtype == np.float32
            assert np.all((torch_sequence_np == 0.0) | (torch_sequence_np == 1.0))
            assert np.isfinite(torch_target_np).all()
