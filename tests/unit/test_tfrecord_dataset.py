"""Tests for Baskerville TFRecord fine-tuning dataset helpers."""

from __future__ import annotations

import json

import numpy as np
import torch

from alphagenome_pytorch.extensions.finetuning.tfrecord_dataset import (
    BaskervilleTFRecordDataset,
    collate_tfr_genomic,
)
from alphagenome_pytorch.extensions.finetuning.heads import create_finetuning_head
from scripts.finetune_tfr_heads import compute_loss


def _write_minimal_dataset(tmp_path):
    data_dir = tmp_path / "dataset"
    tfrecord_dir = data_dir / "tfrecords"
    tfrecord_dir.mkdir(parents=True)

    (data_dir / "statistics.json").write_text(
        json.dumps(
            {
                "num_targets": 3,
                "seq_length": 8,
                "pool_width": 32,
                "crop_bp": 128,
                "target_length": 8,
                "train_seqs": 2,
            }
        )
    )
    (data_dir / "targets.txt").write_text(
        "\t".join(["index", "identifier", "modality"]) + "\n"
        "0\ttrack0\tATAC\n"
        "1\ttrack1\tRNA\n"
        "2\ttrack2\tATAC\n"
    )
    (tfrecord_dir / "train-10.tfr").write_bytes(b"")
    (tfrecord_dir / "train-2.tfr").write_bytes(b"")
    return data_dir


def test_metadata_files_and_modality_selection(tmp_path):
    data_dir = _write_minimal_dataset(tmp_path)

    dataset = BaskervilleTFRecordDataset(data_dir, modality="ATAC")

    assert len(dataset) == 2
    assert [path.name for path in dataset.files] == ["train-2.tfr", "train-10.tfr"]
    assert dataset.target_indices.tolist() == [0, 2]
    assert dataset.n_tracks == 2
    assert dataset.assay_type == "atac"
    assert dataset.prediction_crop_128bp == 1
    assert dataset.output_length_128bp == 2


def test_modality_selection_uses_row_positions_for_filtered_targets(tmp_path):
    data_dir = _write_minimal_dataset(tmp_path)
    (data_dir / "statistics.json").write_text(
        json.dumps(
            {
                "num_targets": 2,
                "seq_length": 8,
                "pool_width": 32,
                "crop_bp": 128,
                "target_length": 8,
                "train_seqs": 2,
            }
        )
    )
    (data_dir / "targets.txt").write_text(
        "\t".join(["index", "identifier", "modality"]) + "\n"
        "5\ttrack5\tRNA\n"
        "8\ttrack8\tATAC\n"
    )

    dataset = BaskervilleTFRecordDataset(data_dir, modality="ATAC")

    assert dataset.target_indices.tolist() == [1]


def test_decode_sequence_accepts_indices_and_onehot():
    indexed = np.array([0, 1, 2, 3, 4], dtype=np.uint8)
    decoded = BaskervilleTFRecordDataset._decode_sequence(indexed.tobytes(), 5)

    np.testing.assert_array_equal(
        decoded,
        np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 0],
            ],
            dtype=np.float32,
        ),
    )

    onehot = np.eye(4, dtype=np.uint8)[[0, 1, 2, 3]]
    decoded_onehot = BaskervilleTFRecordDataset._decode_sequence(onehot.tobytes(), 4)
    np.testing.assert_array_equal(decoded_onehot, onehot.astype(np.float32))


def test_decode_and_pool_targets(tmp_path):
    data_dir = _write_minimal_dataset(tmp_path)
    dataset = BaskervilleTFRecordDataset(data_dir, modality="ATAC", pooling="mean")

    raw = np.arange(24, dtype=np.float16).reshape(8, 3)
    selected = dataset._decode_target(raw.tobytes())
    pooled = dataset._pool_target_128bp(selected)

    np.testing.assert_array_equal(selected, raw[:, [0, 2]].astype(np.float32))
    np.testing.assert_allclose(pooled, selected.reshape(2, 4, 2).mean(axis=1))


def test_collate_tfr_genomic():
    seq0 = torch.zeros(4, 4)
    seq1 = torch.ones(4, 4)
    target0 = torch.zeros(2, 3)
    target1 = torch.ones(2, 3)

    sequences, targets = collate_tfr_genomic(
        [(seq0, {128: target0}), (seq1, {128: target1})]
    )

    assert sequences.shape == (2, 4, 4)
    assert targets[128].shape == (2, 2, 3)
    torch.testing.assert_close(sequences[1], seq1)
    torch.testing.assert_close(targets[128][1], target1)


def test_poisson_multinomial_loss_alias_runs():
    head = create_finetuning_head("atac", n_tracks=2, resolutions=(128,))
    organism_idx = torch.zeros(1, dtype=torch.long)
    pred = torch.ones(1, 8, 2)
    target = torch.ones(1, 8, 2)

    loss = compute_loss(
        pred,
        target,
        loss_name="poisson-multinomial",
        head=head,
        organism_idx=organism_idx,
        positional_weight=5.0,
        count_weight=1.0,
    )

    assert torch.isfinite(loss)
