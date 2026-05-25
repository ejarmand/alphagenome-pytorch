"""Tests for Baskerville TFRecord fine-tuning dataset helpers."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import alphagenome_pytorch.extensions.finetuning.tfrecord_dataset as tfr_module
from alphagenome_pytorch.extensions.finetuning.tfrecord_dataset import (
    BaskervilleMultiTFRecordDataset,
    BaskervilleTFRecordDataset,
    collate_tfr_genomic,
    collate_tfr_multimodal,
)
from alphagenome_pytorch.extensions.finetuning.heads import create_finetuning_head
from alphagenome_pytorch.heads import GenomeTracksHead
from alphagenome_pytorch.utils.sequence import (
    reverse_complement_onehot,
    shift_onehot,
)
from scripts.finetune_tfr_heads import (
    TFRHeads,
    compute_loss,
    compute_regression_metrics,
    cell_types_from_target_rows,
    new_metric_stats,
    switch_reverse_predictions,
    update_metric_stats,
)


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


def test_multimodal_metadata_and_modality_selection(tmp_path):
    data_dir = _write_minimal_dataset(tmp_path)

    dataset = BaskervilleMultiTFRecordDataset(
        data_dir,
        modalities=["ATAC", "RNA"],
    )

    assert dataset.modalities == ["ATAC", "RNA"]
    assert dataset.assay_type_by_modality == {"ATAC": "atac", "RNA": "rna_seq"}
    assert dataset.target_indices_by_modality["ATAC"].tolist() == [0, 2]
    assert dataset.target_indices_by_modality["RNA"].tolist() == [1]
    assert dataset.strand_pair_by_modality["ATAC"].tolist() == [0, 1]
    assert dataset.strand_pair_by_modality["RNA"].tolist() == [0]
    assert BaskervilleTFRecordDataset.available_modalities(data_dir) == ["ATAC", "RNA"]


def test_cell_types_from_target_rows_uses_ct_with_identifier_fallback():
    rows = {
        "ATAC": [
            {"identifier": "track0", "ct": "cell_a"},
            {"identifier": "track1", "ct": ""},
        ],
        "RNA": [{"identifier": "track2"}],
    }

    assert cell_types_from_target_rows(rows) == {
        "ATAC": ["cell_a", "track1"],
        "RNA": ["track2"],
    }


def test_cell_types_from_target_rows_keeps_stranded_tracks_distinct():
    rows = {
        "RNA": [
            {"index": "0", "identifier": "cell+", "ct": "cell", "strand_pair": "1"},
            {"index": "1", "identifier": "cell-", "ct": "cell", "strand_pair": "0"},
            {"index": "2", "identifier": "bulk", "ct": "bulk", "strand_pair": "2"},
        ],
    }

    assert cell_types_from_target_rows(rows) == {
        "RNA": ["cell|strand=+", "cell|strand=-", "bulk"],
    }


def test_tfr_heads_shares_cell_layers_across_modalities():
    heads = {
        "ATAC": GenomeTracksHead(
            in_channels={128: 4},
            num_tracks=2,
            resolutions=(128,),
            num_organisms=1,
        ),
        "RNA": GenomeTracksHead(
            in_channels={128: 4},
            num_tracks=1,
            resolutions=(128,),
            num_organisms=1,
            apply_squashing=True,
        ),
    }
    model = TFRHeads(
        heads,
        cell_types_by_modality={
            "ATAC": ["shared_ct", "atac_only"],
            "RNA": ["shared_ct"],
        },
        embedding_dim=4,
        cell_embedding_dim=2,
    )

    assert len(model.cell_layers) == 2
    assert len(model.modality_layers) == 2
    assert model.modality_layers["ATAC"].layers[0].out_channels == 2
    assert model.modality_layers["ATAC"].layers[2].out_channels == 1
    assert model.modality_layers["RNA"].layers[0].out_channels == 2
    assert model.modality_layers["RNA"].layers[2].out_channels == 1
    shared_key = model.cell_key_by_type["shared_ct"]
    assert model.cell_layers[shared_key].proj.in_channels == 4
    assert model.cell_layers[shared_key].proj.out_channels == 2
    assert shared_key in model.track_groups["ATAC"].cell_keys
    assert shared_key in model.track_groups["RNA"].cell_keys

    embeddings = {128: torch.randn(2, 4, 3)}
    organism_idx = torch.zeros(2, dtype=torch.long)

    outputs = model(embeddings, organism_idx, return_scaled=True)

    assert outputs["ATAC"].shape == (2, 3, 2)
    assert outputs["RNA"].shape == (2, 3, 1)


def test_worker_files_shards_by_rank(tmp_path):
    data_dir = _write_minimal_dataset(tmp_path)
    for i in range(4):
        (data_dir / "tfrecords" / f"train-{i}.tfr").write_bytes(b"")

    rank0 = BaskervilleTFRecordDataset(data_dir, modality="ATAC", rank=0, world_size=2)
    rank1 = BaskervilleTFRecordDataset(data_dir, modality="ATAC", rank=1, world_size=2)

    assert [path.name for path in rank0._worker_files()] == [
        "train-0.tfr",
        "train-2.tfr",
        "train-10.tfr",
    ]
    assert [path.name for path in rank1._worker_files()] == [
        "train-1.tfr",
        "train-3.tfr",
    ]


def test_worker_files_shards_by_rank_before_worker(tmp_path, monkeypatch):
    data_dir = _write_minimal_dataset(tmp_path)
    for path in (data_dir / "tfrecords").glob("train-*.tfr"):
        path.unlink()
    for i in range(16):
        (data_dir / "tfrecords" / f"train-{i}.tfr").write_bytes(b"")

    for rank in range(8):
        dataset = BaskervilleTFRecordDataset(
            data_dir,
            modality="ATAC",
            rank=rank,
            world_size=8,
        )
        rank_files = []
        for worker_id in range(8):
            monkeypatch.setattr(
                tfr_module,
                "get_worker_info",
                lambda worker_id=worker_id: SimpleNamespace(
                    id=worker_id,
                    num_workers=8,
                ),
            )
            rank_files.extend(path.name for path in dataset._worker_files())

        assert set(rank_files) == {f"train-{rank}.tfr", f"train-{rank + 8}.tfr"}


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


def test_reverse_complement_sequence_matches_baskerville_order():
    sequence = np.eye(4, dtype=np.float32)

    rc = BaskervilleTFRecordDataset._reverse_complement_sequence(sequence)

    np.testing.assert_array_equal(rc, reverse_complement_onehot(sequence))
    np.testing.assert_array_equal(rc, np.eye(4, dtype=np.float32))


def test_shift_sequence_matches_baskerville_padding():
    sequence = np.arange(20, dtype=np.float32).reshape(5, 4)

    shifted_right = BaskervilleTFRecordDataset._shift_sequence(sequence, 2)
    shifted_left = BaskervilleTFRecordDataset._shift_sequence(sequence, -2)

    np.testing.assert_array_equal(shifted_right, shift_onehot(sequence, 2))
    np.testing.assert_array_equal(shifted_left, shift_onehot(sequence, -2))
    np.testing.assert_array_equal(shifted_right[:2], np.zeros((2, 4), dtype=np.float32))
    np.testing.assert_array_equal(shifted_right[2:], sequence[:-2])
    np.testing.assert_array_equal(shifted_left[-2:], np.zeros((2, 4), dtype=np.float32))
    np.testing.assert_array_equal(shifted_left[:-2], sequence[2:])


def test_strand_pair_uses_baskerville_column(tmp_path):
    data_dir = _write_minimal_dataset(tmp_path)
    (data_dir / "targets.txt").write_text(
        "\t".join(["index", "identifier", "modality", "strand_pair"]) + "\n"
        "0\tcell+\tRNA\t1\n"
        "1\tcell-\tRNA\t0\n"
        "2\tatac\tATAC\t2\n"
    )

    dataset = BaskervilleMultiTFRecordDataset(data_dir, modalities=["RNA", "ATAC"])

    assert dataset.strand_pair_by_modality["RNA"].tolist() == [1, 0]
    assert dataset.strand_pair_by_modality["ATAC"].tolist() == [0]


def test_switch_reverse_predictions_flips_length_and_strand_pairs():
    pred = torch.tensor(
        [
            [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
            [[4.0, 40.0], [5.0, 50.0], [6.0, 60.0]],
        ]
    )
    reverse = torch.tensor([True, False])
    strand_pair = torch.tensor([1, 0])

    switched = switch_reverse_predictions(pred, reverse, strand_pair)

    expected = torch.tensor(
        [
            [[30.0, 3.0], [20.0, 2.0], [10.0, 1.0]],
            [[4.0, 40.0], [5.0, 50.0], [6.0, 60.0]],
        ]
    )
    torch.testing.assert_close(switched, expected)


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


def test_collate_tfr_multimodal():
    seq0 = torch.zeros(4, 4)
    seq1 = torch.ones(4, 4)
    atac0 = torch.zeros(2, 3)
    atac1 = torch.ones(2, 3)
    rna0 = torch.full((2, 1), 2.0)
    rna1 = torch.full((2, 1), 3.0)

    sequences, targets = collate_tfr_multimodal(
        [
            (seq0, {"ATAC": {128: atac0}, "RNA": {128: rna0}}),
            (seq1, {"ATAC": {128: atac1}, "RNA": {128: rna1}}),
        ]
    )

    assert sequences.shape == (2, 4, 4)
    assert targets["ATAC"][128].shape == (2, 2, 3)
    assert targets["RNA"][128].shape == (2, 2, 1)
    torch.testing.assert_close(targets["ATAC"][128][1], atac1)
    torch.testing.assert_close(targets["RNA"][128][1], rna1)


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


def test_regression_metric_stats_accumulate_pearson_and_r2():
    pred = torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]])
    target = pred.clone()
    stats = new_metric_stats(torch.device("cpu"))

    update_metric_stats(stats, pred, target)
    metrics = compute_regression_metrics(stats)

    assert metrics["pearson_r"] == pytest.approx(1.0)
    assert metrics["r2"] == pytest.approx(1.0)
