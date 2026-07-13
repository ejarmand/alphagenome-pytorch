"""Unit tests for gene-level TFRecord strand target selection."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

spec = importlib.util.spec_from_file_location(
    "test_tfr_genes_script",
    SCRIPTS_DIR / "test_tfr_genes.py",
)
assert spec is not None
test_tfr_genes = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = test_tfr_genes
spec.loader.exec_module(test_tfr_genes)


def paired_targets_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "index": 0,
                "global_index": 0,
                "identifier": "atac",
                "strand": "",
                "strand_pair": 0,
            },
            {
                "index": 10,
                "global_index": 1,
                "identifier": "rna_a_forward",
                "strand": "+",
                "strand_pair": 21,
            },
            {
                "index": 20,
                "global_index": 2,
                "identifier": "rna_b_forward",
                "strand": "+",
                "strand_pair": 31,
            },
            {
                "index": 30,
                "global_index": 3,
                "identifier": "rna_c_forward",
                "strand": "+",
                "strand_pair": 41,
            },
            {
                "index": 21,
                "global_index": 4,
                "identifier": "rna_a_reverse",
                "strand": "-",
                "strand_pair": 10,
            },
            {
                "index": 31,
                "global_index": 5,
                "identifier": "rna_b_reverse",
                "strand": "-",
                "strand_pair": 20,
            },
            {
                "index": 41,
                "global_index": 6,
                "identifier": "rna_c_reverse",
                "strand": "-",
                "strand_pair": 30,
            },
        ]
    )


def test_strand_column_indices_use_strand_pair_for_negative_genes() -> None:
    targets_df = paired_targets_df()
    output_targets_df = test_tfr_genes.strand_output_rows(targets_df, flip_strand=False)

    assert output_targets_df["identifier"].tolist() == [
        "atac",
        "rna_a_forward",
        "rna_b_forward",
        "rna_c_forward",
    ]
    np.testing.assert_array_equal(
        test_tfr_genes.strand_column_indices(
            targets_df,
            output_targets_df,
            "+",
            flip_strand=False,
        ),
        np.asarray([0, 1, 2, 3]),
    )
    np.testing.assert_array_equal(
        test_tfr_genes.strand_column_indices(
            targets_df,
            output_targets_df,
            "-",
            flip_strand=False,
        ),
        np.asarray([0, 4, 5, 6]),
    )


def test_strand_column_indices_flip_strand_swaps_gene_orientation() -> None:
    targets_df = paired_targets_df()
    output_targets_df = test_tfr_genes.strand_output_rows(targets_df, flip_strand=True)

    assert output_targets_df["identifier"].tolist() == [
        "atac",
        "rna_a_reverse",
        "rna_b_reverse",
        "rna_c_reverse",
    ]
    np.testing.assert_array_equal(
        test_tfr_genes.strand_column_indices(
            targets_df,
            output_targets_df,
            "+",
            flip_strand=True,
        ),
        np.asarray([0, 4, 5, 6]),
    )
    np.testing.assert_array_equal(
        test_tfr_genes.strand_column_indices(
            targets_df,
            output_targets_df,
            "-",
            flip_strand=True,
        ),
        np.asarray([0, 1, 2, 3]),
    )
