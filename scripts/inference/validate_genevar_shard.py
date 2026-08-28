#!/usr/bin/env python
"""Validate one completed gene-variant inference HDF5 shard."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py


REQUIRED_VARIANT_DATASETS = {
    "source_variant_index",
    "variant_key",
    "chrom",
    "pos",
    "ref",
    "alt",
    "id",
    "gene",
    "gene_name",
    "assignment",
    "distance_bp",
}


def validate_shard(path: Path, expected_start: int, expected_end: int) -> dict[str, int]:
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_rows = expected_end - expected_start
    with h5py.File(path, "r") as handle:
        start = int(handle.attrs["expanded_row_start"])
        end = int(handle.attrs["expanded_row_end"])
        if (start, end) != (expected_start, expected_end):
            raise ValueError(
                f"range is [{start}, {end}), expected [{expected_start}, {expected_end})"
            )
        missing = REQUIRED_VARIANT_DATASETS - set(handle["variants"])
        if missing:
            raise ValueError(f"missing variant datasets: {sorted(missing)}")
        row_count = len(handle["variants/pos"])
        if row_count != expected_rows:
            raise ValueError(f"contains {row_count} rows, expected {expected_rows}")
        score_name = str(handle.attrs["score_name"])
        modalities = list(handle["scores"])
        if not modalities:
            raise ValueError("contains no score groups")
        for modality in modalities:
            score_path = f"scores/{modality}/{score_name}"
            if score_path not in handle:
                raise ValueError(f"missing {score_path}")
            if handle[score_path].shape[0] != expected_rows:
                raise ValueError(
                    f"{score_path} has {handle[score_path].shape[0]} rows, "
                    f"expected {expected_rows}"
                )
        error_count = len(handle["errors/index"])
        if error_count:
            maximum_error_index = int(handle["errors/index"][:].max())
            if maximum_error_index >= expected_rows:
                raise ValueError(
                    f"error index {maximum_error_index} exceeds shard row count"
                )
    return {"rows": row_count, "errors": error_count}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--expected-start", type=int, required=True)
    parser.add_argument("--expected-end", type=int, required=True)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate_shard(args.path, args.expected_start, args.expected_end)
    if not args.quiet:
        print(
            f"Valid shard: {args.path}; rows={result['rows']}; "
            f"errors={result['errors']}"
        )


if __name__ == "__main__":
    main()
