#!/usr/bin/env python
"""Merge validated gene-variant shards without loading all scores into RAM."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import h5py
import numpy as np

from validate_genevar_shard import validate_shard


SHARED_ATTRS = (
    "score_name",
    "source_checkpoint",
    "source_pretrained_weights",
    "source_vcf",
    "source_variant_gene_map",
    "source_gtf",
    "source_fasta",
    "target_resolution",
    "crop_bins",
    "context_length",
    "rc_average",
    "pass_only",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    expected_start = 0
    for row in rows:
        start, end = int(row["start"]), int(row["end"])
        if start != expected_start or end <= start:
            raise ValueError(
                f"Noncontiguous manifest at chunk {row['chunk_id']}: "
                f"expected start {expected_start}, found [{start}, {end})"
            )
        expected_start = end
    return rows


def values_equal(left: object, right: object) -> bool:
    return bool(np.array_equal(np.asarray(left), np.asarray(right)))


def create_row_dataset(
    output: h5py.File,
    source: h5py.Dataset,
    path: str,
    total_rows: int,
) -> h5py.Dataset:
    parent_path, name = path.rsplit("/", 1)
    parent = output.require_group(parent_path)
    dataset = parent.create_dataset(
        name,
        shape=(total_rows, *source.shape[1:]),
        dtype=source.dtype,
        compression="gzip",
    )
    for attr, value in source.attrs.items():
        dataset.attrs[attr] = value
    return dataset


def main() -> None:
    args = parse_args()
    rows = manifest_rows(args.manifest)
    total_rows = int(rows[-1]["end"])
    if args.out.exists():
        raise FileExistsError(f"Refusing to replace existing output: {args.out}")
    partial_out = args.out.with_suffix(args.out.suffix + ".partial")
    if partial_out.exists():
        raise FileExistsError(f"Stale partial merge exists: {partial_out}")

    for row in rows:
        validate_shard(Path(row["output"]), int(row["start"]), int(row["end"]))

    first_path = Path(rows[0]["output"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(first_path, "r") as first, h5py.File(partial_out, "w") as output:
        for attr, value in first.attrs.items():
            output.attrs[attr] = value
        output.attrs["expanded_row_start"] = 0
        output.attrs["expanded_row_end"] = total_rows
        output.attrs["merged_shard_count"] = len(rows)
        output.attrs["merged_manifest"] = str(args.manifest.resolve())
        output.attrs["merged_shards"] = json.dumps(
            [str(Path(row["output"]).resolve()) for row in rows]
        )

        dataset_paths = [f"variants/{name}" for name in first["variants"]]
        for modality in first["scores"]:
            dataset_paths.extend(
                f"scores/{modality}/{name}" for name in first[f"scores/{modality}"]
            )
        destinations = {
            path: create_row_dataset(output, first[path], path, total_rows)
            for path in dataset_paths
        }
        first.copy("tracks", output)

        error_group = output.create_group("errors")
        error_indices = error_group.create_dataset(
            "index",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int64,
            compression="gzip",
        )
        error_reasons = error_group.create_dataset(
            "reason",
            shape=(0,),
            maxshape=(None,),
            dtype=first["errors/reason"].dtype,
            compression="gzip",
        )

        first_attrs = {attr: first.attrs[attr] for attr in SHARED_ATTRS}
        error_offset = 0
        for index, row in enumerate(rows, 1):
            path = Path(row["output"])
            start, end = int(row["start"]), int(row["end"])
            with h5py.File(path, "r") as source:
                for attr, expected in first_attrs.items():
                    if attr not in source.attrs or not values_equal(source.attrs[attr], expected):
                        raise ValueError(f"Shard {path} differs at attribute {attr!r}")
                for dataset_path, destination in destinations.items():
                    destination[start:end] = source[dataset_path][...]

                local_error_count = len(source["errors/index"])
                if local_error_count:
                    new_size = error_offset + local_error_count
                    error_indices.resize((new_size,))
                    error_reasons.resize((new_size,))
                    error_indices[error_offset:new_size] = (
                        source["errors/index"][...] + start
                    )
                    error_reasons[error_offset:new_size] = source["errors/reason"][...]
                    error_offset = new_size
            if index % 10 == 0 or index == len(rows):
                print(f"Merged {index}/{len(rows)} shards")

    os.replace(partial_out, args.out)
    print(f"Wrote merged inference file: {args.out}")
    print(f"Rows: {total_rows}; shards: {len(rows)}")


if __name__ == "__main__":
    main()
