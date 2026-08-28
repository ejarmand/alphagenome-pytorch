#!/usr/bin/env python
"""Merge contiguous inference_geneVar.py HDF5 shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge ordered, contiguous gene-variant inference shards."
    )
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def values_equal(left: object, right: object) -> bool:
    return bool(np.array_equal(np.asarray(left), np.asarray(right)))


def validate_shards(handles: list[h5py.File]) -> tuple[list[int], list[int]]:
    required_attrs = (
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
    first = handles[0]
    for shard_number, handle in enumerate(handles[1:], 1):
        for attr in required_attrs:
            if attr not in first.attrs or attr not in handle.attrs:
                raise ValueError(f"Shard {shard_number} lacks required attribute {attr!r}")
            if not values_equal(first.attrs[attr], handle.attrs[attr]):
                raise ValueError(f"Shard {shard_number} differs at attribute {attr!r}")
        if set(first["variants"]) != set(handle["variants"]):
            raise ValueError(f"Shard {shard_number} has different variant datasets")
        if set(first["scores"]) != set(handle["scores"]):
            raise ValueError(f"Shard {shard_number} has different score groups")

    starts = [int(handle.attrs["expanded_row_start"]) for handle in handles]
    ends = [int(handle.attrs["expanded_row_end"]) for handle in handles]
    for index, handle in enumerate(handles):
        row_count = len(handle["variants/pos"])
        if ends[index] - starts[index] != row_count:
            raise ValueError(
                f"Shard {index} range [{starts[index]}, {ends[index]}) does not "
                f"match its {row_count} rows"
            )
        if index and starts[index] != ends[index - 1]:
            raise ValueError(
                f"Shards are not contiguous: shard {index - 1} ends at "
                f"{ends[index - 1]}, shard {index} starts at {starts[index]}"
            )
    return starts, ends


def create_concatenated_dataset(
    output: h5py.File,
    handles: list[h5py.File],
    dataset_path: str,
) -> None:
    source = handles[0][dataset_path]
    arrays = [handle[dataset_path][...] for handle in handles]
    data = np.concatenate(arrays, axis=0)
    parent_path, name = dataset_path.rsplit("/", 1)
    parent = output.require_group(parent_path)
    dataset = parent.create_dataset(
        name,
        data=data,
        dtype=source.dtype,
        compression="gzip",
    )
    for attr, value in source.attrs.items():
        dataset.attrs[attr] = value


def main() -> None:
    args = parse_args()
    if len(args.input) < 2:
        raise ValueError("Provide at least two --input shards")
    if args.out.exists() and not args.force:
        raise FileExistsError(f"Output already exists: {args.out}; use --force to replace it")

    handles = [h5py.File(path, "r") for path in args.input]
    try:
        starts, ends = validate_shards(handles)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(args.out, "w") as output:
            for attr, value in handles[0].attrs.items():
                output.attrs[attr] = value
            output.attrs["expanded_row_start"] = starts[0]
            output.attrs["expanded_row_end"] = ends[-1]
            output.attrs["merged_shard_count"] = len(handles)
            output.attrs["merged_shards"] = json.dumps(
                [str(path.resolve()) for path in args.input]
            )

            for name in handles[0]["variants"]:
                create_concatenated_dataset(output, handles, f"variants/{name}")

            for modality in handles[0]["scores"]:
                for score_name in handles[0][f"scores/{modality}"]:
                    create_concatenated_dataset(
                        output,
                        handles,
                        f"scores/{modality}/{score_name}",
                    )

            handles[0].copy("tracks", output)

            error_indices = []
            error_reasons = []
            row_offset = 0
            for handle in handles:
                error_indices.append(handle["errors/index"][...] + row_offset)
                error_reasons.append(handle["errors/reason"][...])
                row_offset += len(handle["variants/pos"])
            error_group = output.create_group("errors")
            error_group.create_dataset(
                "index",
                data=np.concatenate(error_indices),
                dtype=np.int64,
                compression="gzip",
            )
            error_group.create_dataset(
                "reason",
                data=np.concatenate(error_reasons),
                dtype=handles[0]["errors/reason"].dtype,
                compression="gzip",
            )
    finally:
        for handle in handles:
            handle.close()

    with h5py.File(args.out, "r") as output:
        row_count = len(output["variants/pos"])
        error_count = len(output["errors/index"])
        score_shapes = {
            modality: output[f"scores/{modality}/{output.attrs['score_name']}"].shape
            for modality in output["scores"]
        }
    print(f"Wrote merged inference file: {args.out}")
    print(f"Rows: {row_count}; errors: {error_count}; score shapes: {score_shapes}")


if __name__ == "__main__":
    main()
