#!/usr/bin/env python
"""Create a contiguous manifest for resumable gene-variant inference."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--row-count", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, default=10_000)
    parser.add_argument("--gpu-count", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.row_count <= 0 or args.chunk_size <= 0 or args.gpu_count <= 0:
        raise ValueError("row count, chunk size, and GPU count must be positive")
    if args.out.exists():
        raise FileExistsError(f"Refusing to replace existing manifest: {args.out}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    chunk_count = math.ceil(args.row_count / args.chunk_size)
    width = max(4, len(str(chunk_count - 1)))
    fields = [
        "chunk_id",
        "start",
        "end",
        "row_count",
        "gpu_slot",
        "output",
        "log",
    ]
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for index, start in enumerate(range(0, args.row_count, args.chunk_size)):
            end = min(start + args.chunk_size, args.row_count)
            chunk_id = f"{index:0{width}d}"
            stem = f"chunk_{chunk_id}_rows_{start:07d}_{end:07d}"
            writer.writerow(
                {
                    "chunk_id": chunk_id,
                    "start": start,
                    "end": end,
                    "row_count": end - start,
                    "gpu_slot": index % args.gpu_count,
                    "output": str((args.output_dir / f"{stem}.h5").resolve()),
                    "log": str((args.log_dir / f"{stem}.log").resolve()),
                }
            )

    print(f"Wrote manifest: {args.out}")
    print(
        f"Rows: {args.row_count}; chunk size: {args.chunk_size}; "
        f"chunks: {chunk_count}; GPU slots: {args.gpu_count}"
    )


if __name__ == "__main__":
    main()
