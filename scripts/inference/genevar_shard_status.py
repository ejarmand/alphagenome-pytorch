#!/usr/bin/env python
"""Report progress for a manifest of gene-variant HDF5 shards."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from validate_genevar_shard import validate_shard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.manifest.open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    total_rows = sum(int(row["row_count"]) for row in rows)
    complete_rows = 0
    complete_chunks = 0
    invalid: list[str] = []
    partial: list[str] = []
    logged_progress: list[tuple[str, int, int]] = []
    for row in rows:
        output = Path(row["output"])
        partial_path = output.with_suffix(output.suffix + ".partial")
        if partial_path.exists():
            partial.append(str(partial_path))
        if not output.exists():
            log = Path(row["log"])
            if log.exists():
                text = log.read_text(errors="replace").replace("\r", "\n")
                matches = re.findall(
                    r"Scoring variants:.*?\|\s*(\d+)/(\d+)\s*\[",
                    text,
                )
                if matches:
                    current, total = map(int, matches[-1])
                    logged_progress.append((row["chunk_id"], current, total))
            continue
        try:
            validate_shard(output, int(row["start"]), int(row["end"]))
        except Exception as error:
            invalid.append(f"{output}: {error}")
            continue
        complete_chunks += 1
        complete_rows += int(row["row_count"])

    percent = 100.0 * complete_rows / total_rows
    print(
        f"Completed chunks: {complete_chunks}/{len(rows)}; "
        f"completed rows: {complete_rows}/{total_rows} ({percent:.3f}%)"
    )
    print(f"Invalid completed files: {len(invalid)}; partial files: {len(partial)}")
    for message in invalid[:5]:
        print(f"INVALID {message}")
    for path in partial[:5]:
        print(f"PARTIAL {path}")
    for chunk_id, current, total in logged_progress[:5]:
        print(
            f"LATEST_INCOMPLETE_LOG chunk={chunk_id} "
            f"rows={current}/{total} ({100.0 * current / total:.2f}%)"
        )


if __name__ == "__main__":
    main()
