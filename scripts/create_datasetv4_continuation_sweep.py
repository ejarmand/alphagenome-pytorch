#!/usr/bin/env python3
"""Create the DatasetV4 Bayesian continuation using completed prior runs."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import wandb
import yaml


DEFAULT_OLD_SWEEP = "ejarmand/rna_alphagenome_dog10k/43fvcxhv"
DEFAULT_CONFIG = Path(__file__).with_name(
    "datasetv4_heads_bayesian_continuation.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-sweep", default=DEFAULT_OLD_SWEEP)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--create",
        action="store_true",
        help="Create the W&B sweep. Without this flag, perform a dry run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entity, project, _ = args.old_sweep.split("/", maxsplit=2)

    config = yaml.safe_load(args.config.read_text())
    objective = config["metric"]["name"]

    if config["method"] != "bayes":
        raise SystemExit("Continuation config must use method: bayes")
    if config["run_cap"] != 1000:
        raise SystemExit("Continuation config must use run_cap: 1000")

    old_sweep = wandb.Api(timeout=60).sweep(args.old_sweep)
    old_runs = list(old_sweep.runs)
    unfinished = [run for run in old_runs if run.state != "finished"]

    if unfinished:
        details = ", ".join(
            f"{run.id}={run.state},epoch={run.summary.get('epoch')}"
            for run in unfinished
        )
        raise SystemExit(f"Old sweep still has unfinished runs: {details}")

    prior_ids = [run.id for run in old_runs]
    informative_ids: list[str] = []
    missing_objective_ids: list[str] = []
    for run in old_runs:
        value = run.summary.get(objective)
        if isinstance(value, (int, float)) and math.isfinite(value):
            informative_ids.append(run.id)
        else:
            missing_objective_ids.append(run.id)

    print(f"Old sweep: {args.old_sweep}")
    print(f"Completed prior runs to attach: {len(prior_ids)}")
    print(f"Prior runs with finite {objective}: {len(informative_ids)}")
    print(
        "Attached runs without a finite objective: "
        f"{missing_objective_ids}"
    )
    print(f"Continuation run cap: {config['run_cap']}")

    if not args.create:
        print("Dry run only. Re-run with --create after reviewing this summary.")
        return

    new_id = wandb.sweep(
        sweep=config,
        entity=entity,
        project=project,
        prior_runs=prior_ids,
    )
    new_path = new_id if "/" in new_id else f"{entity}/{project}/{new_id}"
    print(f"New sweep path: {new_path}")


if __name__ == "__main__":
    main()
