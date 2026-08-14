#!/usr/bin/env python3
"""Extract and visualize TFR fine-tuned cell embedding head weights."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


DEFAULT_COLORS = {
    "Glut": "#2B93DF",
    "Gaba": "#FF3358",
    "NN": "#666666",
    "Imn": "#03EDFF",
    "Other": "#8A8A8A",
    "other_nt": "#0a9964",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract per-cell projection weights from TFRHeads checkpoints and "
            "make low-dimensional PCA/UMAP visualizations."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("final_model/best_heads.pt"),
        help="Path to a TFR heads checkpoint.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help=(
            "Path to metadata.json. Defaults to metadata.json next to the checkpoint, "
            "then checkpoint['metadata'] if present."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("cell_embedding_viz/final_model"),
        help="Output directory for CSVs and plots.",
    )
    parser.add_argument(
        "--feature-mode",
        choices=("flatten", "summary"),
        default="flatten",
        help=(
            "flatten uses all projection weights; summary uses compact row-wise "
            "statistics plus bias."
        ),
    )
    parser.add_argument(
        "--append-bias",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append each cell projection bias to the feature vector.",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=50,
        help="Number of PCA components to compute/save before taking PC1/PC2 plots.",
    )
    parser.add_argument(
        "--umap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run UMAP if umap-learn is installed.",
    )
    parser.add_argument("--umap-neighbors", type=int, default=20)
    parser.add_argument("--umap-min-dist", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def strip_strand(cell_name: str) -> str:
    return cell_name.split("|strand=")[0]


def module_key(name: str, used: set[str]) -> str:
    key = re.sub(r"[^A-Za-z0-9_]", "_", name).strip("_") or "cell"
    if key[0].isdigit():
        key = f"cell_{key}"
    base = key
    suffix = 1
    while key in used:
        suffix += 1
        key = f"{base}_{suffix}"
    used.add(key)
    return key


def cell_class(cell_name: str) -> str:
    name = cell_name.upper()
    if "DOPA" in name or "SERO" in name or "SER" in name or "CHOL" in name:
        return "other_nt"
    if "GLUT" in name:
        return "Glut"
    if "GABA" in name:
        return "Gaba"
    if "NN" in name:
        return "NN"
    if "IMN" in name:
        return "Imn"
    return "Other"


def load_checkpoint_and_metadata(
    checkpoint_path: Path,
    metadata_path: Path | None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"{checkpoint_path} did not load as a checkpoint dictionary")

    cell_state = checkpoint.get("cell_layers_state_dict")
    if cell_state is None:
        heads_state = checkpoint.get("heads_state_dict") or checkpoint.get(
            "heads_model_state_dict"
        )
        if heads_state is None:
            raise KeyError(
                "Checkpoint does not contain cell_layers_state_dict or heads_state_dict"
            )
        cell_state = {
            key.removeprefix("cell_layers."): value
            for key, value in heads_state.items()
            if key.startswith("cell_layers.")
        }
    if not cell_state:
        raise KeyError("No cell layer tensors were found in the checkpoint")

    if metadata_path is None:
        inferred_metadata_path = checkpoint_path.with_name("metadata.json")
        metadata_path = inferred_metadata_path if inferred_metadata_path.exists() else None

    if metadata_path is not None:
        metadata = json.loads(metadata_path.read_text())
    else:
        metadata = checkpoint.get("metadata")
        if metadata is None:
            raise KeyError("No metadata path was provided and checkpoint has no metadata")

    return dict(cell_state), metadata


def unique_metadata_cells(metadata: dict[str, Any]) -> list[str]:
    cell_types_by_modality = metadata.get("cell_types")
    if not isinstance(cell_types_by_modality, dict):
        raise KeyError("metadata must contain a cell_types dictionary")

    cells: list[str] = []
    seen: set[str] = set()
    for cell_types in cell_types_by_modality.values():
        for raw_cell in cell_types:
            cell = strip_strand(str(raw_cell))
            if cell not in seen:
                cells.append(cell)
                seen.add(cell)
    return cells


def metadata_key_map(cells: list[str]) -> dict[str, str]:
    used: set[str] = set()
    return {cell: module_key(cell, used) for cell in cells}


def checkpoint_cell_keys(cell_state: dict[str, torch.Tensor]) -> set[str]:
    return {
        key[: -len(".proj.weight")]
        for key in cell_state
        if key.endswith(".proj.weight")
    }


def build_feature_matrix(
    cell_state: dict[str, torch.Tensor],
    cell_to_key: dict[str, str],
    *,
    feature_mode: str,
    append_bias: bool,
) -> tuple[np.ndarray, pd.DataFrame]:
    rows: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    missing: list[str] = []

    for cell_name, cell_key in cell_to_key.items():
        weight_key = f"{cell_key}.proj.weight"
        bias_key = f"{cell_key}.proj.bias"
        if weight_key not in cell_state:
            missing.append(cell_key)
            continue

        weight = cell_state[weight_key].detach().float().squeeze(-1).numpy()
        if weight.ndim != 2:
            raise ValueError(f"{weight_key} should be 2D after squeeze, got {weight.shape}")

        if feature_mode == "flatten":
            features = weight.reshape(-1)
        elif feature_mode == "summary":
            features = np.concatenate(
                [
                    weight.mean(axis=1),
                    weight.std(axis=1),
                    np.linalg.norm(weight, axis=1),
                    weight.mean(axis=0),
                    weight.std(axis=0),
                ]
            )
        else:
            raise ValueError(f"Unknown feature mode: {feature_mode}")

        if append_bias:
            bias = cell_state[bias_key].detach().float().numpy()
            features = np.concatenate([features, bias.reshape(-1)])

        rows.append(features.astype(np.float32, copy=False))
        records.append(
            {
                "cell_name": cell_name,
                "cell_key": cell_key,
                "cell_class": cell_class(cell_name),
            }
        )

    if missing:
        preview = ", ".join(missing[:10])
        raise KeyError(f"{len(missing)} metadata cells missing from checkpoint: {preview}")

    return np.vstack(rows), pd.DataFrame.from_records(records)


def standardize(features: np.ndarray) -> np.ndarray:
    centered = features - features.mean(axis=0, keepdims=True)
    scale = centered.std(axis=0, keepdims=True)
    scale[scale == 0.0] = 1.0
    return centered / scale


def pca(features: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    x = standardize(features.astype(np.float64, copy=False))
    n_components = max(2, min(n_components, x.shape[0] - 1, x.shape[1]))
    _, singular_values, vt = np.linalg.svd(x, full_matrices=False)
    coords = x @ vt[:n_components].T
    variance = singular_values**2 / max(1, x.shape[0] - 1)
    explained = variance / variance.sum()
    return coords.astype(np.float32), explained[:n_components].astype(np.float32)


def maybe_umap(
    pca_coords: np.ndarray,
    *,
    n_neighbors: int,
    min_dist: float,
    seed: int,
) -> np.ndarray | None:
    try:
        import umap  # type: ignore[import-not-found]
    except ImportError:
        return None

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=seed,
    )
    return reducer.fit_transform(pca_coords).astype(np.float32)


def plot_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    output_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6.2), dpi=160)
    for class_name in sorted(df["cell_class"].unique()):
        subset = df[df["cell_class"] == class_name]
        ax.scatter(
            subset[x_col],
            subset[y_col],
            s=28,
            alpha=0.82,
            linewidths=0,
            label=f"{class_name} (n={len(subset)})",
            color=DEFAULT_COLORS.get(class_name, DEFAULT_COLORS["Other"]),
        )
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    cell_state, metadata = load_checkpoint_and_metadata(args.checkpoint, args.metadata)
    cells = unique_metadata_cells(metadata)
    cell_to_key = metadata_key_map(cells)
    observed_keys = checkpoint_cell_keys(cell_state)
    expected_keys = set(cell_to_key.values())

    extra_keys = observed_keys - expected_keys
    missing_keys = expected_keys - observed_keys
    if extra_keys or missing_keys:
        raise RuntimeError(
            "Metadata/checkpoint cell key mismatch: "
            f"{len(missing_keys)} missing, {len(extra_keys)} extra. "
            f"missing={sorted(missing_keys)[:5]} extra={sorted(extra_keys)[:5]}"
        )

    features, df = build_feature_matrix(
        cell_state,
        cell_to_key,
        feature_mode=args.feature_mode,
        append_bias=args.append_bias,
    )
    if metadata.get("n_cell_types") is not None and len(df) != int(metadata["n_cell_types"]):
        raise RuntimeError(
            f"Extracted {len(df)} cells, metadata expects {metadata['n_cell_types']}"
        )

    pca_coords, explained = pca(features, args.pca_components)
    df["pca_1"] = pca_coords[:, 0]
    df["pca_2"] = pca_coords[:, 1]
    for idx in range(pca_coords.shape[1]):
        df[f"pc_{idx + 1}"] = pca_coords[:, idx]

    features_path = args.out_dir / "cell_embedding_features.npz"
    np.savez_compressed(
        features_path,
        features=features,
        cell_name=df["cell_name"].to_numpy(),
        cell_key=df["cell_key"].to_numpy(),
        cell_class=df["cell_class"].to_numpy(),
    )

    np.savetxt(
        args.out_dir / "pca_explained_variance.tsv",
        np.column_stack([np.arange(1, len(explained) + 1), explained]),
        fmt=["%d", "%.8g"],
        delimiter="\t",
        header="pc\texplained_variance_ratio",
        comments="",
    )

    if args.umap:
        umap_coords = maybe_umap(
            pca_coords,
            n_neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist,
            seed=args.seed,
        )
        if umap_coords is not None:
            df["umap_1"] = umap_coords[:, 0]
            df["umap_2"] = umap_coords[:, 1]

    df.to_csv(args.out_dir / "cell_embedding_coordinates.tsv", sep="\t", index=False)

    class_counts = Counter(df["cell_class"])
    pd.DataFrame(
        [{"cell_class": key, "n": value} for key, value in sorted(class_counts.items())]
    ).to_csv(args.out_dir / "cell_class_counts.tsv", sep="\t", index=False)

    plot_scatter(
        df,
        "pca_1",
        "pca_2",
        args.out_dir / "cell_embedding_pca.png",
        "TFR Cell Projection Weights: PCA",
    )
    if "umap_1" in df:
        plot_scatter(
            df,
            "umap_1",
            "umap_2",
            args.out_dir / "cell_embedding_umap.png",
            "TFR Cell Projection Weights: UMAP",
        )

    print(f"Extracted {len(df)} cells from {args.checkpoint}")
    print(f"Feature matrix: {features.shape[0]} cells x {features.shape[1]} features")
    print("Cell classes:", dict(sorted(class_counts.items())))
    print(f"PCA variance PC1/PC2: {explained[0]:.4f}, {explained[1]:.4f}")
    if args.umap and "umap_1" not in df:
        print("UMAP skipped: install umap-learn to enable it")
    print(f"Wrote outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
