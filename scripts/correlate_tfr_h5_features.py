#!/usr/bin/env python
"""Compute peak and gene cross-cell correlations from saved TFRecord eval HDF5s."""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
REPO_ROOT = Path(__file__).resolve().parents[3]
ALPHAGENOME_ROOT = Path(__file__).resolve().parents[2]
EVAL_UTILS_SRC = REPO_ROOT / "baskerville_dnn" / "cherry_tree" / "eval_utils" / "src"
if str(EVAL_UTILS_SRC) not in sys.path:
    sys.path.insert(0, str(EVAL_UTILS_SRC))

# Seq5 imports plotting helpers that import pyBigWig at module import time.
# This CLI only uses Seq5's HDF5/interval methods, so a stub is enough when
# pyBigWig is absent from the lightweight Alphagenome environment.
sys.modules.setdefault("pyBigWig", types.ModuleType("pyBigWig"))
from eval_utils.eval.utils import coefficient_of_variation, vectorized_pearson_correlation
from eval_utils.seq5.class_module import _construct_chrom_intervals, _sorted_overlap_ids


DEFAULT_PEAKS = REPO_ROOT / "baskerville_dnn" / "data" / "ref" / "peak_locs.bed"
DEFAULT_GENES = REPO_ROOT / "baskerville_dnn" / "data" / "genes.bed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute eval_nov.ipynb-style peak and gene cross-cell correlations from "
            "preds.h5/targets.h5 produced by eval_tfr_heads.py --save."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("model_dir", type=Path, help="Fine-tuned model/checkpoint run directory.")
    parser.add_argument("out_dir", type=Path, help="Directory for correlation TSV outputs.")
    parser.add_argument("--preds-h5", type=Path, default=None)
    parser.add_argument("--targets-h5", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--acc", type=Path, default=None, help="Track acc.txt from eval output.")
    parser.add_argument("--peaks", type=Path, default=DEFAULT_PEAKS)
    parser.add_argument("--genes", type=Path, default=DEFAULT_GENES)
    parser.add_argument("--split", default="test")
    parser.add_argument("--peak-min-overlap", type=float, default=498)
    parser.add_argument("--gene-min-coverage", type=float, default=0.99)
    parser.add_argument("--peak-chunk-size", type=int, default=50_000)
    parser.add_argument("--gene-chunk-size", type=int, default=2_000)
    parser.add_argument("--max-peaks", type=int, default=None, help="Optional debug limit.")
    parser.add_argument("--max-genes", type=int, default=None, help="Optional debug limit.")
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=None,
        help="Modalities to include. Defaults to all modalities in the saved target order.",
    )
    parser.add_argument("--skip-peaks", action="store_true")
    parser.add_argument("--skip-genes", action="store_true")
    return parser.parse_args()


def load_metadata(model_dir: Path) -> dict:
    metadata_path = model_dir / "metadata.json"
    if not metadata_path.exists():
        return {}
    with metadata_path.open() as metadata_file:
        return json.load(metadata_file)


def resolve_existing_path(path: Path | None, metadata_value: str | None, model_dir: Path) -> Path | None:
    if path is None and metadata_value:
        path = Path(metadata_value)
    if path is None:
        return None
    if path.exists() or path.is_absolute():
        return path
    for base in (Path.cwd(), model_dir, model_dir.parent, ALPHAGENOME_ROOT, REPO_ROOT):
        candidate = base / path
        if candidate.exists():
            return candidate
    return path


def default_eval_file(model_dir: Path, filename: str) -> Path | None:
    candidates = [
        model_dir / filename,
        model_dir / "eval" / filename,
        model_dir / "test" / filename,
        model_dir.parent / "eval" / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def read_h5_target_order(path: Path) -> tuple[list[str], dict[str, int]]:
    import h5py

    with h5py.File(path, "r") as h5_file:
        if "target_order" not in h5_file.attrs:
            raise ValueError(f"{path} is missing target_order HDF5 attribute")
        order = json.loads(h5_file.attrs["target_order"])
    return list(order["modalities"]), {str(k): int(v) for k, v in order["n_tracks_by_modality"].items()}


class H5FeatureExtractor:
    """Minimal Seq5-compatible extractor that respects cropped eval HDF5 geometry."""

    def __init__(self, data_dir: Path, h5_path: Path, data_key: str, split: str):
        import h5py

        self.data_dir = data_dir
        self.h5file = h5_path
        self.data_key = data_key
        self.tracks = pd.read_csv(data_dir / "targets.txt", sep="\t")
        self.seqs = pd.read_csv(data_dir / "sequences.bed", sep="\t", header=None)
        self.seqs.columns = ["chrom", "start", "end", "split"]
        self.seqs = self.seqs.loc[self.seqs["split"] == split].copy()
        self.seqs["array_idx"] = np.arange(len(self.seqs))
        self.chrom_intervals = _construct_chrom_intervals(self.seqs)
        with (data_dir / "statistics.json").open() as stats_file:
            self.stats = json.load(stats_file)
        with h5py.File(h5_path, "r") as h5_file:
            if data_key not in h5_file:
                raise ValueError(f"{data_key} not found in {h5_path}; keys={list(h5_file.keys())}")
            shape = h5_file[data_key].shape
        if shape[0] != self.seqs.shape[0]:
            raise ValueError(
                f"{h5_path} has {shape[0]} sequences, but {split} has {self.seqs.shape[0]}"
            )
        if shape[2] != self.tracks.shape[0]:
            raise ValueError(
                f"{h5_path} has {shape[2]} tracks, but targets.txt has {self.tracks.shape[0]}"
            )
        self.target_length = int(shape[1])
        self.seq_length = int(self.stats["seq_length"])
        self.crop_bp = int(self.stats.get("crop_bp", 0))
        covered_bp = self.seq_length - (2 * self.crop_bp)
        self.pool_width = covered_bp / self.target_length

    def _expand_interval(self, start: int, end: int) -> pd.DataFrame:
        starts = start + self.crop_bp + np.rint(np.arange(self.target_length) * self.pool_width).astype(int)
        ends = start + self.crop_bp + np.rint(np.arange(1, self.target_length + 1) * self.pool_width).astype(int)
        return pd.DataFrame({"start": starts, "end": ends})

    def bed_overlaps(
        self,
        bed_df: pd.DataFrame,
        chrom_col: str = "chrom",
        start_col: str = "start",
        end_col: str = "end",
        bin_level: bool = False,
        name_col: str | None = None,
        collect_indices: bool = False,
    ) -> pd.DataFrame:
        del bin_level, collect_indices
        required_cols = [chrom_col, start_col, end_col]
        if not all(col in bed_df.columns for col in required_cols):
            raise ValueError(f"Input DataFrame must contain the columns: {required_cols}")
        use_chrom = np.intersect1d(bed_df[chrom_col].unique(), list(self.chrom_intervals.keys()))
        if sum(self.chrom_intervals[chrom].shape[0] for chrom in use_chrom) == 0:
            raise AssertionError("no overlapping chromosomes in bedfile")

        all_overlaps = []
        for chrom in use_chrom:
            use_data = self.chrom_intervals[chrom]
            use_lookup = bed_df.loc[bed_df[chrom_col] == chrom]
            for interval in use_data.index:
                overlaps = use_lookup.loc[
                    (use_lookup[end_col] >= interval.left)
                    & (use_lookup[start_col] <= interval.right)
                ].copy()
                if overlaps.empty:
                    continue
                overlaps["seq5_start"] = interval.left
                overlaps["seq5_end"] = interval.right
                overlaps["overlap_length"] = np.minimum(overlaps[end_col], interval.right) - np.maximum(
                    overlaps[start_col], interval.left
                )
                overlaps["array_idx"] = int(use_data.loc[interval, "array_idx"])
                all_overlaps.append(overlaps)
        if not all_overlaps:
            columns = [
                chrom_col,
                "input_start",
                "input_end",
                "seq5_start",
                "seq5_end",
                "overlap_length",
                "array_idx",
            ]
            if name_col is not None:
                columns.append(name_col)
            return pd.DataFrame(columns=columns)
        all_overlaps = pd.concat(all_overlaps, axis=0)
        overlap_df = pd.DataFrame(
            {
                chrom_col: all_overlaps[chrom_col],
                "input_start": all_overlaps[start_col],
                "input_end": all_overlaps[end_col],
                "seq5_start": all_overlaps["seq5_start"],
                "seq5_end": all_overlaps["seq5_end"],
                "overlap_length": all_overlaps["overlap_length"],
                "array_idx": all_overlaps["array_idx"].astype(int),
            }
        )
        if name_col is not None:
            overlap_df[name_col] = all_overlaps[name_col]
        return overlap_df

    def get_covered_features(
        self,
        bed_df: pd.DataFrame,
        min_coverage: float = 0.9,
        chrom_col: str = "chrom",
        start_col: str = "start",
        end_col: str = "end",
        name_col: str | None = None,
        return_coverage: bool = True,
    ) -> pd.DataFrame:
        if name_col is None:
            bed_df = bed_df.copy()
            bed_df["name"] = (
                bed_df[chrom_col].astype(str)
                + "_"
                + bed_df[start_col].astype(str)
                + "_"
                + bed_df[end_col].astype(str)
            )
            name_col = "name"
        overlaps = self.bed_overlaps(bed_df, chrom_col, start_col, end_col, name_col=name_col)
        if overlaps.empty:
            return overlaps
        covered = overlaps.groupby(name_col)["overlap_length"].sum()
        if min_coverage > 1:
            keep = covered.loc[covered > min_coverage].index
        else:
            lengths = (
                overlaps.groupby(name_col)["input_end"].first()
                - overlaps.groupby(name_col)["input_start"].first()
            )
            pct_coverage = covered / lengths
            keep = pct_coverage.loc[pct_coverage > min_coverage].index
            if return_coverage:
                overlaps["pct_coverage"] = overlaps[name_col].map(pct_coverage)
        if return_coverage:
            overlaps["covered"] = overlaps[name_col].map(covered)
        return overlaps.loc[overlaps[name_col].isin(keep)]

    def extract_feature_matrix(
        self,
        bed_df: pd.DataFrame,
        chrom_col: str = "chrom",
        start_col: str = "start",
        end_col: str = "end",
        name_col: str | None = None,
        min_coverage: float = 0.9,
    ) -> pd.DataFrame:
        import h5py

        if name_col is None:
            bed_df = bed_df.copy()
            bed_df["name"] = (
                bed_df[chrom_col].astype(str)
                + "_"
                + bed_df[start_col].astype(str)
                + "_"
                + bed_df[end_col].astype(str)
            )
            name_col = "name"
        overlaps = self.get_covered_features(
            bed_df,
            min_coverage=min_coverage,
            chrom_col=chrom_col,
            start_col=start_col,
            end_col=end_col,
            name_col=name_col,
        )
        overlaps = overlaps.loc[~overlaps[name_col].duplicated(keep=False)]
        if overlaps.empty:
            return pd.DataFrame(columns=self.tracks["identifier"])

        vals = []
        names = []
        with h5py.File(self.h5file, "r") as h5_file:
            data = h5_file[self.data_key]
            for location, location_overlaps in overlaps.groupby(
                [chrom_col, "seq5_start", "seq5_end", "array_idx"]
            ):
                _, seq5_start, seq5_end, array_idx = location
                bins_df = self._expand_interval(int(seq5_start), int(seq5_end))
                starts, ends = _sorted_overlap_ids(
                    location_overlaps[["input_start", "input_end"]].values,
                    bins_df[["start", "end"]].values,
                )
                seq_data = data[int(array_idx), :, :]
                vals.append(
                    np.array([np.mean(seq_data[i:j, :], axis=0) for i, j in zip(starts, ends)])
                )
                names.append(location_overlaps[name_col].values)
        return pd.DataFrame(
            np.vstack(vals),
            index=np.concatenate(names),
            columns=self.tracks["identifier"],
        )


def read_acc_or_targets(acc_path: Path | None, data_dir: Path) -> pd.DataFrame:
    path = acc_path if acc_path is not None and acc_path.exists() else data_dir / "targets.txt"
    tracks = pd.read_csv(path, sep="\t")
    if "identifier" not in tracks.columns:
        raise ValueError(f"{path} must contain an identifier column")
    if "modality" not in tracks.columns:
        raise ValueError(f"{path} must contain a modality column")
    tracks = tracks.copy()
    if "index" not in tracks.columns:
        tracks["index"] = np.arange(len(tracks))
    return tracks


def target_tracks_for_h5(
    tracks: pd.DataFrame,
    modalities: list[str],
    n_tracks_by_modality: dict[str, int],
) -> pd.DataFrame:
    pieces = []
    h5_col = 0
    for modality in modalities:
        n_tracks = n_tracks_by_modality[modality]
        mod_tracks = tracks.loc[tracks["modality"] == modality].head(n_tracks).copy()
        if len(mod_tracks) != n_tracks:
            raise ValueError(
                f"Only found {len(mod_tracks)} {modality} tracks in metadata, expected {n_tracks}"
            )
        mod_tracks["h5_col"] = np.arange(h5_col, h5_col + n_tracks)
        h5_col += n_tracks
        pieces.append(mod_tracks)
    return pd.concat(pieces, axis=0, ignore_index=True)


def read_bed(path: Path, has_strand: bool) -> pd.DataFrame:
    usecols = [0, 1, 2, 3, 5] if has_strand else [0, 1, 2]
    names = ["chrom", "start", "end", "name", "strand"] if has_strand else ["chrom", "start", "end"]
    bed = pd.read_csv(path, sep="\t", header=None, usecols=usecols, names=names)
    bed["start"] = bed["start"].astype(int)
    bed["end"] = bed["end"].astype(int)
    if not has_strand:
        bed["name"] = (
            bed["chrom"].astype(str) + "_" + bed["start"].astype(str) + "_" + bed["end"].astype(str)
        )
    return bed


def iter_chunks(df: pd.DataFrame, chunk_size: int) -> Iterable[pd.DataFrame]:
    if chunk_size <= 0:
        yield df
        return
    for start in range(0, len(df), chunk_size):
        yield df.iloc[start : start + chunk_size].copy()


def feature_metrics(values: pd.DataFrame, prefix: str) -> pd.DataFrame:
    array = values.to_numpy(dtype=np.float64, copy=False)
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = coefficient_of_variation(array)
    return pd.DataFrame(
        {
            f"{prefix}_mean": np.nanmean(array, axis=1),
            f"{prefix}_max": np.nanmax(array, axis=1),
            f"{prefix}_coefficient_of_variation": cv,
        },
        index=values.index,
    )


def long_correlations(
    pred_matrix: pd.DataFrame,
    target_matrix: pd.DataFrame,
    tracks: pd.DataFrame,
    feature_col: str,
    strand_by_feature: pd.Series | None = None,
) -> pd.DataFrame:
    rows = []
    target_metric_frames = []
    pred_metric_frames = []
    for modality, mod_tracks in tracks.groupby("modality", sort=False):
        if modality == "RNA" and strand_by_feature is not None:
            feature_rows = []
            for strand, strand_tracks in (("+", "forward"), ("-", "reverse")):
                features = strand_by_feature.index[strand_by_feature == strand]
                mod_ids = mod_tracks.loc[
                    mod_tracks["identifier"].str.contains(strand_tracks, case=False, regex=False),
                    "identifier",
                ].to_list()
                common_features = pred_matrix.index.intersection(features)
                mod_ids = [identifier for identifier in mod_ids if identifier in pred_matrix.columns]
                if len(mod_ids) < 2 or len(common_features) == 0:
                    continue
                preds = pred_matrix.loc[common_features, mod_ids]
                targets = target_matrix.loc[common_features, mod_ids]
                feature_rows.append((preds, targets))
        else:
            mod_ids = [
                identifier
                for identifier in mod_tracks["identifier"].to_list()
                if identifier in pred_matrix.columns
            ]
            if len(mod_ids) < 2:
                continue
            feature_rows = [(pred_matrix[mod_ids], target_matrix[mod_ids])]

        for preds, targets in feature_rows:
            with np.errstate(divide="ignore", invalid="ignore"):
                corrs = vectorized_pearson_correlation(
                    preds.to_numpy(dtype=np.float64, copy=False),
                    targets.to_numpy(dtype=np.float64, copy=False),
                )
            rows.append(
                pd.DataFrame(
                    {
                        feature_col: preds.index,
                        "modality": modality,
                        "pearsonr": corrs,
                        "n_tracks": preds.shape[1],
                    }
                )
            )
            target_metric_frames.append(
                feature_metrics(targets, "target").assign(
                    **{feature_col: targets.index, "modality": modality}
                )
            )
            pred_metric_frames.append(
                feature_metrics(preds, "pred").assign(
                    **{feature_col: preds.index, "modality": modality}
                )
            )
    if not rows:
        return pd.DataFrame(columns=[feature_col, "modality", "pearsonr", "n_tracks"])
    corr_df = pd.concat(rows, axis=0, ignore_index=True)
    metrics = pd.concat(target_metric_frames + pred_metric_frames, axis=0)
    metric_cols = [col for col in metrics.columns if col not in {feature_col, "modality"}]
    metrics = metrics.groupby([feature_col, "modality"], sort=False)[metric_cols].first().reset_index()
    return corr_df.merge(metrics, on=[feature_col, "modality"], how="left")


def write_summary(correlations: pd.DataFrame, path: Path) -> None:
    summary = (
        correlations.groupby("modality", sort=False)
        .agg(
            n=("pearsonr", "size"),
            n_finite=("pearsonr", lambda x: np.isfinite(x).sum()),
            mean_pearsonr=("pearsonr", "mean"),
            median_pearsonr=("pearsonr", "median"),
        )
        .reset_index()
    )
    summary.to_csv(path, sep="\t", index=False)


def compute_peak_correlations(
    target_data: H5FeatureExtractor,
    pred_data: H5FeatureExtractor,
    peaks: pd.DataFrame,
    tracks: pd.DataFrame,
    out_dir: Path,
    min_overlap: float,
    chunk_size: int,
) -> None:
    out_path = out_dir / "peak_correlations.tsv"
    wrote_header = False
    for chunk_i, chunk in enumerate(iter_chunks(peaks, chunk_size), start=1):
        overlaps = target_data.bed_overlaps(chunk)
        overlaps = overlaps.loc[overlaps["overlap_length"] >= min_overlap].copy()
        if overlaps.empty:
            continue
        overlaps["loc"] = (
            overlaps["chrom"].astype(str)
            + "_"
            + overlaps["input_start"].astype(str)
            + "_"
            + overlaps["input_end"].astype(str)
        )
        target_matrix = target_data.extract_feature_matrix(
            overlaps,
            start_col="input_start",
            end_col="input_end",
            name_col="loc",
            min_coverage=0,
        )
        pred_matrix = pred_data.extract_feature_matrix(
            overlaps,
            start_col="input_start",
            end_col="input_end",
            name_col="loc",
            min_coverage=0,
        )
        corr_df = long_correlations(pred_matrix, target_matrix, tracks, "loc")
        corr_df.to_csv(out_path, sep="\t", index=False, mode="a", header=not wrote_header)
        wrote_header = True
        print(f"processed peak chunk {chunk_i}: {len(corr_df)} correlations", flush=True)
    if not wrote_header:
        pd.DataFrame(columns=["loc", "modality", "pearsonr", "n_tracks"]).to_csv(
            out_path, sep="\t", index=False
        )
    write_summary(pd.read_csv(out_path, sep="\t"), out_dir / "peak_correlation_summary.tsv")


def compute_gene_correlations(
    target_data: H5FeatureExtractor,
    pred_data: H5FeatureExtractor,
    genes: pd.DataFrame,
    tracks: pd.DataFrame,
    out_dir: Path,
    min_coverage: float,
    chunk_size: int,
) -> None:
    out_path = out_dir / "gene_correlations.tsv"
    wrote_header = False
    genes = genes.drop_duplicates("name").copy()
    for chunk_i, chunk in enumerate(iter_chunks(genes, chunk_size), start=1):
        overlaps = target_data.get_covered_features(
            chunk,
            min_coverage=min_coverage,
            name_col="name",
            return_coverage=True,
        )
        if overlaps.empty:
            continue
        target_matrix = target_data.extract_feature_matrix(
            overlaps,
            start_col="input_start",
            end_col="input_end",
            name_col="name",
            min_coverage=0,
        )
        pred_matrix = pred_data.extract_feature_matrix(
            overlaps,
            start_col="input_start",
            end_col="input_end",
            name_col="name",
            min_coverage=0,
        )
        strand_by_gene = chunk.set_index("name")["strand"]
        corr_df = long_correlations(pred_matrix, target_matrix, tracks, "gene", strand_by_gene)
        corr_df.to_csv(out_path, sep="\t", index=False, mode="a", header=not wrote_header)
        wrote_header = True
        print(f"processed gene chunk {chunk_i}: {len(corr_df)} correlations", flush=True)
    if not wrote_header:
        pd.DataFrame(columns=["gene", "modality", "pearsonr", "n_tracks"]).to_csv(
            out_path, sep="\t", index=False
        )
    write_summary(pd.read_csv(out_path, sep="\t"), out_dir / "gene_correlation_summary.tsv")


def main() -> None:
    args = parse_args()
    metadata = load_metadata(args.model_dir)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    preds_h5 = args.preds_h5 or default_eval_file(args.model_dir, "preds.h5")
    targets_h5 = (
        args.targets_h5
        or default_eval_file(args.model_dir, "targets.h5")
    )
    if preds_h5 is None or targets_h5 is None:
        raise SystemExit("Could not find preds.h5/targets.h5; pass --preds-h5 and --targets-h5")

    data_dir = resolve_existing_path(args.data_dir, metadata.get("data_dir"), args.model_dir)
    if data_dir is None and (ALPHAGENOME_ROOT / "tfr_data").exists():
        data_dir = ALPHAGENOME_ROOT / "tfr_data"
    if data_dir is None or not data_dir.exists():
        raise SystemExit("Could not resolve data_dir; pass --data-dir")

    modalities, n_tracks_by_modality = read_h5_target_order(preds_h5)
    if args.modalities is not None:
        modalities = [modality for modality in modalities if modality in set(args.modalities)]
        n_tracks_by_modality = {
            modality: n_tracks_by_modality[modality] for modality in modalities
        }
    tracks = target_tracks_for_h5(read_acc_or_targets(args.acc, data_dir), modalities, n_tracks_by_modality)

    target_data = H5FeatureExtractor(data_dir=data_dir, h5_path=targets_h5, data_key="targets", split=args.split)
    pred_data = H5FeatureExtractor(data_dir=data_dir, h5_path=preds_h5, data_key="preds", split=args.split)

    if not args.skip_peaks:
        peaks = read_bed(args.peaks, has_strand=False)
        if args.max_peaks is not None:
            peaks = peaks.head(args.max_peaks)
        compute_peak_correlations(
            target_data,
            pred_data,
            peaks,
            tracks,
            out_dir,
            args.peak_min_overlap,
            args.peak_chunk_size,
        )
    if not args.skip_genes:
        genes = read_bed(args.genes, has_strand=True)
        if args.max_genes is not None:
            genes = genes.head(args.max_genes)
        compute_gene_correlations(
            target_data,
            pred_data,
            genes,
            tracks,
            out_dir,
            args.gene_min_coverage,
            args.gene_chunk_size,
        )


if __name__ == "__main__":
    main()
