#!/usr/bin/env python
"""Gene-level evaluation for AlphaGenome TFRecord fine-tuned heads.

This mirrors the analysis performed by Borzoi's ``borzoi_test_genes.py``:
per-bin predictions and targets are aggregated over gene exons or spans,
written as gene-by-track matrices, and summarized with per-track Pearson/R2
metrics plus within-gene profile correlations.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from alphagenome_pytorch import AlphaGenome
from alphagenome_pytorch.config import DtypePolicy
from alphagenome_pytorch.extensions.finetuning.transfer import load_trunk, remove_all_heads

try:
    from scripts.eval_tfr_heads import (
        create_heads_model,
        load_checkpoint,
        load_metadata,
        resolve_checkpoint,
        resolve_modalities,
        resolve_path,
    )
    from scripts.finetune_tfr_heads import (
        TFRHeads,
        cleanup_torchrun,
        create_loader,
        create_tfr_dataset,
        forward_heads,
        is_main_process,
        setup_torchrun,
        unpack_tfr_batch,
    )
except ModuleNotFoundError:
    from eval_tfr_heads import (  # type: ignore[no-redef]
        create_heads_model,
        load_checkpoint,
        load_metadata,
        resolve_checkpoint,
        resolve_modalities,
        resolve_path,
    )
    from finetune_tfr_heads import (  # type: ignore[no-redef]
        TFRHeads,
        cleanup_torchrun,
        create_loader,
        create_tfr_dataset,
        forward_heads,
        is_main_process,
        setup_torchrun,
        unpack_tfr_batch,
    )


@dataclass(frozen=True)
class GeneSegment:
    chrom: str
    start: int
    end: int
    gene_id: str
    strand: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate AlphaGenome TFRecord heads at gene level.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Checkpoint file, or run directory containing best_heads.pt/last_heads.pt.",
    )
    parser.add_argument("genes_gtf", type=Path, help="Gene annotation GTF.")
    parser.add_argument("--checkpoint-name", default="best_heads.pt")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--pretrained-weights", type=Path, default=None)
    parser.add_argument(
        "--modality",
        action="append",
        default=None,
        help="Modality to evaluate. Repeat to evaluate a subset.",
    )
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--pooling", choices=["mean", "sum"], default=None)
    parser.add_argument("--target-resolution", type=int, choices=[32, 128], default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-n", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--tfr-num-parallel-reads", type=int, default=1)
    parser.add_argument("-o", "--out-dir", type=Path, default=Path("testg_out"))
    parser.add_argument("--span", action="store_true", help="Aggregate full gene spans.")
    parser.add_argument(
        "--store-span",
        action="store_true",
        help="Store per-bin gene profiles under gene_within/.",
    )
    parser.add_argument(
        "--pseudo-qtl",
        type=float,
        default=None,
        help="Quantile of nonzero gene coverage to add as pseudo counts.",
    )
    parser.add_argument(
        "--drop-length-norm",
        action="store_true",
        help="Keep per-base mean coverage instead of scaling by gene length.",
    )
    parser.add_argument(
        "--flip-strand",
        action="store_true",
        help="Swap which target strand is used for +/- strand genes.",
    )
    parser.add_argument("--crop-bins", type=int, default=None)
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', 'cuda', or e.g. 'cuda:0'")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default=None)
    parser.add_argument("--organism-idx", type=int, default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def parse_gtf_attributes(attributes: str) -> dict[str, str]:
    parsed = {}
    for field in attributes.rstrip(";").split(";"):
        field = field.strip()
        if not field:
            continue
        if " " in field:
            key, value = field.split(" ", 1)
            parsed[key] = value.strip().strip('"')
        elif "=" in field:
            key, value = field.split("=", 1)
            parsed[key] = value.strip().strip('"')
    return parsed


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    merged = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def load_gene_segments(genes_gtf: Path, *, span: bool) -> list[GeneSegment]:
    gene_exons: dict[str, list[tuple[int, int]]] = defaultdict(list)
    gene_meta: dict[str, tuple[str, str]] = {}
    gene_span: dict[str, tuple[int, int]] = {}

    with genes_gtf.open() as gtf_file:
        for line in gtf_file:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9:
                continue
            chrom, _, feature, start_s, end_s, _, strand, _, attrs_s = fields
            attrs = parse_gtf_attributes(attrs_s)
            gene_id = attrs.get("gene_id") or attrs.get("gene_name")
            if not gene_id:
                continue
            start = int(start_s) - 1
            end = int(end_s)
            gene_meta.setdefault(gene_id, (chrom, strand))
            prev = gene_span.get(gene_id)
            gene_span[gene_id] = (
                start if prev is None else min(prev[0], start),
                end if prev is None else max(prev[1], end),
            )
            if feature == "exon":
                gene_exons[gene_id].append((start, end))

    segments = []
    if span:
        for gene_id, (start, end) in gene_span.items():
            chrom, strand = gene_meta[gene_id]
            segments.append(GeneSegment(chrom, start, end, gene_id, strand))
        return remove_overlapping_gene_spans(segments)

    for gene_id, intervals in gene_exons.items():
        chrom, strand = gene_meta[gene_id]
        for start, end in merge_intervals(intervals):
            segments.append(GeneSegment(chrom, start, end, gene_id, strand))
    return remove_exons_overlapping_other_genes(segments)


def remove_exons_overlapping_other_genes(segments: list[GeneSegment]) -> list[GeneSegment]:
    by_key: dict[tuple[str, str], list[GeneSegment]] = defaultdict(list)
    for segment in segments:
        by_key[(segment.chrom, segment.strand)].append(segment)

    overlapping = set()
    for key_segments in by_key.values():
        active: list[GeneSegment] = []
        for segment in sorted(key_segments, key=lambda item: (item.start, item.end)):
            active = [item for item in active if item.end > segment.start]
            for other in active:
                if other.gene_id != segment.gene_id and other.end > segment.start:
                    overlapping.add((other.gene_id, other.start, other.end))
                    overlapping.add((segment.gene_id, segment.start, segment.end))
            active.append(segment)

    return [
        segment
        for segment in segments
        if (segment.gene_id, segment.start, segment.end) not in overlapping
    ]


def remove_overlapping_gene_spans(segments: list[GeneSegment]) -> list[GeneSegment]:
    by_key: dict[tuple[str, str], list[GeneSegment]] = defaultdict(list)
    for segment in segments:
        by_key[(segment.chrom, segment.strand)].append(segment)

    overlapping_genes = set()
    for key_segments in by_key.values():
        active: list[GeneSegment] = []
        for segment in sorted(key_segments, key=lambda item: (item.start, item.end)):
            active = [item for item in active if item.end > segment.start]
            for other in active:
                if other.gene_id != segment.gene_id and other.end > segment.start:
                    overlapping_genes.add(other.gene_id)
                    overlapping_genes.add(segment.gene_id)
            active.append(segment)

    return [segment for segment in segments if segment.gene_id not in overlapping_genes]


def write_genes_bed(path: Path, segments: list[GeneSegment]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as output_file:
        for segment in sorted(segments, key=lambda item: (item.chrom, item.start, item.end)):
            print(
                "\t".join(
                    [
                        segment.chrom,
                        str(segment.start),
                        str(segment.end),
                        segment.gene_id,
                        ".",
                        segment.strand,
                    ]
                ),
                file=output_file,
            )


def read_sequences(data_dir: Path, split: str) -> pd.DataFrame:
    seqs = pd.read_csv(
        data_dir / "sequences.bed",
        sep="\t",
        names=["Chromosome", "Start", "End", "Name"],
    )
    seqs = seqs[seqs.Name == split].reset_index(drop=True)
    return seqs


def overlap_segments_by_sequence(
    seqs_df: pd.DataFrame,
    segments: list[GeneSegment],
) -> list[list[GeneSegment]]:
    by_chrom: dict[str, list[GeneSegment]] = defaultdict(list)
    for segment in segments:
        by_chrom[segment.chrom].append(segment)
    for chrom in by_chrom:
        by_chrom[chrom].sort(key=lambda item: item.start)

    overlaps = []
    for seq in seqs_df.itertuples(index=False):
        seq_overlaps = [
            segment
            for segment in by_chrom.get(seq.Chromosome, [])
            if segment.end > seq.Start and segment.start < seq.End
        ]
        overlaps.append(seq_overlaps)
    return overlaps


def target_metadata(dataset: Any, modalities: list[str]) -> pd.DataFrame:
    rows = []
    global_index = 0
    for modality in modalities:
        for local_index, row in enumerate(dataset.target_rows_by_modality[modality]):
            copied = dict(row)
            copied["modality"] = modality
            copied["local_index"] = local_index
            copied["global_index"] = global_index
            rows.append(copied)
            global_index += 1
    return pd.DataFrame(rows)


def strand_output_rows(targets_df: pd.DataFrame, *, flip_strand: bool) -> pd.DataFrame:
    drop_strand = "+" if flip_strand else "-"
    return targets_df[targets_df.get("strand", "") != drop_strand].reset_index(drop=True)


def strand_mask(targets_df: pd.DataFrame, gene_strand: str, *, flip_strand: bool) -> np.ndarray:
    if (gene_strand == "+") ^ flip_strand:
        return (targets_df.get("strand", "") != "-").to_numpy()
    return (targets_df.get("strand", "") != "+").to_numpy()


def finite_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.var(x) <= 1e-12 or np.var(y) <= 1e-12:
        return float("nan")
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = np.sqrt(np.sum(x_centered * x_centered) * np.sum(y_centered * y_centered))
    if denominator <= 0:
        return float("nan")
    return float(np.sum(x_centered * y_centered) / denominator)


def explained_variance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    variance = np.var(y_true)
    if variance <= 1e-12:
        return float("nan")
    return float(1.0 - np.var(y_true - y_pred) / variance)


def quantile_normalize_matrix(values: np.ndarray) -> np.ndarray:
    sorted_values = np.sort(values, axis=0)
    rank_means = sorted_values.mean(axis=1)
    order = np.argsort(values, axis=0, kind="mergesort")
    normalized = np.empty_like(values, dtype=np.float32)
    for col_i in range(values.shape[1]):
        normalized[order[:, col_i], col_i] = rank_means
    return normalized


def nanmean(values: list[float] | np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).any():
        return float("nan")
    return float(np.nanmean(array))


def aggregate_gene_predictions(
    model: torch.nn.Module,
    heads_model: TFRHeads,
    loader: DataLoader,
    seqs_df: pd.DataFrame,
    seq_gene_overlaps: list[list[GeneSegment]],
    targets_df: pd.DataFrame,
    device: torch.device,
    args: argparse.Namespace,
    crop_bins: int,
    target_resolution: int,
) -> tuple[dict[str, list[np.ndarray]], dict[str, list[np.ndarray]], dict[str, str]]:
    model.eval()
    heads_model.eval()

    gene_preds: dict[str, list[np.ndarray]] = defaultdict(list)
    gene_targets: dict[str, list[np.ndarray]] = defaultdict(list)
    gene_strands: dict[str, str] = {}
    strand_pair_by_modality = {
        modality: torch.as_tensor(indices, dtype=torch.long, device=device)
        for modality, indices in loader.dataset.strand_pair_by_modality.items()
    }
    modalities = list(loader.dataset.modalities)

    seq_i = 0
    pbar = tqdm(total=args.max_steps, desc=args.split, disable=args.quiet)
    data_iter = iter(loader)
    steps = 0
    while args.max_steps is None or steps < args.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            break
        sequences, modality_targets, augmentation = unpack_tfr_batch(batch)
        batch_size = sequences.shape[0]
        batch_overlaps = seq_gene_overlaps[seq_i:seq_i + batch_size]

        if any(batch_overlaps):
            sequences = sequences.to(device, non_blocking=True)
            organism_idx = torch.full(
                (batch_size,),
                args.organism_idx,
                dtype=torch.long,
                device=device,
            )
            head_organism_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
            reverse_complement = None
            if augmentation is not None:
                reverse_complement = augmentation["reverse_complement"].to(
                    device,
                    non_blocking=True,
                )

            _, metric_predictions = forward_heads(
                model,
                heads_model,
                sequences,
                organism_idx,
                crop_bins,
                use_amp=not args.no_amp,
                return_scaled=False,
                requires_grad=False,
                reverse_complement=reverse_complement,
                strand_pair_by_modality=strand_pair_by_modality,
                head_organism_idx=head_organism_idx,
            )
            pred_np = torch.cat(
                [metric_predictions[modality] for modality in modalities],
                dim=-1,
            ).detach().cpu().float().numpy()
            target_np = torch.cat(
                [
                    modality_targets[modality][target_resolution]
                    for modality in modalities
                ],
                dim=-1,
            ).float().numpy()

            for batch_i, overlaps in enumerate(batch_overlaps):
                if not overlaps:
                    continue
                seq = seqs_df.iloc[seq_i + batch_i]
                for segment in overlaps:
                    gene_seq_start = max(0, segment.start - int(seq.Start))
                    gene_seq_end = min(int(seq.End) - int(seq.Start), segment.end - int(seq.Start))
                    bin_start = int(np.round(gene_seq_start / target_resolution))
                    bin_end = int(np.round(gene_seq_end / target_resolution))
                    bin_start = max(0, min(bin_start, pred_np.shape[1]))
                    bin_end = max(0, min(bin_end, pred_np.shape[1]))
                    if bin_end <= bin_start:
                        continue
                    gene_id = segment.gene_id
                    gene_strands[gene_id] = segment.strand
                    gene_preds[gene_id].append(pred_np[batch_i, bin_start:bin_end])
                    gene_targets[gene_id].append(target_np[batch_i, bin_start:bin_end])

        seq_i += batch_size
        steps += 1
        pbar.update(1)
    pbar.close()
    return gene_preds, gene_targets, gene_strands


def write_profile(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array.astype("float16"))


def compute_gene_tables(
    gene_preds_dict: dict[str, list[np.ndarray]],
    gene_targets_dict: dict[str, list[np.ndarray]],
    gene_strands: dict[str, str],
    gene_lengths: dict[str, int],
    targets_df: pd.DataFrame,
    output_targets_df: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gene_ids = sorted(set(gene_targets_dict).intersection(gene_preds_dict))
    n_targets = output_targets_df.shape[0]
    gene_preds = []
    gene_targets = []
    gene_within = []
    gene_wvar = []

    for gene_id in gene_ids:
        preds = np.concatenate(gene_preds_dict[gene_id], axis=0).astype("float32")
        targets = np.concatenate(gene_targets_dict[gene_id], axis=0).astype("float32")
        mask = strand_mask(targets_df, gene_strands.get(gene_id, "+"), flip_strand=args.flip_strand)
        preds = preds[:, mask]
        targets = targets[:, mask]
        if preds.shape[1] != n_targets:
            raise ValueError(
                f"{gene_id}: stranded target count {preds.shape[1]} does not match "
                f"output target count {n_targets}"
            )

        within = np.full(n_targets, np.nan, dtype=np.float32)
        for target_i in range(n_targets):
            if preds[:, target_i].var() > 1e-6 and targets[:, target_i].var() > 1e-6:
                within[target_i] = finite_pearson(
                    np.log2(targets[:, target_i] + 1.0),
                    np.log2(preds[:, target_i] + 1.0),
                )
        gene_within.append(within)
        gene_wvar.append(targets.var(axis=0))

        if args.store_span:
            hash_code = str(gene_id.split(".")[0][-1])
            write_profile(
                args.out_dir / "gene_within" / hash_code / "preds" / f"{gene_id}_preds.npy",
                preds,
            )
            write_profile(
                args.out_dir / "gene_within" / hash_code / "targets" / f"{gene_id}_targets.npy",
                targets,
            )

        preds_gene = preds.mean(axis=0) / float(args.target_resolution)
        targets_gene = targets.mean(axis=0) / float(args.target_resolution)
        if not args.drop_length_norm:
            preds_gene = preds_gene * gene_lengths[gene_id]
            targets_gene = targets_gene * gene_lengths[gene_id]
        gene_preds.append(preds_gene)
        gene_targets.append(targets_gene)

    gene_preds_np = np.asarray(gene_preds, dtype=np.float32)
    gene_targets_np = np.asarray(gene_targets, dtype=np.float32)
    gene_within_np = np.asarray(gene_within, dtype=np.float32)
    gene_wvar_np = np.asarray(gene_wvar, dtype=np.float32)

    if args.pseudo_qtl is not None:
        for target_i in range(n_targets):
            nonzero = np.nonzero(gene_targets_np[:, target_i] != 0.0)[0]
            if nonzero.size == 0:
                continue
            gene_targets_np[:, target_i] += np.quantile(
                gene_targets_np[:, target_i][nonzero],
                q=args.pseudo_qtl,
            )
            gene_preds_np[:, target_i] += np.quantile(
                gene_preds_np[:, target_i][nonzero],
                q=args.pseudo_qtl,
            )

    gene_targets_log = np.log2(gene_targets_np + 1.0)
    gene_preds_log = np.log2(gene_preds_np + 1.0)
    columns = output_targets_df["identifier"].astype(str).tolist()

    gene_targets_df = pd.DataFrame(gene_targets_log, index=gene_ids, columns=columns)
    gene_preds_df = pd.DataFrame(gene_preds_log, index=gene_ids, columns=columns)
    gene_within_df = pd.DataFrame(gene_within_np, index=gene_ids, columns=columns)
    gene_var_df = pd.DataFrame(gene_wvar_np, index=gene_ids, columns=columns)
    acc_df = compute_accuracy(
        gene_targets_log,
        gene_preds_log,
        gene_within_np,
        gene_wvar_np,
        output_targets_df,
    )
    return gene_targets_df, gene_preds_df, gene_within_df, gene_var_df, acc_df


def compute_accuracy(
    gene_targets: np.ndarray,
    gene_preds: np.ndarray,
    gene_within: np.ndarray,
    gene_wvar: np.ndarray,
    output_targets_df: pd.DataFrame,
) -> pd.DataFrame:
    gene_targets_norm = quantile_normalize_matrix(gene_targets)
    gene_targets_norm = gene_targets_norm - gene_targets_norm.mean(axis=-1, keepdims=True)
    gene_preds_norm = quantile_normalize_matrix(gene_preds)
    gene_preds_norm = gene_preds_norm - gene_preds_norm.mean(axis=-1, keepdims=True)

    wvar_t = np.percentile(gene_wvar, 80, axis=0)
    rows = []
    for target_i, row in output_targets_df.reset_index(drop=True).iterrows():
        var_mask = gene_wvar[:, target_i] > wvar_t[target_i]
        rows.append({
            "identifier": row.get("identifier", ""),
            "pearsonr": finite_pearson(gene_targets[:, target_i], gene_preds[:, target_i]),
            "r2": explained_variance(gene_targets[:, target_i], gene_preds[:, target_i]),
            "pearsonr_norm": finite_pearson(
                gene_targets_norm[:, target_i],
                gene_preds_norm[:, target_i],
            ),
            "r2_norm": explained_variance(
                gene_targets_norm[:, target_i],
                gene_preds_norm[:, target_i],
            ),
            "pearsonr_gene": nanmean(gene_within[:, target_i][var_mask]),
            "description": row.get("description", ""),
            "modality": row.get("modality", ""),
        })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True

    checkpoint_path = resolve_checkpoint(args.checkpoint, args.checkpoint_name)
    checkpoint = load_checkpoint(checkpoint_path)
    metadata = load_metadata(checkpoint, checkpoint_path)

    data_dir = resolve_path(args.data_dir, metadata.get("data_dir"), checkpoint_path)
    if data_dir is None:
        raise SystemExit("--data-dir is required when checkpoint metadata omits data_dir")
    pretrained_weights = resolve_path(
        args.pretrained_weights,
        metadata.get("pretrained_weights"),
        checkpoint_path,
    )
    if pretrained_weights is None:
        raise SystemExit(
            "--pretrained-weights is required when checkpoint metadata omits pretrained_weights"
        )

    if args.pooling is None:
        args.pooling = metadata.get("pooling", "mean")
    if args.target_resolution is None:
        args.target_resolution = int(metadata.get("target_resolution", 128))
    if args.batch_size is None:
        args.batch_size = int(metadata.get("batch_size", 1))
    if args.dtype is None:
        args.dtype = metadata.get("dtype", "bfloat16")
    if args.organism_idx is None:
        args.organism_idx = int(metadata.get("organism_idx", 0))
    if not args.no_amp and metadata.get("use_amp") is False:
        args.no_amp = True
    if args.num_workers != 0:
        raise SystemExit(
            "Gene-level evaluation requires --num-workers 0 so TFRecord examples "
            "stay aligned with sequences.bed"
        )
    if args.tfr_num_parallel_reads != 1:
        raise SystemExit(
            "Gene-level evaluation requires --tfr-num-parallel-reads 1 so TFRecord "
            "examples stay aligned with sequences.bed"
        )

    rank, world_size, local_rank, device = setup_torchrun(args.device)
    if world_size != 1:
        cleanup_torchrun()
        raise SystemExit("test_tfr_genes.py currently supports single-process evaluation only")

    try:
        modalities = resolve_modalities(args, metadata, data_dir)
        dataset = create_tfr_dataset(
            data_dir,
            args.split,
            modalities,
            args.pooling,
            args.target_resolution,
            repeat=False,
            shuffle_files=False,
            num_parallel_reads=args.tfr_num_parallel_reads,
            rank=rank,
            world_size=world_size,
        )
        loader = create_loader(
            dataset,
            args.batch_size,
            args.num_workers,
            prefetch_n=args.prefetch_n,
            persistent_workers=False,
        )
        crop_bins = (
            args.crop_bins
            if args.crop_bins is not None
            else int(metadata.get("crop_bins", dataset.prediction_crop_bins))
        )

        args.out_dir.mkdir(parents=True, exist_ok=True)
        segments = load_gene_segments(args.genes_gtf, span=args.span)
        write_genes_bed(args.out_dir / "genes.bed", segments)
        gene_lengths = defaultdict(int)
        for segment in segments:
            gene_lengths[segment.gene_id] += segment.end - segment.start

        seqs_df = read_sequences(data_dir, args.split)
        seq_gene_overlaps = overlap_segments_by_sequence(seqs_df, segments)

        if is_main_process(rank):
            print(
                " ".join(
                    (
                        f"Checkpoint: {checkpoint_path}",
                        f"split={args.split}",
                        f"device={device}",
                        f"local_rank={local_rank}",
                    )
                )
            )
            print(
                " ".join(
                    (
                        f"Data: {data_dir}",
                        f"modalities={','.join(modalities)}",
                        f"genes={len(gene_lengths)}",
                        f"segments={len(segments)}",
                        f"target_resolution={args.target_resolution}",
                    )
                )
            )

        dtype_policy = (
            DtypePolicy.full_float32()
            if args.dtype == "float32"
            else DtypePolicy.mixed_precision()
        )
        model = AlphaGenome(dtype_policy=dtype_policy)
        model = load_trunk(model, str(pretrained_weights), exclude_heads=True)
        model = remove_all_heads(model).to(device)
        for param in model.parameters():
            param.requires_grad = False

        heads_model = create_heads_model(
            dataset,
            modalities,
            metadata,
            checkpoint,
            device,
            args.target_resolution,
        )
        targets_df = target_metadata(dataset, modalities)
        output_targets_df = strand_output_rows(targets_df, flip_strand=args.flip_strand)

        gene_preds_dict, gene_targets_dict, gene_strands = aggregate_gene_predictions(
            model,
            heads_model,
            loader,
            seqs_df,
            seq_gene_overlaps,
            targets_df,
            device,
            args,
            crop_bins,
            args.target_resolution,
        )

        (
            gene_targets_df,
            gene_preds_df,
            gene_within_df,
            gene_var_df,
            acc_df,
        ) = compute_gene_tables(
            gene_preds_dict,
            gene_targets_dict,
            gene_strands,
            dict(gene_lengths),
            targets_df,
            output_targets_df,
            args,
        )

        gene_targets_df.to_csv(args.out_dir / "gene_targets.tsv", sep="\t")
        gene_preds_df.to_csv(args.out_dir / "gene_preds.tsv", sep="\t")
        gene_within_df.to_csv(args.out_dir / "gene_within.tsv", sep="\t")
        gene_var_df.to_csv(args.out_dir / "gene_var.tsv", sep="\t")
        acc_df.to_csv(args.out_dir / "acc.txt", sep="\t", index=False)

        print(f"{gene_targets_df.shape[0]} genes")
        print("Overall PearsonR:     %.4f" % nanmean(acc_df.pearsonr.to_numpy()))
        print("Overall R2:           %.4f" % nanmean(acc_df.r2.to_numpy()))
        print("Normalized PearsonR:  %.4f" % nanmean(acc_df.pearsonr_norm.to_numpy()))
        print("Normalized R2:        %.4f" % nanmean(acc_df.r2_norm.to_numpy()))
        print("Within-gene PearsonR: %.4f" % nanmean(acc_df.pearsonr_gene.to_numpy()))
        print(f"Wrote gene-level outputs to {args.out_dir}")
    finally:
        cleanup_torchrun()


if __name__ == "__main__":
    main()
