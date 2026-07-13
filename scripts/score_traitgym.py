#!/usr/bin/env python
"""Score TraitGym variants with AlphaGenome track predictions.

By default this writes a Baskerville eval_utils-style parquet file: one row per
variant and one column per track for ``--pred-type L2``. The optional HDF5 output
matches the shape consumed by EpiBRAIN's TraitGym analysis:

    variants/{chr,pos,ref,alt}
    results/log_square

The L2 score is the norm across predicted bins of
``log2(1 + alt) - log2(1 + ref)``. ``--pred-type L2L2`` then collapses all track
scores for each variant with a second L2 norm, matching Baskerville's
``VEPSeqnn.pred`` behavior.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import pysam
import torch
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
for path in (SCRIPT_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from alphagenome_pytorch import AlphaGenome  # noqa: E402
from alphagenome_pytorch.config import DtypePolicy  # noqa: E402
from alphagenome_pytorch.extensions.finetuning.heads import create_finetuning_head  # noqa: E402
from alphagenome_pytorch.extensions.finetuning.transfer import load_trunk, remove_all_heads  # noqa: E402
from alphagenome_pytorch.named_outputs import TrackMetadataCatalog  # noqa: E402
from finetune_tfr_heads import TFRHeads, forward_heads  # noqa: E402


DEFAULT_ALL_FOLDS_WEIGHTS = (
    SCRIPT_DIR.parent / "weights" / "model_all_folds.safetensors"
)
DEFAULT_TRAITGYM_DIR = (
    SCRIPT_DIR.parents[2] / "baskerville_dnn" / "data" / "ref" / "traitgym"
)
DEFAULT_TRAITGYM_PARQUET = DEFAULT_TRAITGYM_DIR / "test.parquet"
DEFAULT_TRAITGYM_FASTA = (
    DEFAULT_TRAITGYM_DIR / "Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa"
)
BUILTIN_TRACK_HEADS = (
    "atac",
    "dnase",
    "procap",
    "cage",
    "rna_seq",
    "chip_tf",
    "chip_histone",
)


@dataclass(frozen=True)
class Variant:
    row_index: int
    source_index: int
    chrom: str
    pos: int
    ref: str
    alt: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score TraitGym variants with AlphaGenome in Baskerville eval_utils style.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-kind",
        choices=["all-folds", "precomp32"],
        required=True,
        help="Use built-in AlphaGenome all-fold heads or local mouse-brain heads.",
    )
    parser.add_argument("--variants", type=Path, default=DEFAULT_TRAITGYM_PARQUET)
    parser.add_argument("--fasta", type=Path, default=DEFAULT_TRAITGYM_FASTA)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--output-format",
        choices=["parquet", "h5"],
        default="parquet",
        help="Baskerville-style parquet or EpiBRAIN-compatible HDF5.",
    )
    parser.add_argument(
        "--pred-type",
        choices=["L2", "L2L2"],
        default="L2",
        help="Baskerville TraitGym prediction type.",
    )
    parser.add_argument(
        "--track-anno-out",
        type=Path,
        default=None,
        help="Optional CSV containing output track labels and grouping columns.",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["mixed", "float32"], default="mixed")
    parser.add_argument(
        "--organism-idx",
        type=int,
        default=None,
        help=(
            "AlphaGenome organism index. Defaults to 0 for all-folds and 1 for "
            "precomp32."
        ),
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=None,
        help=(
            "Input sequence length. Defaults to 1,048,576 for all-folds and to "
            "(target_length + 2 * crop_bins) * target_resolution for precomp32."
        ),
    )
    parser.add_argument(
        "--rc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Average forward and reverse-complement per-track scores.",
    )
    parser.add_argument(
        "--h5-chr-prefix",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store HDF5 variants/chr values with chr prefix for EpiBRAIN merges.",
    )
    parser.add_argument(
        "--head",
        action="append",
        default=None,
        help=(
            "Built-in AlphaGenome head to score. Repeat for a subset. "
            f"Defaults to {','.join(BUILTIN_TRACK_HEADS)}."
        ),
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_ALL_FOLDS_WEIGHTS,
        help="All-fold AlphaGenome weights for --model-kind all-folds.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="precomp32 best_heads.pt/last_heads.pt, or a run directory.",
    )
    parser.add_argument("--checkpoint-name", default="best_heads.pt")
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument(
        "--pretrained-weights",
        type=Path,
        default=None,
        help="Base trunk weights for --model-kind precomp32. Defaults to metadata.",
    )
    parser.add_argument(
        "--modality",
        action="append",
        default=None,
        help="precomp32 modality to score. Repeat for a subset. Defaults to metadata.",
    )
    return parser.parse_args()


def dtype_policy(name: str) -> DtypePolicy:
    if name == "float32":
        return DtypePolicy.full_float32()
    return DtypePolicy.mixed_precision()


def resolve_checkpoint(path: Path, checkpoint_name: str) -> Path:
    if path.is_dir():
        path = path / checkpoint_name
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def load_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Expected checkpoint dict in {path}")
    return checkpoint


def load_metadata(checkpoint: dict[str, Any], checkpoint_path: Path, metadata_path: Path | None) -> dict[str, Any]:
    if metadata_path is not None:
        with metadata_path.open() as handle:
            return json.load(handle)
    metadata = checkpoint.get("metadata")
    if isinstance(metadata, dict):
        return metadata
    with (checkpoint_path.parent / "metadata.json").open() as handle:
        return json.load(handle)


def resolve_existing_path(path: Path | None, metadata_value: str | None, checkpoint_path: Path, name: str) -> Path:
    if path is None and metadata_value:
        path = Path(metadata_value)
    if path is None:
        raise ValueError(f"{name} is required")
    candidates = [
        path,
        Path.cwd() / path,
        checkpoint_path.parent / path,
        checkpoint_path.parent.parent / path,
        SCRIPT_DIR.parents[1] / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


def load_variants(path: Path, start: int, end: int | None) -> tuple[pd.DataFrame, list[Variant]]:
    df = pd.read_parquet(path)
    required = {"chrom", "pos", "ref", "alt"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    sliced = df.iloc[start:end].reset_index(drop=False).rename(columns={"index": "source_index"})
    variants = [
        Variant(
            row_index=i,
            source_index=int(row.source_index),
            chrom=str(row.chrom),
            pos=int(row.pos),
            ref=str(row.ref).upper(),
            alt=str(row.alt).upper(),
        )
        for i, row in enumerate(sliced.itertuples(index=False))
    ]
    return sliced, variants


def fasta_chrom(fasta: pysam.Fastafile, chrom: str) -> str:
    refs = set(fasta.references)
    if chrom in refs:
        return chrom
    if chrom.startswith("chr") and chrom[3:] in refs:
        return chrom[3:]
    prefixed = f"chr{chrom}"
    if prefixed in refs:
        return prefixed
    raise KeyError(f"Chromosome {chrom!r} not found in FASTA")


def fetch_centered_sequence(fasta: pysam.Fastafile, chrom: str, pos: int, seq_len: int) -> tuple[str, int, str]:
    fetch_chrom = fasta_chrom(fasta, chrom)
    start0 = (pos - 1) - ((seq_len - 1) // 2)
    end0 = start0 + seq_len
    pad_left = max(0, -start0)
    fetch_start = max(0, start0)
    seq = fasta.fetch(fetch_chrom, fetch_start, end0).upper()
    if pad_left:
        seq = ("N" * pad_left) + seq
    if len(seq) < seq_len:
        seq = seq + ("N" * (seq_len - len(seq)))
    return seq[:seq_len], start0, fetch_chrom


def make_alt_sequence(ref_seq: str, context_start0: int, variant: Variant) -> tuple[str | None, str | None]:
    if len(variant.ref) != 1 or len(variant.alt) != 1:
        return None, "non_snv"
    rel = (variant.pos - 1) - context_start0
    if rel < 0 or rel >= len(ref_seq):
        return None, "variant_outside_context"
    fasta_ref = ref_seq[rel].upper()
    if fasta_ref != variant.ref:
        return None, f"ref_mismatch:{variant.ref}!={fasta_ref}"
    return ref_seq[:rel] + variant.alt + ref_seq[rel + 1 :], None


_ONEHOT_LOOKUP = np.full(128, -1, dtype=np.int8)
for _idx, _base in enumerate("ACGT"):
    _ONEHOT_LOOKUP[ord(_base)] = _idx
    _ONEHOT_LOOKUP[ord(_base.lower())] = _idx


def sequence_to_onehot(sequence: str) -> np.ndarray:
    seq = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
    onehot = np.zeros((len(seq), 4), dtype=np.float32)
    indices = _ONEHOT_LOOKUP[seq.clip(0, 127)]
    mask = indices >= 0
    onehot[np.where(mask)[0], indices[mask]] = 1
    return onehot


def reverse_complement_onehot(batch: torch.Tensor) -> torch.Tensor:
    return torch.flip(batch, dims=(1, 2))


def traitgym_l2(ref_pred: torch.Tensor, alt_pred: torch.Tensor) -> np.ndarray:
    lfc = (torch.log1p(alt_pred.float()) - torch.log1p(ref_pred.float())) / np.log(2.0)
    return torch.linalg.vector_norm(lfc, dim=1).detach().cpu().numpy().astype(np.float32)


def chrom_for_h5(chrom: str, use_chr_prefix: bool) -> str:
    if use_chr_prefix:
        return chrom if chrom.startswith("chr") else f"chr{chrom}"
    return chrom[3:] if chrom.startswith("chr") else chrom


def create_precomp_heads_model(
    checkpoint: dict[str, Any],
    metadata: dict[str, Any],
    modalities: list[str],
    device: torch.device,
) -> TFRHeads:
    heads = {}
    for modality in modalities:
        heads[modality] = create_finetuning_head(
            assay_type=metadata["assay_types"][modality],
            n_tracks=int(metadata["n_tracks"][modality]),
            resolutions=(128,),
            num_organisms=1,
        )

    cell_types = {
        modality: list(metadata["cell_types"][modality])
        for modality in modalities
        if modality in metadata.get("cell_types", {})
    }
    modality_types = {
        modality: list(metadata["modality_types"][modality])
        for modality in modalities
        if modality in metadata.get("modality_types", {})
    }
    heads_model = TFRHeads(
        heads,
        cell_types,
        modality_types,
        cell_embedding_dim=int(metadata.get("cell_embedding_dim", 16)),
        target_resolution=int(metadata.get("target_resolution", 128)),
    ).to(device)

    state_dict = (
        checkpoint.get("heads_model_state_dict")
        or checkpoint.get("heads_state_dict")
        or checkpoint.get("state_dict")
    )
    if state_dict is None:
        raise KeyError("Checkpoint lacks heads_model_state_dict, heads_state_dict, or state_dict")
    missing, unexpected = heads_model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: missing head keys: {missing[:5]}...")
    if unexpected:
        print(f"Warning: unexpected head keys: {unexpected[:5]}...")
    heads_model.eval()
    for param in heads_model.parameters():
        param.requires_grad = False
    return heads_model


def make_track_anno(track_blocks: list[tuple[str, int | list[int], list[str] | None]]) -> pd.DataFrame:
    rows = []
    global_i = 0
    for group, track_spec, names in track_blocks:
        local_indices = range(track_spec) if isinstance(track_spec, int) else track_spec
        for out_i, local_i in enumerate(local_indices):
            name = names[out_i] if names and out_i < len(names) else f"{group}_{local_i}"
            rows.append({
                "track_index": global_i,
                "local_index": local_i,
                "modality": group,
                "cell_type": name.rsplit(".", 2)[0] if "." in name else name,
                "identifier": name,
            })
            global_i += 1
    return pd.DataFrame(rows)


def all_folds_track_blocks(
    model: AlphaGenome,
    heads: list[str],
    organism_idx: int,
) -> tuple[list[tuple[str, list[int], list[str]]], dict[str, list[int]], dict[str, int]]:
    catalog = TrackMetadataCatalog.load_builtin(organism_idx)
    track_blocks = []
    keep_indices_by_head = {}
    dropped_by_head = {}
    for head in heads:
        n_tracks = model.heads[head].num_tracks
        tracks = catalog.get_tracks(head, organism=organism_idx, num_tracks=n_tracks, strict=True)
        keep_tracks = [track for track in tracks if not track.is_padding]
        keep_indices = [int(track.track_index) for track in keep_tracks]
        track_blocks.append((head, keep_indices, [track.track_name for track in keep_tracks]))
        keep_indices_by_head[head] = keep_indices
        dropped_by_head[head] = n_tracks - len(keep_indices)
    return track_blocks, keep_indices_by_head, dropped_by_head


def unique_columns(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    columns = []
    for name in names:
        key = str(name)
        count = seen.get(key, 0)
        seen[key] = count + 1
        columns.append(key if count == 0 else f"{key}.{count}")
    return columns


def apply_pred_type(scores: np.ndarray, pred_type: str) -> np.ndarray:
    if pred_type == "L2":
        return scores
    if pred_type == "L2L2":
        return np.linalg.norm(scores, axis=1).reshape(-1, 1).astype(np.float32)
    raise ValueError(f"Unknown pred_type: {pred_type}")


def write_parquet(path: Path, scores: np.ndarray, track_anno: pd.DataFrame, pred_type: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if pred_type == "L2L2":
        columns = ["L2L2"]
    else:
        columns = unique_columns(track_anno["identifier"].astype(str).tolist())
    pd.DataFrame(scores, columns=columns).to_parquet(path, index=False)


def write_h5(
    path: Path,
    variants: list[Variant],
    scores: np.ndarray,
    errors: list[tuple[int, str]],
    use_chr_prefix: bool,
    attrs: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    string_dtype = h5py.string_dtype("utf-8")
    with h5py.File(path, "w") as h5:
        for key, value in attrs.items():
            if value is not None:
                h5.attrs[key] = value
        h5.attrs["score_name"] = "log_square"
        variants_group = h5.create_group("variants")
        variants_group.create_dataset(
            "chr",
            data=np.array([chrom_for_h5(v.chrom, use_chr_prefix) for v in variants], dtype=object),
            dtype=string_dtype,
        )
        variants_group.create_dataset("pos", data=np.array([v.pos for v in variants], dtype=np.int64))
        variants_group.create_dataset("ref", data=np.array([v.ref for v in variants], dtype=object), dtype=string_dtype)
        variants_group.create_dataset("alt", data=np.array([v.alt for v in variants], dtype=object), dtype=string_dtype)
        variants_group.create_dataset("row", data=np.array([v.row_index for v in variants], dtype=np.int64))
        variants_group.create_dataset("source_index", data=np.array([v.source_index for v in variants], dtype=np.int64))
        h5.create_dataset("results/log_square", data=scores, dtype="f4", compression="gzip")
        error_group = h5.create_group("errors")
        error_group.create_dataset("row", data=np.array([row for row, _reason in errors], dtype=np.int64))
        error_group.create_dataset("reason", data=np.array([reason for _row, reason in errors], dtype=object), dtype=string_dtype)


@torch.no_grad()
def score_all_folds(args: argparse.Namespace, variants: list[Variant]) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any], list[tuple[int, str]]]:
    heads = args.head or list(BUILTIN_TRACK_HEADS)
    device = torch.device(args.device)
    model = AlphaGenome.from_pretrained(
        args.weights,
        dtype_policy=dtype_policy(args.dtype),
        device=device,
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    context_length = args.context_length or 1_048_576
    organism_idx = torch.full((args.batch_size,), args.organism_idx, dtype=torch.long, device=device)
    track_blocks, keep_indices_by_head, dropped_by_head = all_folds_track_blocks(
        model,
        heads,
        args.organism_idx,
    )
    keep_tensors_by_head = {
        head: torch.as_tensor(indices, dtype=torch.long, device=device)
        for head, indices in keep_indices_by_head.items()
    }
    track_anno = make_track_anno(track_blocks)
    scores = np.full((len(variants), len(track_anno)), np.nan, dtype=np.float32)
    errors: list[tuple[int, str]] = []

    fasta = pysam.Fastafile(str(args.fasta))
    try:
        for start in tqdm(range(0, len(variants), args.batch_size), desc="Scoring TraitGym"):
            batch = variants[start : start + args.batch_size]
            ref_arrays = []
            alt_arrays = []
            valid_rows = []
            for variant in batch:
                try:
                    ref_seq, context_start0, _fetch_chrom = fetch_centered_sequence(
                        fasta, variant.chrom, variant.pos, context_length
                    )
                    alt_seq, error = make_alt_sequence(ref_seq, context_start0, variant)
                    if error is not None:
                        errors.append((variant.row_index, error))
                        continue
                    ref_arrays.append(sequence_to_onehot(ref_seq))
                    alt_arrays.append(sequence_to_onehot(alt_seq))
                    valid_rows.append(variant.row_index)
                except Exception as exc:
                    errors.append((variant.row_index, str(exc)))
            if not valid_rows:
                continue

            ref = torch.as_tensor(np.stack(ref_arrays), dtype=torch.float32, device=device)
            alt = torch.as_tensor(np.stack(alt_arrays), dtype=torch.float32, device=device)
            org = organism_idx[: ref.shape[0]]
            pred_ref = model.predict(
                ref,
                org,
                resolutions=(128,),
                heads=tuple(heads),
                channels_last=True,
            )
            pred_alt = model.predict(
                alt,
                org,
                resolutions=(128,),
                heads=tuple(heads),
                channels_last=True,
            )

            batch_scores = []
            if args.rc:
                pred_ref_rc = model.predict(
                    reverse_complement_onehot(ref),
                    org,
                    resolutions=(128,),
                    heads=tuple(heads),
                    channels_last=True,
                )
                pred_alt_rc = model.predict(
                    reverse_complement_onehot(alt),
                    org,
                    resolutions=(128,),
                    heads=tuple(heads),
                    channels_last=True,
                )
            for head in heads:
                keep = keep_tensors_by_head[head]
                head_scores = traitgym_l2(
                    pred_ref[head][128].index_select(-1, keep),
                    pred_alt[head][128].index_select(-1, keep),
                )
                if args.rc:
                    head_scores = (
                        head_scores
                        + traitgym_l2(
                            pred_ref_rc[head][128].index_select(-1, keep),
                            pred_alt_rc[head][128].index_select(-1, keep),
                        )
                    ) / 2.0
                batch_scores.append(head_scores)
            scores[np.array(valid_rows, dtype=int), :] = np.concatenate(batch_scores, axis=1)
    finally:
        fasta.close()

    attrs = {
        "model_kind": "all-folds",
        "weights": str(args.weights),
        "context_length": context_length,
        "rc": args.rc,
        "organism_idx": args.organism_idx,
        "padding_filtered": True,
        "padding_dropped_by_head": json.dumps(dropped_by_head, sort_keys=True),
    }
    return scores, track_anno, attrs, errors


@torch.no_grad()
def score_precomp32(args: argparse.Namespace, variants: list[Variant]) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any], list[tuple[int, str]]]:
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required for --model-kind precomp32")
    checkpoint_path = resolve_checkpoint(args.checkpoint, args.checkpoint_name)
    checkpoint = load_checkpoint(checkpoint_path)
    metadata = load_metadata(checkpoint, checkpoint_path, args.metadata)
    metadata_path = args.metadata or checkpoint_path.parent / "metadata.json"
    modalities = [str(m) for m in (args.modality or metadata["modalities"])]
    pretrained_weights = resolve_existing_path(
        args.pretrained_weights,
        metadata.get("pretrained_weights"),
        checkpoint_path,
        "--pretrained-weights",
    )

    device = torch.device(args.device)
    trunk = AlphaGenome(dtype_policy=dtype_policy(args.dtype))
    trunk = load_trunk(trunk, str(pretrained_weights), exclude_heads=True)
    trunk = remove_all_heads(trunk).to(device)
    trunk.eval()
    for param in trunk.parameters():
        param.requires_grad = False
    heads_model = create_precomp_heads_model(checkpoint, metadata, modalities, device)

    target_resolution = int(metadata["target_resolution"])
    crop_bins = int(metadata.get("crop_bins", 0))
    target_length = int(metadata["target_length"])
    context_length = args.context_length or ((target_length + 2 * crop_bins) * target_resolution)
    organism_idx = torch.full((args.batch_size,), args.organism_idx, dtype=torch.long, device=device)
    track_blocks = [
        (modality, int(metadata["n_tracks"][modality]), metadata.get("track_names", {}).get(modality))
        for modality in modalities
    ]
    track_anno = make_track_anno(track_blocks)
    scores = np.full((len(variants), len(track_anno)), np.nan, dtype=np.float32)
    errors: list[tuple[int, str]] = []

    fasta = pysam.Fastafile(str(args.fasta))
    try:
        for start in tqdm(range(0, len(variants), args.batch_size), desc="Scoring TraitGym"):
            batch = variants[start : start + args.batch_size]
            ref_arrays = []
            alt_arrays = []
            valid_rows = []
            for variant in batch:
                try:
                    ref_seq, context_start0, _fetch_chrom = fetch_centered_sequence(
                        fasta, variant.chrom, variant.pos, context_length
                    )
                    alt_seq, error = make_alt_sequence(ref_seq, context_start0, variant)
                    if error is not None:
                        errors.append((variant.row_index, error))
                        continue
                    ref_arrays.append(sequence_to_onehot(ref_seq))
                    alt_arrays.append(sequence_to_onehot(alt_seq))
                    valid_rows.append(variant.row_index)
                except Exception as exc:
                    errors.append((variant.row_index, str(exc)))
            if not valid_rows:
                continue

            ref = torch.as_tensor(np.stack(ref_arrays), dtype=torch.float32, device=device)
            alt = torch.as_tensor(np.stack(alt_arrays), dtype=torch.float32, device=device)
            org = organism_idx[: ref.shape[0]]
            head_org = torch.zeros(ref.shape[0], dtype=torch.long, device=device)
            _loss_ref, pred_ref = forward_heads(
                trunk,
                heads_model,
                ref,
                org,
                crop_bins,
                use_amp=args.dtype == "mixed",
                return_scaled=False,
                requires_grad=False,
                head_organism_idx=head_org,
            )
            _loss_alt, pred_alt = forward_heads(
                trunk,
                heads_model,
                alt,
                org,
                crop_bins,
                use_amp=args.dtype == "mixed",
                return_scaled=False,
                requires_grad=False,
                head_organism_idx=head_org,
            )

            batch_scores = []
            if args.rc:
                _loss_ref_rc, pred_ref_rc = forward_heads(
                    trunk,
                    heads_model,
                    reverse_complement_onehot(ref),
                    org,
                    crop_bins,
                    use_amp=args.dtype == "mixed",
                    return_scaled=False,
                    requires_grad=False,
                    head_organism_idx=head_org,
                )
                _loss_alt_rc, pred_alt_rc = forward_heads(
                    trunk,
                    heads_model,
                    reverse_complement_onehot(alt),
                    org,
                    crop_bins,
                    use_amp=args.dtype == "mixed",
                    return_scaled=False,
                    requires_grad=False,
                    head_organism_idx=head_org,
                )
            for modality in modalities:
                modality_scores = traitgym_l2(pred_ref[modality], pred_alt[modality])
                if args.rc:
                    modality_scores = (
                        modality_scores
                        + traitgym_l2(pred_ref_rc[modality], pred_alt_rc[modality])
                    ) / 2.0
                batch_scores.append(modality_scores)
            scores[np.array(valid_rows, dtype=int), :] = np.concatenate(batch_scores, axis=1)
    finally:
        fasta.close()

    attrs = {
        "model_kind": "precomp32",
        "checkpoint": str(checkpoint_path),
        "metadata": str(metadata_path),
        "pretrained_weights": str(pretrained_weights),
        "target_resolution": target_resolution,
        "target_length": target_length,
        "crop_bins": crop_bins,
        "context_length": context_length,
        "rc": args.rc,
        "organism_idx": args.organism_idx,
    }
    return scores, track_anno, attrs, errors


def main() -> None:
    args = parse_args()
    _df, variants = load_variants(args.variants, args.start, args.end)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.organism_idx is None:
        args.organism_idx = 1 if args.model_kind == "precomp32" else 0

    if args.model_kind == "all-folds":
        scores, track_anno, attrs, errors = score_all_folds(args, variants)
    else:
        scores, track_anno, attrs, errors = score_precomp32(args, variants)

    scores = apply_pred_type(scores, args.pred_type)
    attrs.update({
        "variants": str(args.variants),
        "fasta": str(args.fasta),
        "start": args.start,
        "end": -1 if args.end is None else args.end,
        "pred_type": args.pred_type,
    })
    if args.output_format == "parquet":
        write_parquet(args.out, scores, track_anno, args.pred_type)
    else:
        write_h5(args.out, variants, scores, errors, args.h5_chr_prefix, attrs)
    if args.track_anno_out is not None:
        args.track_anno_out.parent.mkdir(parents=True, exist_ok=True)
        track_anno.to_csv(args.track_anno_out, index=False)

    finite_rows = np.isfinite(scores).any(axis=1)
    print(f"Wrote {args.out}")
    print(f"Variants: {len(variants)}; scored: {int(finite_rows.sum())}; errors: {len(errors)}")
    if args.track_anno_out is not None:
        print(f"Wrote track annotation: {args.track_anno_out}")


if __name__ == "__main__":
    main()
