#!/usr/bin/env python
"""Score eQTL VCFs with fine-tuned AlphaGenome heads.

This writes the HDF5 artifact shape expected by EpiBRAIN's
Analysis/07_eQTL ``alphagenome_custom_heads`` loader:

    variants/{chrom,pos,ref,alt,id,gene}
    scores/{modality}/gene_abs_lfc

The score is the per-track gene-level log fold change between alt and ref
predictions, averaged over bins overlapping exons for the VCF INFO gene.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pysam
import torch
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
SRC_DIR = SCRIPT_DIR.parent / "src"
for path in (SCRIPT_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from alphagenome_pytorch import AlphaGenome  # noqa: E402
from alphagenome_pytorch.config import DtypePolicy  # noqa: E402
from alphagenome_pytorch.extensions.finetuning.heads import create_finetuning_head  # noqa: E402
from alphagenome_pytorch.extensions.finetuning.transfer import load_trunk, remove_all_heads  # noqa: E402
from finetune_tfr_heads import TFRHeads, forward_heads  # noqa: E402


@dataclass(frozen=True)
class VcfVariant:
    chrom: str
    pos: int
    variant_id: str
    ref: str
    alt: str
    info: str
    gene: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an EpiBRAIN 07_eQTL-compatible HDF5 artifact.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="best_heads.pt/last_heads.pt, or a run directory containing best_heads.pt.",
    )
    parser.add_argument(
        "--checkpoint-name",
        default="best_heads.pt",
        help="Checkpoint filename to use when checkpoint is a directory.",
    )
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument(
        "--pretrained-weights",
        type=Path,
        default=None,
        help="Base AlphaGenome trunk weights. Defaults to metadata['pretrained_weights'].",
    )
    parser.add_argument("--vcf", type=Path, required=True)
    parser.add_argument("--gtf", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--modality",
        action="append",
        default=None,
        help="Modality to score. Repeat for a subset. Defaults to metadata modalities.",
    )
    parser.add_argument(
        "--gene-info-keys",
        default="gene_ID,gene_id,gene_name,MT",
        help="Comma-separated VCF INFO keys used to map variants to genes.",
    )
    parser.add_argument("--organism-idx", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["mixed", "float32"], default="mixed")
    parser.add_argument("--start", type=int, default=0, help="First VCF row to score.")
    parser.add_argument("--end", type=int, default=None, help="Stop before this VCF row.")
    parser.add_argument(
        "--score-dataset",
        default="gene_abs_lfc",
        help="Dataset name under scores/{modality}/.",
    )
    parser.add_argument("--signed", action="store_true", help="Write signed LFC instead of abs(LFC).")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help=(
            "Optional EpiBRAIN data_paths JSON for this artifact. Requires "
            "--eval-vcf and --eval-info for 07_eQTL evaluation."
        ),
    )
    parser.add_argument("--eval-vcf", type=Path, default=None, help="VCF path to place in --json-out.")
    parser.add_argument("--eval-info", type=Path, default=None, help="Info CSV path to place in --json-out.")
    parser.add_argument("--exclude", type=Path, default=None, help="Exclude file path to place in --json-out.")
    parser.add_argument("--model-name", default="ag_precomp32_heads")
    parser.add_argument("--plot-name", default="AlphaGenome precomp32 heads")
    parser.add_argument("--output-dir", type=Path, default=None, help="EpiBRAIN eval output_dir for --json-out.")
    return parser.parse_args()


def open_text(path: str | Path):
    path = str(path)
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path)


def normalize_gene_id(gene: str | None) -> str | None:
    if not gene:
        return None
    return gene.strip().split(".")[0]


def parse_info(info: str) -> dict[str, str]:
    fields = {}
    for item in info.split(";"):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        fields[key] = value
    return fields


def read_vcf_variants(
    vcf_path: str | Path,
    gene_info_keys: Iterable[str],
) -> list[VcfVariant]:
    variants = []
    gene_info_keys = tuple(gene_info_keys)
    with open_text(vcf_path) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 8:
                continue
            chrom, pos, variant_id, ref, alts, _qual, _filt, info = parts[:8]
            if "," in alts:
                continue
            fields = parse_info(info)
            gene = None
            for key in gene_info_keys:
                if key in fields:
                    gene = normalize_gene_id(fields[key])
                    break
            variants.append(
                VcfVariant(
                    chrom=chrom,
                    pos=int(pos),
                    variant_id=variant_id,
                    ref=ref.upper(),
                    alt=alts.upper(),
                    info=info,
                    gene=gene,
                )
            )
    return variants


def parse_gtf_exons(gtf_path: str | Path, gene_keys: Iterable[str]) -> dict[str, list[tuple[str, int, int]]]:
    keep = {normalize_gene_id(gene) for gene in gene_keys if gene}
    keep.discard(None)
    gid_re = re.compile(r'gene_id "([^"]+)"')
    gname_re = re.compile(r'gene_name "([^"]+)"')
    gene_exons: dict[str, set[tuple[str, int, int]]] = {}

    with open_text(gtf_path) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] != "exon":
                continue
            chrom = parts[0]
            start = int(parts[3])
            end = int(parts[4])
            attrs = parts[8]
            keys = []
            match = gid_re.search(attrs)
            if match:
                keys.append(normalize_gene_id(match.group(1)))
            match = gname_re.search(attrs)
            if match:
                keys.append(normalize_gene_id(match.group(1)))
            for key in keys:
                if key in keep:
                    gene_exons.setdefault(key, set()).add((chrom, start, end))

    return {key: sorted(value) for key, value in gene_exons.items()}


_ONEHOT_LOOKUP = np.full(128, -1, dtype=np.int8)
for _idx, _base in enumerate("ACGT"):
    _ONEHOT_LOOKUP[ord(_base)] = _idx
    _ONEHOT_LOOKUP[ord(_base.lower())] = _idx


def sequence_to_onehot(sequence: str, dtype=np.float32) -> np.ndarray:
    seq = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
    onehot = np.zeros((len(seq), 4), dtype=dtype)
    indices = _ONEHOT_LOOKUP[seq.clip(0, 127)]
    mask = indices >= 0
    onehot[np.where(mask)[0], indices[mask]] = 1
    return onehot


def fetch_centered_sequence(fasta, chrom: str, pos: int, seq_len: int) -> tuple[str, int, str]:
    start0 = (pos - 1) - ((seq_len - 1) // 2)
    end0 = start0 + seq_len
    fetch_chrom = chrom
    references = set(getattr(fasta, "references", []))
    if references and fetch_chrom not in references:
        alt_chrom = chrom[3:] if chrom.startswith("chr") else f"chr{chrom}"
        if alt_chrom in references:
            fetch_chrom = alt_chrom

    pad_left = max(0, -start0)
    fetch_start = max(0, start0)
    seq = fasta.fetch(fetch_chrom, fetch_start, end0).upper()
    if pad_left:
        seq = ("N" * pad_left) + seq
    if len(seq) < seq_len:
        seq = seq + ("N" * (seq_len - len(seq)))
    return seq[:seq_len], start0, fetch_chrom


def make_snv_alt_sequence(ref_seq: str, context_start0: int, variant: VcfVariant) -> tuple[str | None, str | None]:
    if len(variant.ref) != 1 or len(variant.alt) != 1:
        return None, "non_snv"
    rel = (variant.pos - 1) - context_start0
    if rel < 0 or rel >= len(ref_seq):
        return None, "variant_outside_context"
    fasta_ref = ref_seq[rel].upper()
    if fasta_ref != variant.ref:
        return None, f"ref_mismatch:{variant.ref}!={fasta_ref}"
    return ref_seq[:rel] + variant.alt + ref_seq[rel + 1 :], None


def exon_bin_mask(
    chrom: str,
    pos: int,
    gene: str | None,
    gene_exons: dict[str, list[tuple[str, int, int]]],
    context_length: int,
    n_bins: int,
    bin_size: int,
) -> np.ndarray:
    mask = np.zeros(n_bins, dtype=bool)
    if gene is None or gene not in gene_exons:
        return mask

    covered = n_bins * bin_size
    crop_bp = (context_length - covered) // 2
    context_start = (pos - 1) - ((context_length - 1) // 2)
    bins_start = context_start + crop_bp
    bins_end = bins_start + covered

    for exon_chrom, exon_start, exon_end in gene_exons[gene]:
        if exon_chrom != chrom:
            continue
        exon_start0 = exon_start - 1
        exon_end0 = exon_end
        if exon_end0 <= bins_start or exon_start0 >= bins_end:
            continue
        lo = max(0, (exon_start0 - bins_start) // bin_size)
        hi = min(n_bins, (exon_end0 - bins_start + bin_size - 1) // bin_size)
        if hi > lo:
            mask[lo:hi] = True
    return mask


def gene_lfc_from_prediction_stack(
    ref_preds: np.ndarray,
    alt_preds: np.ndarray,
    mask: np.ndarray,
    eps: float = 0.001,
) -> np.ndarray:
    if ref_preds.ndim == 2:
        ref_slice = ref_preds[mask, :]
        alt_slice = alt_preds[mask, :]
    elif ref_preds.ndim == 3:
        ref_slice = ref_preds[:, mask, :]
        alt_slice = alt_preds[:, mask, :]
    else:
        raise ValueError(f"Expected predictions with 2 or 3 dims, got {ref_preds.shape}")
    mean_ref = ref_slice.mean(axis=tuple(range(ref_slice.ndim - 1)))
    mean_alt = alt_slice.mean(axis=tuple(range(alt_slice.ndim - 1)))
    return (np.log(mean_alt + eps) - np.log(mean_ref + eps)).astype(np.float32)


def resolve_checkpoint(path: Path, checkpoint_name: str) -> Path:
    if path.is_dir():
        path = path / checkpoint_name
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def load_checkpoint(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Expected checkpoint dict in {path}")
    return checkpoint


def load_metadata(checkpoint: dict, checkpoint_path: Path, metadata_path: Path | None) -> dict:
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
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


def resolve_modalities(args: argparse.Namespace, metadata: dict) -> list[str]:
    if args.modality:
        return [str(modality) for modality in args.modality]
    return [str(modality) for modality in metadata["modalities"]]


def create_heads_model(
    checkpoint: dict,
    metadata: dict,
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


def write_variant_group(h5: h5py.File, variants: list[VcfVariant], chrom_key: str) -> None:
    group = h5.create_group("variants")
    string_dtype = h5py.string_dtype("utf-8")
    group.create_dataset(chrom_key, data=np.array([v.chrom for v in variants], dtype=object), dtype=string_dtype)
    group.create_dataset("pos", data=np.array([v.pos for v in variants], dtype=np.int64))
    group.create_dataset("ref", data=np.array([v.ref for v in variants], dtype=object), dtype=string_dtype)
    group.create_dataset("alt", data=np.array([v.alt for v in variants], dtype=object), dtype=string_dtype)
    group.create_dataset("id", data=np.array([v.variant_id for v in variants], dtype=object), dtype=string_dtype)
    group.create_dataset("gene", data=np.array([v.gene or "" for v in variants], dtype=object), dtype=string_dtype)


def write_errors(h5: h5py.File, errors: list[tuple[int, str]]) -> None:
    group = h5.create_group("errors")
    string_dtype = h5py.string_dtype("utf-8")
    group.create_dataset("index", data=np.array([i for i, _reason in errors], dtype=np.int64))
    group.create_dataset("reason", data=np.array([reason for _i, reason in errors], dtype=object), dtype=string_dtype)


def write_epibrain_json(args: argparse.Namespace, metadata_path: Path, modalities: list[str]) -> None:
    if args.json_out is None:
        return
    if args.eval_vcf is None or args.eval_info is None:
        raise ValueError("--json-out requires --eval-vcf and --eval-info")

    cfg = {
        "output_dir": str((args.output_dir or args.out.parent / "eqtl_eval").resolve()),
        "vcf": str(args.eval_vcf.resolve()),
        "info": str(args.eval_info.resolve()),
        "models": {
            args.model_name: {
                "type": "alphagenome_custom_heads",
                "h5": str(args.out.resolve()),
                "score_name": f"scores/{{head}}/{args.score_dataset}",
                "track_anno": str(metadata_path.resolve()),
                "heads": modalities,
            }
        },
        "plot_name": {
            args.model_name: args.plot_name,
        },
    }
    if args.exclude is not None:
        cfg["exclude"] = str(args.exclude.resolve())

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    with args.json_out.open("w") as handle:
        json.dump(cfg, handle, indent=2)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    checkpoint_path = resolve_checkpoint(args.checkpoint, args.checkpoint_name)
    checkpoint = load_checkpoint(checkpoint_path)
    metadata = load_metadata(checkpoint, checkpoint_path, args.metadata)
    metadata_path = args.metadata or checkpoint_path.parent / "metadata.json"
    modalities = resolve_modalities(args, metadata)

    pretrained_weights = resolve_existing_path(
        args.pretrained_weights,
        metadata.get("pretrained_weights"),
        checkpoint_path,
        "--pretrained-weights",
    )

    device = torch.device(args.device)
    dtype_policy = (
        DtypePolicy.full_float32()
        if args.dtype == "float32"
        else DtypePolicy.mixed_precision()
    )
    model = AlphaGenome(dtype_policy=dtype_policy)
    model = load_trunk(model, str(pretrained_weights), exclude_heads=True)
    model = remove_all_heads(model).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    heads_model = create_heads_model(checkpoint, metadata, modalities, device)

    target_resolution = int(metadata["target_resolution"])
    crop_bins = int(metadata.get("crop_bins", 0))
    target_length = int(metadata["target_length"])
    context_length = (target_length + 2 * crop_bins) * target_resolution

    gene_info_keys = [key for key in args.gene_info_keys.split(",") if key]
    all_variants = read_vcf_variants(args.vcf, gene_info_keys=gene_info_keys)
    variants = all_variants[args.start : args.end]
    genes = sorted({variant.gene for variant in variants if variant.gene})
    gene_exons = parse_gtf_exons(args.gtf, genes)

    scores = {
        modality: np.full(
            (len(variants), int(metadata["n_tracks"][modality])),
            np.nan,
            dtype=np.float32,
        )
        for modality in modalities
    }
    errors: list[tuple[int, str]] = []
    organism_idx = torch.tensor([args.organism_idx], dtype=torch.long, device=device)

    fasta = pysam.Fastafile(str(args.fasta))
    try:
        for row_i, variant in enumerate(tqdm(variants, desc="Scoring variants")):
            if variant.gene is None:
                errors.append((row_i, "missing_gene"))
                continue
            if variant.gene not in gene_exons:
                errors.append((row_i, "missing_gene_exons"))
                continue

            ref_seq, context_start0, fetch_chrom = fetch_centered_sequence(
                fasta,
                variant.chrom,
                variant.pos,
                context_length,
            )
            alt_seq, error = make_snv_alt_sequence(ref_seq, context_start0, variant)
            if error is not None:
                errors.append((row_i, error))
                continue

            mask = exon_bin_mask(
                fetch_chrom,
                variant.pos,
                variant.gene,
                gene_exons,
                context_length,
                target_length,
                target_resolution,
            )
            if not mask.any():
                mask = exon_bin_mask(
                    variant.chrom,
                    variant.pos,
                    variant.gene,
                    gene_exons,
                    context_length,
                    target_length,
                    target_resolution,
                )
            if not mask.any():
                errors.append((row_i, "no_exon_bins"))
                continue

            ref = torch.as_tensor(
                sequence_to_onehot(ref_seq),
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)
            alt = torch.as_tensor(
                sequence_to_onehot(alt_seq),
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)

            with torch.no_grad():
                _loss_ref, pred_ref = forward_heads(
                    model,
                    heads_model,
                    ref,
                    organism_idx,
                    crop_bins,
                    use_amp=args.dtype == "mixed",
                    return_scaled=False,
                    requires_grad=False,
                )
                _loss_alt, pred_alt = forward_heads(
                    model,
                    heads_model,
                    alt,
                    organism_idx,
                    crop_bins,
                    use_amp=args.dtype == "mixed",
                    return_scaled=False,
                    requires_grad=False,
                )

            for modality in modalities:
                ref_np = pred_ref[modality].detach().cpu().numpy()[0]
                alt_np = pred_alt[modality].detach().cpu().numpy()[0]
                lfc = gene_lfc_from_prediction_stack(ref_np, alt_np, mask)
                scores[modality][row_i] = lfc if args.signed else np.abs(lfc)
    finally:
        fasta.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.out, "w") as h5:
        h5.attrs["score_name"] = args.score_dataset
        h5.attrs["source_checkpoint"] = str(checkpoint_path)
        h5.attrs["source_metadata"] = str(metadata_path)
        h5.attrs["source_pretrained_weights"] = str(pretrained_weights)
        h5.attrs["target_resolution"] = target_resolution
        h5.attrs["crop_bins"] = crop_bins
        h5.attrs["context_length"] = context_length
        write_variant_group(h5, variants, chrom_key="chrom")
        for modality, arr in scores.items():
            h5.create_dataset(
                f"scores/{modality}/{args.score_dataset}",
                data=arr,
                dtype="f4",
                compression="gzip",
            )
        write_errors(h5, errors)

    finite_rows = np.zeros(len(variants), dtype=bool)
    for arr in scores.values():
        finite_rows |= np.isfinite(arr).any(axis=1)
    print(f"Wrote {args.out}")
    print(f"Variants: {len(variants)}; scored: {finite_rows.sum()}; errors: {len(errors)}")

    write_epibrain_json(args, metadata_path, modalities)
    if args.json_out is not None:
        print(f"Wrote EpiBRAIN data paths JSON: {args.json_out}")


if __name__ == "__main__":
    main()
