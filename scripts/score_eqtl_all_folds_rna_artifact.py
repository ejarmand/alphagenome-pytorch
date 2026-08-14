#!/usr/bin/env python
"""Score eQTL VCFs with pretrained AlphaGenome all-fold RNA outputs.

This writes the HDF5 artifact shape expected by EpiBRAIN's
Analysis/07_eQTL ``alphagenome_custom_heads`` loader:

    variants/{chrom,pos,ref,alt,id,gene}
    scores/RNA/gene_abs_lfc

Only the pretrained built-in ``rna_seq`` head is scored. Padding tracks are
dropped, and reverse-complement inference is averaged after strand-pair
switching derived from the built-in human track metadata.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
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
from alphagenome_pytorch.named_outputs import TrackMetadata, TrackMetadataCatalog  # noqa: E402
from alphagenome_pytorch.utils.sequence import reverse_complement_onehot_tensor  # noqa: E402
from finetune_tfr_heads import switch_reverse_predictions  # noqa: E402
from score_eqtl_heads_artifact import (  # noqa: E402
    exon_bin_mask,
    fetch_centered_sequence,
    gene_lfc_from_prediction_stack,
    make_snv_alt_sequence,
    parse_gtf_exons,
    read_vcf_variants,
    sequence_to_onehot,
    write_errors,
    write_variant_group,
)


DEFAULT_WEIGHTS = SCRIPT_DIR.parent / "weights" / "model_all_folds.safetensors"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an EpiBRAIN 07_eQTL artifact from all-fold AlphaGenome RNA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--vcf", type=Path, required=True)
    parser.add_argument("--gtf", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--metadata-out",
        type=Path,
        default=None,
        help="Track metadata JSON for EpiBRAIN. Defaults to <out>.metadata.json.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional EpiBRAIN data_paths JSON for this artifact.",
    )
    parser.add_argument("--eval-vcf", type=Path, default=None)
    parser.add_argument("--eval-info", type=Path, default=None)
    parser.add_argument("--exclude", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-name", default="ag_all_folds_rna")
    parser.add_argument("--plot-name", default="AlphaGenome all-folds RNA")
    parser.add_argument(
        "--gene-info-keys",
        default="gene_name,gene_ID,gene_id,MT",
        help="Comma-separated VCF INFO keys used to map variants to genes.",
    )
    parser.add_argument("--organism-idx", type=int, default=0, help="0 is human.")
    parser.add_argument("--context-length", type=int, default=1_048_576)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["mixed", "float32"], default="mixed")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--score-dataset", default="gene_abs_lfc")
    parser.add_argument("--signed", action="store_true")
    parser.add_argument(
        "--no-rc-average",
        action="store_true",
        help="Disable reverse-complement test-time averaging.",
    )
    return parser.parse_args()


def dtype_policy(name: str) -> DtypePolicy:
    if name == "float32":
        return DtypePolicy.full_float32()
    return DtypePolicy.mixed_precision()


def metadata_pair_key(track: TrackMetadata) -> tuple:
    data = track.to_dict()
    return tuple(
        sorted(
            (key, value)
            for key, value in data.items()
            if key not in {"track_index", "strand"}
        )
    )


def rna_tracks_and_strand_pair(
    model: AlphaGenome,
    organism_idx: int,
) -> tuple[list[TrackMetadata], list[int], torch.Tensor]:
    catalog = TrackMetadataCatalog.load_builtin(organism_idx)
    tracks = list(
        catalog.get_tracks(
            "rna_seq",
            organism=organism_idx,
            num_tracks=model.heads["rna_seq"].num_tracks,
            strict=True,
        )
    )
    key_to_index_by_strand: dict[tuple, dict[str, int]] = {}
    for track in tracks:
        key_to_index_by_strand.setdefault(metadata_pair_key(track), {})[
            str(track.get("strand", "."))
        ] = int(track.track_index)

    strand_pair = []
    for track in tracks:
        strand = str(track.get("strand", "."))
        mate_strand = "-" if strand == "+" else "+" if strand == "-" else strand
        mate_index = key_to_index_by_strand.get(metadata_pair_key(track), {}).get(
            mate_strand,
            int(track.track_index),
        )
        strand_pair.append(mate_index)

    keep_tracks = [track for track in tracks if not track.is_padding]
    keep_indices = [int(track.track_index) for track in keep_tracks]
    return keep_tracks, keep_indices, torch.as_tensor(strand_pair, dtype=torch.long)


def write_track_metadata_json(
    path: Path,
    tracks: list[TrackMetadata],
    keep_indices: list[int],
    strand_pair_full: torch.Tensor,
    args: argparse.Namespace,
) -> None:
    full_to_local = {full_i: local_i for local_i, full_i in enumerate(keep_indices)}
    local_strand_pair = []
    for full_i in keep_indices:
        pair_full_i = int(strand_pair_full[full_i].item())
        local_strand_pair.append(full_to_local.get(pair_full_i, full_to_local[full_i]))

    track_names = []
    cell_types = []
    track_strands = []
    full_indices = []
    for track in tracks:
        track_names.append(track.track_name)
        cell_types.append(str(track.get("biosample_name", track.track_name)))
        track_strands.append(str(track.get("strand", ".")))
        full_indices.append(int(track.track_index))

    metadata = {
        "source": "alphagenome_all_folds_builtin_rna_seq",
        "weights": str(args.weights),
        "organism_idx": args.organism_idx,
        "target_resolution": 128,
        "target_length": args.context_length // 128,
        "crop_bins": 0,
        "modalities": ["RNA"],
        "assay_types": {"RNA": "rna_seq"},
        "n_tracks": {"RNA": len(tracks)},
        "track_names": {"RNA": track_names},
        "cell_types": {"RNA": cell_types},
        "strand_pair": {"RNA": local_strand_pair},
        "track_strands": {"RNA": track_strands},
        "source_track_indices": {"RNA": full_indices},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")


def write_epibrain_json(
    args: argparse.Namespace,
    metadata_path: Path,
) -> None:
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
                "heads": ["RNA"],
            }
        },
        "plot_name": {args.model_name: args.plot_name},
    }
    if args.exclude is not None:
        cfg["exclude"] = str(args.exclude.resolve())
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    with args.json_out.open("w") as handle:
        json.dump(cfg, handle, indent=2)
        handle.write("\n")


def predict_rna(
    model: AlphaGenome,
    sequences: torch.Tensor,
    organism_idx: torch.Tensor,
    strand_pair: torch.Tensor,
    keep_indices: torch.Tensor,
    rc_average: bool,
) -> torch.Tensor:
    pred_fwd = model.predict(
        sequences,
        organism_idx,
        resolutions=(128,),
        heads=("rna_seq",),
        channels_last=True,
    )["rna_seq"][128]
    if not rc_average:
        return pred_fwd.index_select(-1, keep_indices)

    pred_rc = model.predict(
        reverse_complement_onehot_tensor(sequences),
        organism_idx,
        resolutions=(128,),
        heads=("rna_seq",),
        channels_last=True,
    )["rna_seq"][128]
    reverse_complement = torch.ones(
        sequences.shape[0],
        dtype=torch.bool,
        device=sequences.device,
    )
    pred_rc = switch_reverse_predictions(pred_rc, reverse_complement, strand_pair)
    return ((pred_fwd + pred_rc) * 0.5).index_select(-1, keep_indices)


def main() -> None:
    args = parse_args()
    metadata_path = args.metadata_out or args.out.with_suffix(".metadata.json")
    rc_average = not args.no_rc_average
    device = torch.device(args.device)

    model = AlphaGenome.from_pretrained(
        args.weights,
        dtype_policy=dtype_policy(args.dtype),
        device=device,
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    tracks, keep_indices, strand_pair = rna_tracks_and_strand_pair(model, args.organism_idx)
    strand_pair = strand_pair.to(device)
    keep_indices_tensor = torch.as_tensor(keep_indices, dtype=torch.long, device=device)

    gene_info_keys = [key for key in args.gene_info_keys.split(",") if key]
    all_variants = read_vcf_variants(args.vcf, gene_info_keys=gene_info_keys)
    variants = all_variants[args.start : args.end]
    genes = sorted({variant.gene for variant in variants if variant.gene})
    gene_exons = parse_gtf_exons(args.gtf, genes)

    scores = np.full((len(variants), len(keep_indices)), np.nan, dtype=np.float32)
    errors: list[tuple[int, str]] = []
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    fasta = pysam.Fastafile(str(args.fasta))
    try:
        pbar = tqdm(range(0, len(variants), args.batch_size), desc="Scoring all-fold RNA eQTL")
        for batch_start in pbar:
            batch = variants[batch_start : batch_start + args.batch_size]
            ref_arrays = []
            alt_arrays = []
            masks = []
            valid_rows = []
            target_length = args.context_length // 128

            for offset, variant in enumerate(batch):
                row_i = batch_start + offset
                if variant.gene is None:
                    errors.append((row_i, "missing_gene"))
                    continue
                if variant.gene not in gene_exons:
                    errors.append((row_i, "missing_gene_exons"))
                    continue

                try:
                    ref_seq, context_start0, fetch_chrom = fetch_centered_sequence(
                        fasta,
                        variant.chrom,
                        variant.pos,
                        args.context_length,
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
                        args.context_length,
                        target_length,
                        128,
                    )
                    if not mask.any():
                        mask = exon_bin_mask(
                            variant.chrom,
                            variant.pos,
                            variant.gene,
                            gene_exons,
                            args.context_length,
                            target_length,
                            128,
                        )
                    if not mask.any():
                        errors.append((row_i, "no_exon_bins"))
                        continue

                    ref_arrays.append(sequence_to_onehot(ref_seq))
                    alt_arrays.append(sequence_to_onehot(alt_seq))
                    masks.append(mask)
                    valid_rows.append(row_i)
                except Exception as exc:
                    errors.append((row_i, str(exc)))

            if not valid_rows:
                continue

            ref = torch.as_tensor(np.stack(ref_arrays), dtype=torch.float32, device=device)
            alt = torch.as_tensor(np.stack(alt_arrays), dtype=torch.float32, device=device)
            organism_idx = torch.full(
                (ref.shape[0],),
                args.organism_idx,
                dtype=torch.long,
                device=device,
            )

            with torch.no_grad():
                pred_ref = predict_rna(
                    model,
                    ref,
                    organism_idx,
                    strand_pair,
                    keep_indices_tensor,
                    rc_average=rc_average,
                )
                pred_alt = predict_rna(
                    model,
                    alt,
                    organism_idx,
                    strand_pair,
                    keep_indices_tensor,
                    rc_average=rc_average,
                )

            pred_ref_np = pred_ref.detach().cpu().numpy()
            pred_alt_np = pred_alt.detach().cpu().numpy()
            for batch_i, row_i in enumerate(valid_rows):
                lfc = gene_lfc_from_prediction_stack(
                    pred_ref_np[batch_i],
                    pred_alt_np[batch_i],
                    masks[batch_i],
                )
                scores[row_i] = lfc if args.signed else np.abs(lfc)
    finally:
        fasta.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.out, "w") as h5:
        h5.attrs["score_name"] = args.score_dataset
        h5.attrs["source_weights"] = str(args.weights)
        h5.attrs["target_resolution"] = 128
        h5.attrs["context_length"] = args.context_length
        h5.attrs["batch_size"] = args.batch_size
        h5.attrs["organism_idx"] = args.organism_idx
        h5.attrs["rc_average"] = rc_average
        write_variant_group(h5, variants, chrom_key="chrom")
        h5.create_dataset(
            f"scores/RNA/{args.score_dataset}",
            data=scores,
            dtype="f4",
            compression="gzip",
        )
        write_errors(h5, errors)

    write_track_metadata_json(metadata_path, tracks, keep_indices, strand_pair.cpu(), args)
    write_epibrain_json(args, metadata_path)

    finite_rows = np.isfinite(scores).any(axis=1)
    print(f"Wrote {args.out}")
    print(f"Wrote track metadata: {metadata_path}")
    if args.json_out is not None:
        print(f"Wrote EpiBRAIN data paths JSON: {args.json_out}")
    print(f"Variants: {len(variants)}; scored: {int(finite_rows.sum())}; errors: {len(errors)}")


if __name__ == "__main__":
    main()
