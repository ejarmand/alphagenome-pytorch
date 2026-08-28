#!/usr/bin/env python
"""Prepare chromosome 1 variants for window-aware gene-level inference.

For each biallelic SNV, this script constructs the same centered window used by
``inference_geneVar.py``. It retains the variant when at least one exon from a
protein-coding gene overlaps that window, then assigns one nearest qualifying
gene. The output VCF contains sites only because inference scores each allele
once and does not read sample genotypes.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import heapq
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TextIO


@dataclass(frozen=True)
class Gene:
    chrom: str
    start: int
    end: int
    strand: str
    gene_id: str
    gene_name: str
    gene_biotype: str

    @property
    def tss(self) -> int:
        return self.start if self.strand == "+" else self.end


@dataclass(frozen=True)
class Exon:
    start: int
    end: int
    gene_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Retain SNVs whose centered model window overlaps a protein-coding "
            "exon and assign the nearest qualifying gene."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--vcf", type=Path, required=True)
    parser.add_argument("--gtf", type=Path, required=True)
    parser.add_argument("--chrom", required=True)
    parser.add_argument("--out-vcf", type=Path, required=True)
    parser.add_argument("--out-map", type=Path, required=True)
    parser.add_argument("--metadata-out", type=Path, required=True)
    parser.add_argument("--context-length", type=int, default=524_288)
    parser.add_argument("--gene-biotype", default="protein_coding")
    parser.add_argument(
        "--pass-only",
        action="store_true",
        help="Keep only source records whose FILTER value is PASS.",
    )
    parser.add_argument(
        "--max-source-records",
        type=int,
        default=None,
        help="Stop after this many source records. Intended for validation only.",
    )
    return parser.parse_args()


def open_text(path: Path, mode: str = "rt") -> TextIO:
    if str(path).endswith(".gz"):
        return gzip.open(path, mode, newline="")
    return path.open(mode, newline="")


def parse_gtf_attributes(attributes: str) -> dict[str, str]:
    return {
        key: value
        for key, value in re.findall(r'(\S+)\s+"([^"]*)"\s*;?', attributes)
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_genes_and_exons(
    gtf_path: Path,
    chrom: str,
    gene_biotype: str,
) -> tuple[dict[str, Gene], list[Exon]]:
    genes: dict[str, Gene] = {}
    raw_exons: list[tuple[str, int, int]] = []

    with open_text(gtf_path) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[0] != chrom:
                continue
            feature = parts[2]
            if feature not in {"gene", "exon"}:
                continue
            attributes = parse_gtf_attributes(parts[8])
            gene_id = attributes.get("gene_id")
            if not gene_id:
                continue
            start, end = int(parts[3]), int(parts[4])
            if feature == "gene":
                biotype = attributes.get(
                    "gene_biotype", attributes.get("gene_type", "")
                )
                if gene_biotype and biotype != gene_biotype:
                    continue
                genes[gene_id] = Gene(
                    chrom=chrom,
                    start=start,
                    end=end,
                    strand=parts[6],
                    gene_id=gene_id,
                    gene_name=attributes.get(
                        "gene_name", attributes.get("gene", gene_id)
                    ),
                    gene_biotype=biotype,
                )
            else:
                raw_exons.append((gene_id, start, end))

    exon_keys = {
        (gene_id, start, end)
        for gene_id, start, end in raw_exons
        if gene_id in genes
    }
    exons = sorted(
        (Exon(start=start, end=end, gene_id=gene_id)
         for gene_id, start, end in exon_keys),
        key=lambda exon: (exon.start, exon.end, exon.gene_id),
    )
    genes_with_exons = {exon.gene_id for exon in exons}
    genes = {
        gene_id: gene
        for gene_id, gene in genes.items()
        if gene_id in genes_with_exons
    }
    if not genes or not exons:
        raise ValueError(
            f"No {gene_biotype!r} genes with exons found on {chrom} in {gtf_path}"
        )
    return genes, exons


def distance_to_gene_body(pos: int, gene: Gene) -> int:
    if gene.start <= pos <= gene.end:
        return 0
    if pos < gene.start:
        return gene.start - pos
    return pos - gene.end


def centered_window(pos: int, context_length: int) -> tuple[int, int]:
    """Return the 1-based inclusive window used by inference_geneVar.py."""
    start0 = (pos - 1) - ((context_length - 1) // 2)
    end0 = start0 + context_length
    return start0 + 1, end0


def select_nearest_gene(
    pos: int,
    qualifying_gene_ids: set[str],
    genes: dict[str, Gene],
) -> tuple[Gene, int, int, int]:
    ranked = sorted(
        (
            distance_to_gene_body(pos, genes[gene_id]),
            abs(pos - genes[gene_id].tss),
            gene_id,
            genes[gene_id],
        )
        for gene_id in qualifying_gene_ids
    )
    body_distance, tss_distance, _gene_id, selected = ranked[0]
    primary_tie_count = sum(
        body_distance == candidate_body_distance
        for candidate_body_distance, _candidate_tss_distance, _candidate_id, _candidate
        in ranked
    )
    return selected, body_distance, tss_distance, primary_tie_count


def main() -> None:
    args = parse_args()
    if args.context_length <= 0:
        raise ValueError("--context-length must be positive")
    for output in (args.out_vcf, args.out_map, args.metadata_out):
        output.parent.mkdir(parents=True, exist_ok=True)

    genes, exons = read_genes_and_exons(
        args.gtf,
        args.chrom,
        args.gene_biotype,
    )

    map_fields = [
        "variant_index",
        "source_variant_index",
        "variant_key",
        "chrom",
        "pos",
        "id",
        "ref",
        "alt",
        "filter",
        "window_start",
        "window_end",
        "qualifying_gene_count",
        "gene_id",
        "gene_name",
        "gene_start",
        "gene_end",
        "gene_strand",
        "gene_biotype",
        "gene_tss",
        "assignment",
        "distance_bp",
        "distance_to_tss_bp",
        "nearest_body_distance_tie_count",
    ]

    source_count = 0
    selected_count = 0
    source_sample_count = 0
    skipped: Counter[str] = Counter()
    assignment_counts: Counter[str] = Counter()
    last_pos: int | None = None

    exon_cursor = 0
    active_counts: dict[str, int] = defaultdict(int)
    active_by_end: list[tuple[int, int, str]] = []
    heap_serial = 0

    with (
        open_text(args.vcf) as source,
        open_text(args.out_vcf, "wt") as out_vcf,
        open_text(args.out_map, "wt") as out_map,
    ):
        map_writer = csv.DictWriter(out_map, fieldnames=map_fields, delimiter="\t")
        map_writer.writeheader()

        for line in source:
            if line.startswith("##"):
                out_vcf.write(line)
                continue
            if line.startswith("#CHROM"):
                header_parts = line.rstrip("\n").split("\t")
                source_sample_count = max(0, len(header_parts) - 9)
                out_vcf.write(
                    "##alphagenome_subset_rule=524288bp centered window overlaps "
                    "a protein-coding exon; nearest qualifying gene assigned\n"
                )
                out_vcf.write("\t".join(header_parts[:8]) + "\n")
                continue
            if line.startswith("#"):
                continue
            if args.max_source_records is not None and source_count >= args.max_source_records:
                break

            parts = line.rstrip("\n").split("\t", 9)
            if len(parts) < 8:
                skipped["malformed"] += 1
                continue
            chrom, pos_text, variant_id, ref, alt, qual, filt, info = parts[:8]
            source_index = source_count
            source_count += 1

            if chrom != args.chrom:
                skipped["other_chromosome"] += 1
                continue
            pos = int(pos_text)
            if last_pos is not None and pos < last_pos:
                raise ValueError(
                    f"VCF is not coordinate sorted: {args.chrom}:{pos} follows {last_pos}"
                )
            last_pos = pos
            ref = ref.upper()
            alt = alt.upper()
            if "," in alt:
                skipped["multiallelic"] += 1
                continue
            if len(ref) != 1 or len(alt) != 1 or ref not in "ACGT" or alt not in "ACGT":
                skipped["non_snv"] += 1
                continue
            if args.pass_only and filt != "PASS":
                skipped["filter_not_pass"] += 1
                continue

            window_start, window_end = centered_window(pos, args.context_length)

            while exon_cursor < len(exons) and exons[exon_cursor].start <= window_end:
                exon = exons[exon_cursor]
                active_counts[exon.gene_id] += 1
                heapq.heappush(
                    active_by_end,
                    (exon.end, heap_serial, exon.gene_id),
                )
                heap_serial += 1
                exon_cursor += 1
            while active_by_end and active_by_end[0][0] < window_start:
                _end, _serial, gene_id = heapq.heappop(active_by_end)
                active_counts[gene_id] -= 1
                if active_counts[gene_id] == 0:
                    del active_counts[gene_id]

            qualifying_gene_ids = set(active_counts)
            if not qualifying_gene_ids:
                skipped["no_protein_coding_exon_in_window"] += 1
                continue

            gene, body_distance, tss_distance, tie_count = select_nearest_gene(
                pos,
                qualifying_gene_ids,
                genes,
            )
            if body_distance == 0:
                assignment = "overlapping_gene_body"
            else:
                assignment = "nearest_qualifying_gene"
            if tie_count > 1:
                assignment += "_tie_resolved_by_tss_then_gene_id"

            out_vcf.write("\t".join(parts[:8]) + "\n")
            variant_key = f"{chrom}:{pos}:{ref}:{alt}"
            map_writer.writerow(
                {
                    "variant_index": selected_count,
                    "source_variant_index": source_index,
                    "variant_key": variant_key,
                    "chrom": chrom,
                    "pos": pos,
                    "id": variant_id,
                    "ref": ref,
                    "alt": alt,
                    "filter": filt,
                    "window_start": window_start,
                    "window_end": window_end,
                    "qualifying_gene_count": len(qualifying_gene_ids),
                    "gene_id": gene.gene_id,
                    "gene_name": gene.gene_name,
                    "gene_start": gene.start,
                    "gene_end": gene.end,
                    "gene_strand": gene.strand,
                    "gene_biotype": gene.gene_biotype,
                    "gene_tss": gene.tss,
                    "assignment": assignment,
                    "distance_bp": body_distance,
                    "distance_to_tss_bp": tss_distance,
                    "nearest_body_distance_tie_count": tie_count,
                }
            )
            assignment_counts[assignment] += 1
            selected_count += 1

    if selected_count == 0:
        raise ValueError("No variants met the window and gene criteria")

    metadata = {
        "source_vcf": str(args.vcf.resolve()),
        "source_vcf_size_bytes": args.vcf.stat().st_size,
        "source_vcf_mtime_ns": args.vcf.stat().st_mtime_ns,
        "source_vcf_sample_count": source_sample_count,
        "source_records_examined": source_count,
        "gtf": str(args.gtf.resolve()),
        "gtf_sha256": sha256(args.gtf),
        "chrom": args.chrom,
        "context_length": args.context_length,
        "coordinate_convention": "VCF/GTF 1-based inclusive",
        "window_rule": (
            "start0=(POS-1)-((context_length-1)//2); end0=start0+context_length; "
            "reported window is start0+1 through end0, inclusive"
        ),
        "candidate_gene_rule": (
            f"{args.gene_biotype} gene with at least one exon overlapping the "
            "centered model window"
        ),
        "assignment_rule": (
            "minimum distance from POS to candidate gene body; ties resolved by "
            "distance to transcription start site, then gene_id"
        ),
        "variant_rule": "biallelic A/C/G/T SNVs",
        "pass_only": args.pass_only,
        "max_source_records": args.max_source_records,
        "output_vcf": str(args.out_vcf.resolve()),
        "output_vcf_is_sites_only": True,
        "output_vcf_sample_count": 0,
        "output_vcf_sha256": sha256(args.out_vcf),
        "output_map": str(args.out_map.resolve()),
        "output_map_sha256": sha256(args.out_map),
        "selected_variant_count": selected_count,
        "mapping_row_count": selected_count,
        "protein_coding_gene_count": len(genes),
        "unique_exon_count": len(exons),
        "assignment_counts": dict(sorted(assignment_counts.items())),
        "skipped_counts": dict(sorted(skipped.items())),
    }
    args.metadata_out.write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"Wrote sites-only VCF: {args.out_vcf}")
    print(f"Wrote variant-gene map: {args.out_map}")
    print(f"Wrote metadata: {args.metadata_out}")
    print(
        f"Source records examined: {source_count}; retained variants: "
        f"{selected_count}; source samples: {source_sample_count}"
    )
    print(f"Skipped: {dict(sorted(skipped.items()))}")


if __name__ == "__main__":
    main()
