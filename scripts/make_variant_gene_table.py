#!/usr/bin/env python
"""Create a reproducible variant-to-gene mapping table from a VCF and GTF.

By default, candidate genes are protein-coding genes whose gene bodies overlap
the coordinate span covered by the VCF. A variant is assigned to every gene
body it overlaps. If it overlaps none, it is assigned to the nearest candidate
gene body on the same chromosome.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, TextIO


@dataclass(frozen=True)
class Variant:
    index: int
    chrom: str
    pos: int
    variant_id: str
    ref: str
    alt: str
    filter: str

    @property
    def key(self) -> str:
        return f"{self.chrom}:{self.pos}:{self.ref}:{self.alt}"


@dataclass(frozen=True)
class Gene:
    chrom: str
    start: int
    end: int
    strand: str
    gene_id: str
    gene_name: str
    gene_biotype: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map VCF variants to overlapping or nearest genes from a GTF.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--vcf", type=Path, required=True)
    parser.add_argument("--gtf", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--metadata-out",
        type=Path,
        default=None,
        help="JSON provenance file. Defaults to <out>.metadata.json.",
    )
    parser.add_argument("--gene-biotype", default="protein_coding")
    parser.add_argument(
        "--gene-id",
        action="append",
        default=None,
        help="Restrict candidates to these gene IDs. Repeat for multiple genes.",
    )
    return parser.parse_args()


def open_text(path: Path, mode: str = "rt") -> TextIO:
    if str(path).endswith(".gz"):
        return gzip.open(path, mode, newline="")
    return path.open(mode, newline="")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_gtf_attributes(attributes: str) -> dict[str, str]:
    return {
        key: value
        for key, value in re.findall(r'(\S+)\s+"([^"]*)"\s*;?', attributes)
    }


def read_variants(path: Path) -> tuple[list[Variant], list[str]]:
    variants: list[Variant] = []
    samples: list[str] = []
    with open_text(path) as handle:
        for line in handle:
            if line.startswith("#CHROM"):
                samples = line.rstrip("\n").split("\t")[9:]
                continue
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 8:
                raise ValueError(f"VCF row has fewer than 8 columns: {line[:120]!r}")
            chrom, pos, variant_id, ref, alts, _qual, filt = parts[:7]
            if "," in alts:
                raise ValueError(
                    f"Multiallelic VCF row found at {chrom}:{pos}; split ALT alleles first"
                )
            variants.append(
                Variant(
                    index=len(variants),
                    chrom=chrom,
                    pos=int(pos),
                    variant_id=variant_id,
                    ref=ref.upper(),
                    alt=alts.upper(),
                    filter=filt,
                )
            )
    if not variants:
        raise ValueError(f"No variants found in {path}")
    return variants, samples


def vcf_spans(variants: Iterable[Variant]) -> dict[str, tuple[int, int]]:
    positions: dict[str, list[int]] = defaultdict(list)
    for variant in variants:
        positions[variant.chrom].append(variant.pos)
    return {
        chrom: (min(chrom_positions), max(chrom_positions))
        for chrom, chrom_positions in positions.items()
    }


def read_candidate_genes(
    path: Path,
    spans: dict[str, tuple[int, int]],
    gene_biotype: str,
    requested_gene_ids: set[str] | None,
) -> list[Gene]:
    genes: list[Gene] = []
    with open_text(path) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] != "gene":
                continue
            chrom = parts[0]
            if chrom not in spans:
                continue
            start, end = int(parts[3]), int(parts[4])
            span_start, span_end = spans[chrom]
            if end < span_start or start > span_end:
                continue
            attributes = parse_gtf_attributes(parts[8])
            gene_id = attributes.get("gene_id")
            if not gene_id:
                continue
            biotype = attributes.get(
                "gene_biotype", attributes.get("gene_type", "")
            )
            if gene_biotype and biotype != gene_biotype:
                continue
            if requested_gene_ids is not None and gene_id not in requested_gene_ids:
                continue
            genes.append(
                Gene(
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
            )
    if not genes:
        raise ValueError(
            f"No {gene_biotype!r} candidate genes overlap the VCF coordinate span"
        )
    return sorted(genes, key=lambda gene: (gene.chrom, gene.start, gene.end, gene.gene_id))


def distance_to_gene_body(pos: int, gene: Gene) -> int:
    if gene.start <= pos <= gene.end:
        return 0
    if pos < gene.start:
        return gene.start - pos
    return pos - gene.end


def assign_genes(variant: Variant, genes: list[Gene]) -> list[tuple[Gene, str, int]]:
    same_chrom = [gene for gene in genes if gene.chrom == variant.chrom]
    if not same_chrom:
        return []
    overlaps = [
        gene for gene in same_chrom if gene.start <= variant.pos <= gene.end
    ]
    if overlaps:
        return [(gene, "overlap", 0) for gene in overlaps]
    distances = [(distance_to_gene_body(variant.pos, gene), gene) for gene in same_chrom]
    minimum = min(distance for distance, _gene in distances)
    nearest = [gene for distance, gene in distances if distance == minimum]
    assignment = "nearest" if len(nearest) == 1 else "nearest_tie"
    return [(gene, assignment, minimum) for gene in nearest]


def default_metadata_path(out_path: Path) -> Path:
    name = out_path.name
    if name.endswith(".gz"):
        name = name[:-3]
    return out_path.with_name(f"{name}.metadata.json")


def main() -> None:
    args = parse_args()
    variants, samples = read_variants(args.vcf)
    spans = vcf_spans(variants)
    requested_gene_ids = set(args.gene_id) if args.gene_id else None
    genes = read_candidate_genes(
        args.gtf,
        spans,
        args.gene_biotype,
        requested_gene_ids,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "variant_index",
        "variant_key",
        "chrom",
        "pos",
        "id",
        "ref",
        "alt",
        "filter",
        "gene_id",
        "gene_name",
        "gene_start",
        "gene_end",
        "gene_strand",
        "gene_biotype",
        "assignment",
        "distance_bp",
    ]
    assignment_counts: Counter[str] = Counter()
    assigned_variant_indices: set[int] = set()
    row_count = 0
    with open_text(args.out, "wt") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for variant in variants:
            assignments = assign_genes(variant, genes)
            for gene, assignment, distance in assignments:
                writer.writerow(
                    {
                        "variant_index": variant.index,
                        "variant_key": variant.key,
                        "chrom": variant.chrom,
                        "pos": variant.pos,
                        "id": variant.variant_id,
                        "ref": variant.ref,
                        "alt": variant.alt,
                        "filter": variant.filter,
                        "gene_id": gene.gene_id,
                        "gene_name": gene.gene_name,
                        "gene_start": gene.start,
                        "gene_end": gene.end,
                        "gene_strand": gene.strand,
                        "gene_biotype": gene.gene_biotype,
                        "assignment": assignment,
                        "distance_bp": distance,
                    }
                )
                row_count += 1
                assignment_counts[assignment] += 1
                assigned_variant_indices.add(variant.index)

    metadata_path = args.metadata_out or default_metadata_path(args.out)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "vcf": str(args.vcf.resolve()),
        "vcf_sha256": sha256(args.vcf),
        "gtf": str(args.gtf.resolve()),
        "gtf_sha256": sha256(args.gtf),
        "output": str(args.out.resolve()),
        "coordinate_convention": "VCF/GTF 1-based inclusive",
        "candidate_gene_rule": (
            f"{args.gene_biotype} genes whose gene bodies overlap each "
            "chromosome span represented in the VCF"
        ),
        "assignment_rule": (
            "all overlapping candidate gene bodies; otherwise nearest candidate "
            "gene body on the same chromosome"
        ),
        "vcf_spans": {
            chrom: {"start": start, "end": end}
            for chrom, (start, end) in spans.items()
        },
        "variant_count": len(variants),
        "sample_count": len(samples),
        "mapping_row_count": row_count,
        "assigned_variant_count": len(assigned_variant_indices),
        "unassigned_variant_count": len(variants) - len(assigned_variant_indices),
        "assignment_row_counts": dict(sorted(assignment_counts.items())),
        "candidate_genes": [asdict(gene) for gene in genes],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"Wrote mapping table: {args.out}")
    print(f"Wrote metadata: {metadata_path}")
    print(
        f"Variants: {len(variants)}; samples: {len(samples)}; "
        f"mapping rows: {row_count}; unassigned: "
        f"{len(variants) - len(assigned_variant_indices)}"
    )
    print("Candidate genes: " + ", ".join(gene.gene_id for gene in genes))
    print(
        "Assignments: "
        + ", ".join(
            f"{assignment}={count}"
            for assignment, count in sorted(assignment_counts.items())
        )
    )


if __name__ == "__main__":
    main()
