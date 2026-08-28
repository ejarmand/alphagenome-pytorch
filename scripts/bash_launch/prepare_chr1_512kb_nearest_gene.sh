#!/usr/bin/env bash

set -euo pipefail

project_dir=/home/lilx/projects/doggenetics/alphagenome_rna
source_vcf=/home/datasets/deep_learning/application/lilx_dog/10k_vcf_by_chr/dog10k_chr1.vcf.gz
gtf=/home/datasets/deep_learning/application/lilx_dog/ref/UU_Cfam_GSD_1.0/GCF_011100685.1_UU_Cfam_GSD_1.0_genomic.gtf.gz
output_dir=/home/datasets/deep_learning/application/lilx_dog/inference_chr1_512kb
output_vcf=$output_dir/dog10k_chr1_512kb_exon_nearest_gene.sites.vcf.gz
output_map=$output_dir/dog10k_chr1_512kb_exon_nearest_gene.variant_gene_map.tsv.gz
metadata=$output_dir/dog10k_chr1_512kb_exon_nearest_gene.metadata.json

cd "$project_dir"
mkdir -p "$output_dir" logs

if [[ -e "$output_vcf" || -e "$output_map" || -e "$metadata" ]]; then
  echo "Refusing to replace an existing prepared file in $output_dir" >&2
  exit 1
fi

PYTHONUNBUFFERED=1 \
pixi run python -u \
  alphagenome-pytorch-tfr/scripts/inference/make_chr1_window_variant_gene_table.py \
  --vcf "$source_vcf" \
  --gtf "$gtf" \
  --chrom NC_049222.1 \
  --context-length 524288 \
  --out-vcf "$output_vcf" \
  --out-map "$output_map" \
  --metadata-out "$metadata" \
  2>&1 | tee logs/prepare_chr1_512kb_nearest_gene.log
