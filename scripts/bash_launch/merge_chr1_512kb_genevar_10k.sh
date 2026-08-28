#!/usr/bin/env bash

set -euo pipefail

project_dir=/home/lilx/projects/doggenetics/alphagenome_rna
manifest=/home/datasets/deep_learning/application/lilx_dog/inference_chr1_512kb/chr1_512kb_genevar_10k_shards.tsv
output=results/inference/chr1_512kb/chr1_512kb_nearest_gene_signed.h5

cd "$project_dir"

pixi run python \
  alphagenome-pytorch-tfr/scripts/inference/merge_genevar_h5_from_manifest.py \
  --manifest "$manifest" \
  --out "$output"
