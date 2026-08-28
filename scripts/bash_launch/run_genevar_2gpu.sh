#!/usr/bin/env bash

set -uo pipefail

project_dir=/home/lilx/projects/doggenetics/alphagenome_rna
checkpoint=finetuning_output/tfr_multitask_v4_final/model_final_datasetv4_200ep/best_heads.pt
pretrained_weights=weights/model_fold_1.safetensors
vcf=/home/datasets/deep_learning/application/lilx_dog/inference_benchmark/dog10k_GSD_chr1_180000_702621_200dogs.vcf.gz
variant_gene_map=/home/datasets/deep_learning/application/lilx_dog/inference_benchmark/dog10k_GSD_chr1_180000_702621_200dogs.variant_gene_map.tsv.gz
gtf=/home/datasets/deep_learning/application/lilx_dog/ref/UU_Cfam_GSD_1.0/GCF_011100685.1_UU_Cfam_GSD_1.0_genomic.gtf.gz
fasta=/home/lilx/projects/doggenetics/ref/genomes/ncbi_dataset/ncbi_dataset/data/GCF_011100685.1/GCF_011100685.1_UU_Cfam_GSD_1.0_genomic.fna

mode=${1:-smoke}
case "$mode" in
  smoke)
    start_gpu0=0
    end_gpu0=1
    start_gpu1=1
    end_gpu1=2
    output_gpu0=results/inference/genevar_smoke_gpu0.h5
    output_gpu1=results/inference/genevar_smoke_gpu1.h5
    log_gpu0=logs/genevar_smoke_gpu0.log
    log_gpu1=logs/genevar_smoke_gpu1.log
    ;;
  full)
    start_gpu0=0
    end_gpu0=4985
    start_gpu1=4985
    end_gpu1=9969
    output_gpu0=results/inference/genevar_full_rows_0000_4985.h5
    output_gpu1=results/inference/genevar_full_rows_4985_9969.h5
    log_gpu0=logs/genevar_full_gpu0.log
    log_gpu1=logs/genevar_full_gpu1.log
    ;;
  *)
    echo "Usage: $0 [smoke|full]" >&2
    exit 2
    ;;
esac

cd "$project_dir"
mkdir -p logs results/inference

run_shard() {
  local gpu=$1
  local start=$2
  local end=$3
  local output=$4
  local log=$5

  CUDA_VISIBLE_DEVICES="$gpu" \
  PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  pixi run python -u \
    alphagenome-pytorch-tfr/scripts/inference_geneVar.py \
    "$checkpoint" \
    --pretrained-weights "$pretrained_weights" \
    --vcf "$vcf" \
    --variant-gene-map "$variant_gene_map" \
    --gtf "$gtf" \
    --fasta "$fasta" \
    --modality rna \
    --modality atac \
    --device cuda \
    --dtype mixed \
    --start "$start" \
    --end "$end" \
    --out "$output" \
    > >(tee "$log") 2>&1
}

echo "Starting $mode shard on physical GPU 0: rows [$start_gpu0, $end_gpu0)"
run_shard 0 "$start_gpu0" "$end_gpu0" "$output_gpu0" "$log_gpu0" &
pid_gpu0=$!

echo "Starting $mode shard on physical GPU 1: rows [$start_gpu1, $end_gpu1)"
run_shard 1 "$start_gpu1" "$end_gpu1" "$output_gpu1" "$log_gpu1" &
pid_gpu1=$!

echo "GPU 0 worker PID: $pid_gpu0"
echo "GPU 1 worker PID: $pid_gpu1"

wait "$pid_gpu0"
status_gpu0=$?
wait "$pid_gpu1"
status_gpu1=$?

echo "GPU 0 worker exit status: $status_gpu0"
echo "GPU 1 worker exit status: $status_gpu1"

if (( status_gpu0 != 0 || status_gpu1 != 0 )); then
  exit 1
fi

echo "Both $mode shards completed successfully."

if [[ "$mode" == "full" ]]; then
  merged_output=results/inference/genevar_full_200dogs.h5
  pixi run python alphagenome-pytorch-tfr/scripts/merge_genevar_h5.py \
    --input "$output_gpu0" \
    --input "$output_gpu1" \
    --out "$merged_output"
  echo "Merged full inference output: $merged_output"
fi
