#!/usr/bin/env bash

set -uo pipefail

project_dir=/home/lilx/projects/doggenetics/alphagenome_rna
checkpoint=finetuning_output/tfr_multitask_v4_final/model_final_datasetv4_200ep/best_heads.pt
pretrained_weights=weights/model_fold_1.safetensors
input_dir=/home/datasets/deep_learning/application/lilx_dog/inference_chr1_512kb
vcf=$input_dir/dog10k_chr1_512kb_exon_nearest_gene.sites.vcf.gz
variant_gene_map=$input_dir/dog10k_chr1_512kb_exon_nearest_gene.variant_gene_map.tsv.gz
metadata_json=$input_dir/dog10k_chr1_512kb_exon_nearest_gene.metadata.json
gtf=/home/datasets/deep_learning/application/lilx_dog/ref/UU_Cfam_GSD_1.0/GCF_011100685.1_UU_Cfam_GSD_1.0_genomic.gtf.gz
fasta=/home/lilx/projects/doggenetics/ref/genomes/ncbi_dataset/ncbi_dataset/data/GCF_011100685.1/GCF_011100685.1_UU_Cfam_GSD_1.0_genomic.fna

cd "$project_dir"
mkdir -p logs results/inference/chr1_512kb

for required_file in \
  "$checkpoint" \
  "$pretrained_weights" \
  "$vcf" \
  "$variant_gene_map" \
  "$metadata_json" \
  "$gtf" \
  "$fasta"; do
  if [[ ! -f "$required_file" ]]; then
    echo "Required file not found: $required_file" >&2
    exit 1
  fi
done

row_count=$(pixi run python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["mapping_row_count"])' \
  "$metadata_json")

mode=${1:-smoke}
case "$mode" in
  smoke)
    if (( row_count < 2 )); then
      echo "The prepared map needs at least two rows for the two-GPU smoke test" >&2
      exit 1
    fi
    start_gpu0=0
    end_gpu0=1
    start_gpu1=1
    end_gpu1=2
    output_gpu0=results/inference/chr1_512kb/smoke_gpu0.h5
    output_gpu1=results/inference/chr1_512kb/smoke_gpu1.h5
    log_gpu0=logs/chr1_512kb_smoke_gpu0.log
    log_gpu1=logs/chr1_512kb_smoke_gpu1.log
    ;;
  pilot)
    if (( row_count < 100 )); then
      echo "The prepared map needs at least 100 rows for the pilot" >&2
      exit 1
    fi
    start_gpu0=0
    end_gpu0=50
    start_gpu1=50
    end_gpu1=100
    output_gpu0=results/inference/chr1_512kb/pilot_rows_0_50.h5
    output_gpu1=results/inference/chr1_512kb/pilot_rows_50_100.h5
    log_gpu0=logs/chr1_512kb_pilot_gpu0.log
    log_gpu1=logs/chr1_512kb_pilot_gpu1.log
    ;;
  full)
    if [[ "${ALLOW_LONG_FULL_RUN:-0}" != "1" ]]; then
      echo "Full inference contains $row_count variants and is estimated to take" >&2
      echo "about 33 days at the previously measured two-GPU rate." >&2
      echo "Run the pilot and review runtime before setting ALLOW_LONG_FULL_RUN=1." >&2
      exit 2
    fi
    midpoint=$(( (row_count + 1) / 2 ))
    start_gpu0=0
    end_gpu0=$midpoint
    start_gpu1=$midpoint
    end_gpu1=$row_count
    output_gpu0=results/inference/chr1_512kb/rows_0_${midpoint}.h5
    output_gpu1=results/inference/chr1_512kb/rows_${midpoint}_${row_count}.h5
    log_gpu0=logs/chr1_512kb_full_gpu0.log
    log_gpu1=logs/chr1_512kb_full_gpu1.log
    ;;
  *)
    echo "Usage: $0 [smoke|pilot|full]" >&2
    exit 2
    ;;
esac

for output in "$output_gpu0" "$output_gpu1"; do
  if [[ -e "$output" ]]; then
    echo "Refusing to replace existing output: $output" >&2
    exit 1
  fi
done

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
    alphagenome-pytorch-tfr/scripts/inference/inference_geneVar.py \
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
    --signed \
    --score-dataset gene_lfc \
    --stream-selected-rows \
    --start "$start" \
    --end "$end" \
    --out "$output" \
    > >(tee "$log") 2>&1
}

echo "Prepared rows: $row_count"
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
  merged_output=results/inference/chr1_512kb/chr1_512kb_nearest_gene_signed.h5
  if [[ -e "$merged_output" ]]; then
    echo "Refusing to replace existing merged output: $merged_output" >&2
    exit 1
  fi
  pixi run python alphagenome-pytorch-tfr/scripts/merge_genevar_h5.py \
    --input "$output_gpu0" \
    --input "$output_gpu1" \
    --out "$merged_output"
  echo "Merged full inference output: $merged_output"
fi
