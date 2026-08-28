#!/usr/bin/env bash

set -uo pipefail

project_dir=/home/lilx/projects/doggenetics/alphagenome_rna
checkpoint=finetuning_output/tfr_multitask_v4_final/model_final_datasetv4_200ep/best_heads.pt
pretrained_weights=weights/model_fold_1.safetensors
input_dir=/home/datasets/deep_learning/application/lilx_dog/inference_chr1_512kb
vcf=$input_dir/dog10k_chr1_512kb_exon_nearest_gene.sites.vcf.gz
variant_gene_map=$input_dir/dog10k_chr1_512kb_exon_nearest_gene.variant_gene_map.tsv.gz
manifest=$input_dir/chr1_512kb_genevar_10k_shards.tsv
gtf=/home/datasets/deep_learning/application/lilx_dog/ref/UU_Cfam_GSD_1.0/GCF_011100685.1_UU_Cfam_GSD_1.0_genomic.gtf.gz
fasta=/home/lilx/projects/doggenetics/ref/genomes/ncbi_dataset/ncbi_dataset/data/GCF_011100685.1/GCF_011100685.1_UU_Cfam_GSD_1.0_genomic.fna
inference_script=alphagenome-pytorch-tfr/scripts/inference/inference_geneVar.py
validator=alphagenome-pytorch-tfr/scripts/inference/validate_genevar_shard.py

cd "$project_dir"

for required_file in \
  "$checkpoint" "$pretrained_weights" "$vcf" "$variant_gene_map" \
  "$manifest" "$gtf" "$fasta" "$inference_script" "$validator"; do
  if [[ ! -f "$required_file" ]]; then
    echo "Required file not found: $required_file" >&2
    exit 1
  fi
done

if pgrep -f 'scripts/inference/inference_geneVar.py' >/dev/null; then
  echo "Another gene-variant inference process is already running." >&2
  echo "Stop it or wait for it to finish before starting this launcher." >&2
  exit 1
fi

run_worker() {
  local gpu=$1
  local completed=0
  local skipped=0

  while IFS=$'\t' read -r chunk_id start end row_count gpu_slot output log; do
    if [[ "$chunk_id" == "chunk_id" || "$gpu_slot" != "$gpu" ]]; then
      continue
    fi

    mkdir -p "$(dirname "$output")" "$(dirname "$log")"
    if [[ -f "$output" ]]; then
      if pixi run python "$validator" "$output" \
        --expected-start "$start" --expected-end "$end" --quiet; then
        echo "GPU $gpu skipping valid chunk $chunk_id rows [$start, $end)"
        skipped=$((skipped + 1))
        continue
      fi
      echo "GPU $gpu found an invalid completed file: $output" >&2
      echo "Move that file aside before restarting." >&2
      return 1
    fi

    partial=${output}.partial
    if [[ -e "$partial" ]]; then
      stale=${partial}.stale.$(date -u +%Y%m%dT%H%M%SZ)
      echo "GPU $gpu moving stale partial file to $stale"
      mv "$partial" "$stale"
    fi
    if [[ -e "$log" ]]; then
      stale_log=${log}.previous.$(date -u +%Y%m%dT%H%M%SZ)
      mv "$log" "$stale_log"
    fi

    echo "GPU $gpu starting chunk $chunk_id rows [$start, $end)"
    if ! CUDA_VISIBLE_DEVICES="$gpu" \
      PYTHONFAULTHANDLER=1 \
      PYTHONUNBUFFERED=1 \
      pixi run python -u "$inference_script" \
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
        --out "$partial" \
        2>&1 | tee "$log"; then
      echo "GPU $gpu failed on chunk $chunk_id; see $log" >&2
      return 1
    fi

    if ! pixi run python "$validator" "$partial" \
      --expected-start "$start" --expected-end "$end"; then
      echo "GPU $gpu produced an invalid partial shard for chunk $chunk_id" >&2
      return 1
    fi
    mv "$partial" "$output"
    completed=$((completed + 1))
    echo "GPU $gpu completed chunk $chunk_id: $output"
  done < "$manifest"

  echo "GPU $gpu worker finished; new chunks=$completed skipped chunks=$skipped"
}

echo "Starting resumable 10,000-variant inference workers"
run_worker 0 &
worker_gpu0=$!
run_worker 1 &
worker_gpu1=$!
echo "GPU 0 worker PID: $worker_gpu0"
echo "GPU 1 worker PID: $worker_gpu1"

wait "$worker_gpu0"
status_gpu0=$?
wait "$worker_gpu1"
status_gpu1=$?

echo "GPU 0 worker exit status: $status_gpu0"
echo "GPU 1 worker exit status: $status_gpu1"
if (( status_gpu0 != 0 || status_gpu1 != 0 )); then
  exit 1
fi
echo "All manifest shards completed successfully."
