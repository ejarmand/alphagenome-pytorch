#!/usr/bin/env bash
#SBATCH --job-name=ag_chr1_reverse
#SBATCH --account=bgup-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --array=0-79%2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/chr1_512kb_reverse/slurm_%A_%a.out
#SBATCH --error=logs/chr1_512kb_reverse/slurm_%A_%a.err

set -euo pipefail

# The 80 array tasks cover canonical chunks 0259 through 0180. Array task 0
# starts at the end of chromosome 1, and later task IDs move backward.
last_chunk=259
first_chunk=180
total_rows=2591838
chunk_size=10000

task_id=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is not set}
chunk=$((last_chunk - task_id))
if (( chunk < first_chunk || chunk > last_chunk )); then
  echo "Array task $task_id maps to invalid chunk $chunk" >&2
  exit 1
fi

root=${ALPHAGENOME_DELTA_ROOT:-$HOME/worknvme_agenome}
repo=$root/alphagenome-pytorch
python_bin=${ALPHAGENOME_PYTHON:-/work/nvme/bgup/lxu13/doggenetics/alphagenome/envs/alphagenome/bin/python}

input_dir=$root/inference_needed_data
checkpoint=$input_dir/best_heads.pt
pretrained_weights=$input_dir/model_fold_1.safetensors
vcf=$input_dir/dog10k_chr1_512kb_exon_nearest_gene.sites.vcf.gz
variant_gene_map=$input_dir/dog10k_chr1_512kb_exon_nearest_gene.variant_gene_map.tsv.gz
gtf=$input_dir/GCF_011100685.1_UU_Cfam_GSD_1.0_genomic.gtf.gz
fasta=$input_dir/GCF_011100685.1_UU_Cfam_GSD_1.0_genomic.fna
inference_script=$repo/scripts/inference/inference_geneVar.py
validator=$repo/scripts/inference/validate_genevar_shard.py

output_dir=$root/results/inference/chr1_512kb/chunks_10k
app_log_dir=$root/logs/chr1_512kb_reverse/app

start=$((chunk * chunk_size))
end=$((start + chunk_size))
if (( end > total_rows )); then
  end=$total_rows
fi

printf -v chunk_id '%04d' "$chunk"
printf -v start_id '%07d' "$start"
printf -v end_id '%07d' "$end"
stem=chunk_${chunk_id}_rows_${start_id}_${end_id}
output=$output_dir/${stem}.h5
partial=${output}.partial
app_log=$app_log_dir/${stem}.log

mkdir -p "$output_dir" "$app_log_dir"

for required_file in \
  "$python_bin" "$checkpoint" "$pretrained_weights" \
  "$vcf" "$variant_gene_map" "$gtf" "$fasta" "${fasta}.fai" \
  "$inference_script" "$validator"; do
  if [[ ! -f "$required_file" ]]; then
    echo "Required file not found: $required_file" >&2
    exit 1
  fi
done

export PYTHONPATH=$repo/src${PYTHONPATH:+:$PYTHONPATH}

echo "Host: $(hostname)"
echo "Slurm job: ${SLURM_JOB_ID}; array task: $task_id"
echo "Reverse chunk: $chunk_id; rows [$start, $end)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not-set}"
"$python_bin" -c \
  "import torch; print('CUDA available:', torch.cuda.is_available()); print('visible GPUs:', torch.cuda.device_count()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

if [[ -f "$output" ]]; then
  if "$python_bin" "$validator" "$output" \
    --expected-start "$start" --expected-end "$end" --quiet; then
    echo "Valid output already exists; nothing to do: $output"
    exit 0
  fi
  echo "Existing output is invalid: $output" >&2
  exit 1
fi

if [[ -e "$partial" ]]; then
  stale=${partial}.stale.$(date -u +%Y%m%dT%H%M%SZ)
  echo "Moving stale partial output to $stale"
  mv "$partial" "$stale"
fi

PYTHONFAULTHANDLER=1 \
PYTHONUNBUFFERED=1 \
"$python_bin" -u "$inference_script" \
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
  2>&1 | tee "$app_log"

"$python_bin" "$validator" "$partial" \
  --expected-start "$start" --expected-end "$end"
mv "$partial" "$output"
echo "Completed reverse chunk $chunk_id: $output"
