#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <run_dir_or_checkpoint.pt> [test_tfr_genes.py args...]" >&2
  exit 2
fi

model_path=$1
shift

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "${script_dir}/.." && pwd)
project_dir=$(cd -- "${repo_dir}/.." && pwd)
if [[ "${model_path}" != /* ]]; then
  model_path="$(pwd)/${model_path}"
fi

if [[ -d "${model_path}" ]]; then
  model_label=$(basename -- "${model_path}")
else
  model_label=$(basename -- "$(dirname -- "${model_path}")")
fi

data_dir=${DATA_DIR:-"${project_dir}/tfr_data"}
gtf=${GTF:-"/data/earmand/projects/amb_paired_tag/baskerville_dnn/data/ref/gtf/gencode.vM25.annotation.gtf"}
out_dir=${OUT_DIR:-"${project_dir}/eval/${model_label}_test_genes_rna"}
split=${SPLIT:-test}
modality_csv=${MODALITIES:-RNA}
checkpoint_name=${CHECKPOINT_NAME:-best_heads.pt}

cmd=(
  pixi run -e alphagenome-test python
  alphagenome-pytorch/scripts/test_tfr_genes.py
  "${model_path}"
  "${gtf}"
  --checkpoint-name "${checkpoint_name}"
  --data-dir "${data_dir}"
  --split "${split}"
  -o "${out_dir}"
)

if [[ -n "${PRETRAINED_WEIGHTS:-}" ]]; then
  cmd+=(--pretrained-weights "${PRETRAINED_WEIGHTS}")
fi

if [[ -n "${modality_csv}" ]]; then
  IFS=',' read -r -a modalities <<< "${modality_csv}"
  for modality in "${modalities[@]}"; do
    cmd+=(--modality "${modality}")
  done
fi

cmd+=("$@")

cd "${project_dir}"
printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'
exec "${cmd[@]}"
