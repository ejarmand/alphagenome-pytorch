#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TFR_REPO_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_DIR="$(cd -- "${TFR_REPO_DIR}/.." && pwd)"

cd "${PROJECT_DIR}"

RUN_NAME="${RUN_NAME:-model_final_datasetv4_200ep}"
EMBEDDING_CACHE="${ALPHAGENOME_EMBEDDING_CACHE:-/home/datasets/deep_learning/application/lilx_dog/alphagenome_cache/dog_datasetv4_fold1_embeddings}"
OUTPUT_ROOT="finetuning_output/tfr_multitask_v4_final"
LOG_PATH="logs/${RUN_NAME}.log"

mkdir -p logs "${OUTPUT_ROOT}"

if [[ ! -d "${EMBEDDING_CACHE}" ]]; then
  echo "Embedding cache not found: ${EMBEDDING_CACHE}" >&2
  exit 1
fi

if [[ -e "${OUTPUT_ROOT}/${RUN_NAME}" ]]; then
  echo "Output directory already exists: ${OUTPUT_ROOT}/${RUN_NAME}" >&2
  echo "Set RUN_NAME to a new value before launching another run." >&2
  exit 1
fi

echo "Run name: ${RUN_NAME}"
echo "Embedding cache: ${EMBEDDING_CACHE}"
echo "Output directory: ${OUTPUT_ROOT}/${RUN_NAME}"
echo "Log: ${LOG_PATH}"

CUDA_VISIBLE_DEVICES=0,1 \
PYTHONFAULTHANDLER=1 \
PYTHONUNBUFFERED=1 \
pixi run torchrun \
  --standalone \
  --nproc_per_node=2 \
  alphagenome-pytorch-tfr/scripts/finetune_tfr_heads.py \
  --data-dir data/DatasetV4_rna_atac_norm150M_meansqrt \
  --pretrained-weights weights/model_fold_1.safetensors \
  --modality rna \
  --modality atac \
  --target-resolution 128 \
  --pooling mean \
  --organism-idx 0 \
  --precompute-embeddings \
  --embedding-cache-dir "${EMBEDDING_CACHE}" \
  --embedding-cache-dtype bfloat16 \
  --embedding-cache-chunk-size 16 \
  --augment-rc \
  --cell-embedding-dim 98 \
  --track-means-samples 2787 \
  --loss poisson-multinomial \
  --positional-weight 149.092773783687 \
  --count-weight 1 \
  --epochs 200 \
  --batch-size 64 \
  --gradient-accumulation-steps 2 \
  --lr 0.005501321034060158 \
  --warmup-fraction 0.05 \
  --lr-schedule constant \
  --weight-decay 0.1326315162236689 \
  --seed 20260815 \
  --num-workers 2 \
  --val-num-workers 0 \
  --device cuda \
  --dtype bfloat16 \
  --output-dir "${OUTPUT_ROOT}" \
  --run-name "${RUN_NAME}" \
  --wandb \
  --wandb-project rna_alphagenome_dog10k \
  --wandb-entity ejarmand \
  --wandb-run-name "${RUN_NAME}" \
  --wandb-tags datasetv4,final-training,200ep,ddp2,seed-20260815 \
  2>&1 | tee "${LOG_PATH}"
