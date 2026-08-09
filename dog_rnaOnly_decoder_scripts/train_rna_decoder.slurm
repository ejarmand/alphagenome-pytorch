#!/bin/bash
#SBATCH --job-name=ag_rna_decoder
#SBATCH --account=bgup-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --output=logs/ag_rna_decoder_%j.out
#SBATCH --error=logs/ag_rna_decoder_%j.err
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00

cd ~/worknvme_agenome

mkdir -p logs
mkdir -p finetuning_output/rna_only_512bp_bs64_lr2e-4_h2048_do0.05_log1p_pmheads_v2

export WANDB_PROJECT=rna_alphagenome_dog10k
export WANDB_ENTITY=ejarmand
export WANDB_MODE=online

PYTHON=/work/nvme/bgup/lxu13/doggenetics/alphagenome/envs/alphagenome/bin/python

$PYTHON -c "import sys, numpy, torch; print(sys.executable); print('numpy ok'); print('torch cuda:', torch.cuda.is_available()); print('gpu count:', torch.cuda.device_count())"
$PYTHON -c "import wandb; print('wandb ok:', wandb.__version__)"

$PYTHON -u scripts/train_rna_decoder_from_embeddings.py \
  --targets data/DatasetV2_atac_rna/targets.txt \
  --train-embeddings embeddings/train \
  --train-labels labels/train \
  --valid-embeddings embeddings/valid \
  --valid-labels labels/valid \
  --out finetuning_output/rna_only_512bp_bs64_lr2e-4_h2048_do0.05_log1p_pmheads_v2 \
  --batch-size 64 \
  --epochs 60 \
  --lr 2e-4 \
  --min-lr 1e-5 \
  --weight-decay 5e-5 \
  --warmup-epochs 3 \
  --scheduler cosine \
  --hidden-dim 2048 \
  --dropout 0.05 \
  --num-workers 2 \
  --rna-label-transform log1p \
  --pool-bp 512 \
  --wandb \
  --wandb-project rna_alphagenome_dog10k \
  --wandb-entity ejarmand \
  --wandb-run-name rna_only_512bp_bs64_lr2e-4_h2048_do0.05_log1p_pmheads_v2
