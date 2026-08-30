# Chromosome 1 window-aware inference

This workflow performs variant-level inference. It does not perform one model
run per dog.

The preparation script applies these rules to every source chromosome 1
record:

1. Keep biallelic SNVs.
2. Construct the exact 524,288 bp window used by `inference_geneVar.py`.
3. Keep a variant if at least one exon from a protein-coding gene overlaps the
   window.
4. Assign one qualifying gene with the smallest distance from the variant to
   the gene body.
5. Resolve equal gene-body distances by distance to the transcription start
   site, then by `gene_id`.

The prepared VCF contains no sample columns. Keep the original 1,987-sample
Dog10K VCF for later joins between variant effects and dog genotypes.

## Prepare the inputs

From `/home/lilx/projects/doggenetics/alphagenome_rna`:

```bash
bash alphagenome-pytorch-tfr/scripts/bash_launch/prepare_chr1_512kb_nearest_gene.sh
```

This writes the sites-only VCF, variant-gene map, and metadata under:

```text
/home/datasets/deep_learning/application/lilx_dog/inference_chr1_512kb
```

## Run the two-GPU smoke test

```bash
bash alphagenome-pytorch-tfr/scripts/bash_launch/run_chr1_512kb_genevar_2gpu.sh smoke
```

Both logs must end with a completed progress bar and no recorded errors.

## Measure a 100-variant pilot

```bash
bash alphagenome-pytorch-tfr/scripts/bash_launch/run_chr1_512kb_genevar_2gpu.sh pilot
```

The prepared catalog contains 2,591,838 variants. The previous inference rate
was about 2.22 seconds per variant on each GPU. At that rate, the full job would
take roughly 33 days on two GPUs. Use the pilot logs to measure the current rate
before launching the full catalog. Batching or distributing more shards is the
better next step if the pilot confirms this estimate.

## Run full inference after reviewing the pilot

Start it in tmux so it survives a disconnected terminal:

```bash
tmux new -s chr1_512kb_inference
cd /home/lilx/projects/doggenetics/alphagenome_rna
ALLOW_LONG_FULL_RUN=1 \
  bash alphagenome-pytorch-tfr/scripts/bash_launch/run_chr1_512kb_genevar_2gpu.sh full
```

Detach with `Ctrl-b`, then `d`. Reattach with:

```bash
tmux attach -t chr1_512kb_inference
```

Watch GPU use in another terminal:

```bash
watch -n 1 nvidia-smi
```

The launcher stores signed log fold changes, so positive values mean the ALT
allele raised the predicted signal and negative values mean it lowered the
predicted signal. The final merged file is:

```text
results/inference/chr1_512kb/chr1_512kb_nearest_gene_signed.h5
```

## Resumable 10,000-variant shards

Use the chunked launcher for the full catalog. It assigns alternating chunks
to the two GPUs and writes each completed chunk independently:

```bash
bash alphagenome-pytorch-tfr/scripts/bash_launch/run_chr1_512kb_genevar_10k_2gpu.sh
```

Restarting the command validates and skips completed chunks. Check aggregate
progress with:

```bash
pixi run python \
  alphagenome-pytorch-tfr/scripts/inference/genevar_shard_status.py \
  --manifest /home/datasets/deep_learning/application/lilx_dog/inference_chr1_512kb/chr1_512kb_genevar_10k_shards.tsv
```

After all 260 chunks finish, merge them with:

```bash
bash alphagenome-pytorch-tfr/scripts/bash_launch/merge_chr1_512kb_genevar_10k.sh
```

## Delta reverse array

The Delta Slurm array launcher processes tail chunks 0259 through 0180 while
Cuica continues from the beginning. Array task 0 maps to chunk 0259 and later
tasks move backward. At most two one-GPU tasks run concurrently:

```bash
mkdir -p ~/worknvme_agenome/inference_needed_data
mkdir -p ~/worknvme_agenome/logs/chr1_512kb_reverse
cd ~/worknvme_agenome
sbatch alphagenome-pytorch/scripts/bash_launch/run_chr1_512kb_genevar_delta_reverse.slurm
```

The launcher reads all transferred model and reference inputs from:

```text
~/worknvme_agenome/inference_needed_data
```

From Cuica, create that directory and transfer the inputs with:

```bash
DELTA_LOGIN=lxu13@login.delta.ncsa.illinois.edu

ssh "$DELTA_LOGIN" \
  'mkdir -p "$HOME/worknvme_agenome/inference_needed_data"'

rsync -avP --partial \
  /home/lilx/projects/doggenetics/alphagenome_rna/finetuning_output/tfr_multitask_v4_final/model_final_datasetv4_200ep/best_heads.pt \
  /home/lilx/projects/doggenetics/alphagenome_rna/weights/model_fold_1.safetensors \
  /home/datasets/deep_learning/application/lilx_dog/inference_chr1_512kb/dog10k_chr1_512kb_exon_nearest_gene.sites.vcf.gz \
  /home/datasets/deep_learning/application/lilx_dog/inference_chr1_512kb/dog10k_chr1_512kb_exon_nearest_gene.variant_gene_map.tsv.gz \
  /home/datasets/deep_learning/application/lilx_dog/ref/UU_Cfam_GSD_1.0/GCF_011100685.1_UU_Cfam_GSD_1.0_genomic.gtf.gz \
  /home/lilx/projects/doggenetics/ref/genomes/ncbi_dataset/ncbi_dataset/data/GCF_011100685.1/GCF_011100685.1_UU_Cfam_GSD_1.0_genomic.fna \
  /home/lilx/projects/doggenetics/ref/genomes/ncbi_dataset/ncbi_dataset/data/GCF_011100685.1/GCF_011100685.1_UU_Cfam_GSD_1.0_genomic.fna.fai \
  "$DELTA_LOGIN:worknvme_agenome/inference_needed_data/"

ssh "$DELTA_LOGIN" \
  'ls -lh "$HOME/worknvme_agenome/inference_needed_data"'
```

That directory must contain:

```text
best_heads.pt
model_fold_1.safetensors
dog10k_chr1_512kb_exon_nearest_gene.sites.vcf.gz
dog10k_chr1_512kb_exon_nearest_gene.variant_gene_map.tsv.gz
GCF_011100685.1_UU_Cfam_GSD_1.0_genomic.gtf.gz
GCF_011100685.1_UU_Cfam_GSD_1.0_genomic.fna
GCF_011100685.1_UU_Cfam_GSD_1.0_genomic.fna.fai
```

Do not merge on Delta. Copy the validated tail `chunk_*.h5` files into Cuica's
canonical `results/inference/chr1_512kb/chunks_10k` directory, then use the
normal status and merge commands on Cuica.
