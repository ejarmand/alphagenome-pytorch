#!/usr/bin/env python
"""Evaluate AlphaGenome TFRecord fine-tuned heads in hound_eval.py style."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm

from alphagenome_pytorch import AlphaGenome
from alphagenome_pytorch.config import DtypePolicy
from alphagenome_pytorch.extensions.finetuning.heads import create_finetuning_head
from alphagenome_pytorch.extensions.finetuning.tfrecord_dataset import (
    BaskervilleMultiTFRecordDataset,
    BaskervilleTFRecordDataset,
)
from alphagenome_pytorch.extensions.finetuning.transfer import load_trunk, remove_all_heads

try:
    from scripts.finetune_tfr_heads import (
        TFRHeads,
        cell_types_from_target_rows,
        cleanup_torchrun,
        compute_loss,
        create_loader,
        create_tfr_dataset,
        forward_heads,
        is_main_process,
        setup_torchrun,
        unpack_tfr_batch,
    )
except ModuleNotFoundError:
    from finetune_tfr_heads import (  # type: ignore[no-redef]
        TFRHeads,
        cell_types_from_target_rows,
        cleanup_torchrun,
        compute_loss,
        create_loader,
        create_tfr_dataset,
        forward_heads,
        is_main_process,
        setup_torchrun,
        unpack_tfr_batch,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate TFRecord fine-tuned AlphaGenome heads.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Checkpoint file, or run directory containing best_heads.pt/last_heads.pt.",
    )
    parser.add_argument(
        "--checkpoint-name",
        default="best_heads.pt",
        help="Checkpoint filename to use when checkpoint is a directory.",
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--pretrained-weights", type=Path, default=None)
    parser.add_argument(
        "--modality",
        action="append",
        default=None,
        help="Modality to evaluate. Repeat to evaluate a subset. Defaults to checkpoint metadata.",
    )
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--pooling", choices=["mean", "sum"], default=None)
    parser.add_argument("--target-resolution", type=int, choices=[32, 128], default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-n", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--tfr-num-parallel-reads", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=Path("eval_out"))
    parser.add_argument("--label", default="Test", help="Summary label printed before metrics.")
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save concatenated predictions and targets as HDF5 files.",
    )
    parser.add_argument(
        "--save-dtype",
        choices=["float16", "float32"],
        default="float16",
        help="Dtype for saved HDF5 prediction/target arrays.",
    )
    parser.add_argument(
        "--loss",
        choices=["poisson-multinomial", "multinomial", "poisson", "mse"],
        default=None,
    )
    parser.add_argument("--positional-weight", type=float, default=5.0)
    parser.add_argument("--count-weight", type=float, default=1.0)
    parser.add_argument("--crop-bins", type=int, default=None)
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', 'cuda', or e.g. 'cuda:0'")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="Disable progress bars.")
    return parser.parse_args()


def resolve_checkpoint(path: Path, checkpoint_name: str) -> Path:
    if path.is_dir():
        path = path / checkpoint_name
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def load_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Expected checkpoint dict in {path}")
    return checkpoint


def load_metadata(checkpoint: dict[str, Any], checkpoint_path: Path) -> dict[str, Any]:
    metadata = checkpoint.get("metadata")
    if isinstance(metadata, dict):
        return metadata
    metadata_path = checkpoint_path.parent / "metadata.json"
    if metadata_path.exists():
        with metadata_path.open() as metadata_file:
            return json.load(metadata_file)
    return {}


def resolve_path(path: Path | None, metadata_value: str | None, checkpoint_path: Path) -> Path | None:
    if path is None and metadata_value:
        path = Path(metadata_value)
    if path is None:
        return None
    if path.exists() or path.is_absolute():
        return path
    candidates = [
        Path.cwd() / path,
        checkpoint_path.parent / path,
        checkpoint_path.parent.parent / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


def resolve_modalities(
    args: argparse.Namespace,
    metadata: dict[str, Any],
    data_dir: Path,
) -> list[str]:
    if args.modality:
        return args.modality
    metadata_modalities = metadata.get("modalities")
    if isinstance(metadata_modalities, list) and metadata_modalities:
        return [str(modality) for modality in metadata_modalities]
    return BaskervilleTFRecordDataset.available_modalities(data_dir)


def create_heads_model(
    dataset: BaskervilleMultiTFRecordDataset,
    modalities: list[str],
    metadata: dict[str, Any],
    checkpoint: dict[str, Any],
    device: torch.device,
    target_resolution: int,
) -> TFRHeads:
    heads = {}
    for modality in modalities:
        heads[modality] = create_finetuning_head(
            assay_type=dataset.assay_type_by_modality[modality],
            n_tracks=len(dataset.target_indices_by_modality[modality]),
            resolutions=(128,),
            num_organisms=1,
        )

    metadata_cell_types = metadata.get("cell_types")
    if isinstance(metadata_cell_types, dict):
        cell_types_by_modality = {
            modality: list(metadata_cell_types[modality])
            for modality in modalities
            if modality in metadata_cell_types
        }
    else:
        cell_types_by_modality = {}
    missing_modalities = [
        modality for modality in modalities if modality not in cell_types_by_modality
    ]
    if missing_modalities:
        reconstructed = cell_types_from_target_rows({
            modality: dataset.target_rows_by_modality[modality]
            for modality in missing_modalities
        })
        cell_types_by_modality.update(reconstructed)

    heads_model = TFRHeads(
        heads,
        cell_types_by_modality,
        cell_embedding_dim=int(metadata.get("cell_embedding_dim", 16)),
        target_resolution=target_resolution,
    ).to(device)

    state_dict = (
        checkpoint.get("heads_model_state_dict")
        or checkpoint.get("heads_state_dict")
        or checkpoint.get("state_dict")
    )
    if state_dict is None:
        raise KeyError(
            "Checkpoint does not contain heads_model_state_dict, heads_state_dict, or state_dict"
        )
    missing, unexpected = heads_model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: missing head keys: {missing[:5]}...")
    if unexpected:
        print(f"Warning: unexpected head keys: {unexpected[:5]}...")
    heads_model.eval()
    return heads_model


def new_target_stats(n_tracks: int, device: torch.device) -> torch.Tensor:
    # columns: n, sum_pred, sum_true, sum_pred2, sum_true2, sum_pred_true, sum_squared_error
    return torch.zeros(n_tracks, 7, dtype=torch.float64, device=device)


@torch.no_grad()
def update_target_stats(stats: torch.Tensor, pred: torch.Tensor, target: torch.Tensor) -> None:
    pred = pred.detach().double()
    target = target.detach().double()
    diff = pred - target

    stats[:, 0] += pred.shape[0] * pred.shape[1]
    stats[:, 1] += pred.sum(dim=(0, 1))
    stats[:, 2] += target.sum(dim=(0, 1))
    stats[:, 3] += pred.square().sum(dim=(0, 1))
    stats[:, 4] += target.square().sum(dim=(0, 1))
    stats[:, 5] += (pred * target).sum(dim=(0, 1))
    stats[:, 6] += diff.square().sum(dim=(0, 1))


def compute_target_metrics(stats: torch.Tensor) -> tuple[list[float], list[float]]:
    n = stats[:, 0].clamp_min(1.0)
    pred_var_sum = stats[:, 3] - stats[:, 1].square() / n
    true_var_sum = stats[:, 4] - stats[:, 2].square() / n
    covariance_sum = stats[:, 5] - stats[:, 1] * stats[:, 2] / n

    denominator = torch.sqrt(pred_var_sum.clamp_min(0.0) * true_var_sum.clamp_min(0.0))
    pearson = covariance_sum / denominator
    pearson = torch.where(denominator > 0, pearson, torch.full_like(pearson, float("nan")))

    r2 = 1.0 - stats[:, 6] / true_var_sum
    r2 = torch.where(true_var_sum > 0, r2, torch.full_like(r2, float("nan")))
    return pearson.cpu().tolist(), r2.cpu().tolist()


def nanmean(values: list[float]) -> float:
    finite = [value for value in values if not math.isnan(value)]
    if not finite:
        return float("nan")
    return sum(finite) / len(finite)


class H5PredictionWriter:
    """Append predictions/targets to hound_eval.py-style HDF5 files."""

    def __init__(
        self,
        out_dir: Path,
        modalities: list[str],
        n_tracks_by_modality: dict[str, int],
        dtype: str,
        rank: int,
        world_size: int,
    ):
        import h5py

        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        suffix = f".rank{rank}" if world_size > 1 else ""
        self.preds_h5 = h5py.File(out_dir / f"preds{suffix}.h5", "w")
        self.targets_h5 = h5py.File(out_dir / f"targets{suffix}.h5", "w")
        self.dtype = dtype
        self.preds = None
        self.targets = None
        self.count = 0
        self.modalities = modalities
        self.n_tracks_by_modality = n_tracks_by_modality

    def append(
        self,
        predictions_by_modality: dict[str, torch.Tensor],
        targets_by_modality: dict[str, torch.Tensor],
    ) -> None:
        pred = torch.cat(
            [predictions_by_modality[modality] for modality in self.modalities],
            dim=-1,
        )
        target = torch.cat(
            [targets_by_modality[modality] for modality in self.modalities],
            dim=-1,
        )
        pred_np = pred.detach().to("cpu", dtype=torch.float32).numpy().astype(self.dtype)
        target_np = target.detach().to("cpu", dtype=torch.float32).numpy().astype(self.dtype)

        if self.preds is None or self.targets is None:
            maxshape = (None, pred_np.shape[1], pred_np.shape[2])
            chunks = (1, pred_np.shape[1], pred_np.shape[2])
            self.preds = self.preds_h5.create_dataset(
                "preds",
                shape=(0, pred_np.shape[1], pred_np.shape[2]),
                maxshape=maxshape,
                chunks=chunks,
                dtype=self.dtype,
            )
            self.targets = self.targets_h5.create_dataset(
                "targets",
                shape=(0, target_np.shape[1], target_np.shape[2]),
                maxshape=maxshape,
                chunks=chunks,
                dtype=self.dtype,
            )
            metadata = json.dumps(
                {
                    "modalities": self.modalities,
                    "n_tracks_by_modality": self.n_tracks_by_modality,
                }
            )
            self.preds_h5.attrs["target_order"] = metadata
            self.targets_h5.attrs["target_order"] = metadata

        next_count = self.count + pred_np.shape[0]
        self.preds.resize(next_count, axis=0)
        self.targets.resize(next_count, axis=0)
        self.preds[self.count:next_count] = pred_np
        self.targets[self.count:next_count] = target_np
        self.count = next_count

    def close(self) -> None:
        self.preds_h5.close()
        self.targets_h5.close()


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    heads_model: TFRHeads,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    crop_bins: int,
    target_resolution: int,
    rank: int,
    world_size: int,
    h5_writer: H5PredictionWriter | None = None,
) -> tuple[float, dict[str, dict[str, Any]], int]:
    model.eval()
    heads_model.eval()

    total_loss = torch.tensor(0.0, dtype=torch.float64, device=device)
    total_steps = torch.tensor(0.0, dtype=torch.float64, device=device)
    total_examples = torch.tensor(0.0, dtype=torch.float64, device=device)
    stats_by_modality = {
        modality: new_target_stats(len(indices), device)
        for modality, indices in loader.dataset.target_indices_by_modality.items()
    }
    strand_pair_by_modality = {
        modality: torch.as_tensor(indices, dtype=torch.long, device=device)
        for modality, indices in loader.dataset.strand_pair_by_modality.items()
    }
    eval_args = SimpleNamespace(
        loss=args.loss,
        no_amp=args.no_amp,
        positional_weight=args.positional_weight,
        count_weight=args.count_weight,
    )
    heads = heads_model.heads
    pbar = tqdm(
        total=args.max_steps,
        desc=args.split,
        disable=args.quiet or not is_main_process(rank),
    )
    data_iter = iter(loader)
    steps = 0
    while args.max_steps is None or steps < args.max_steps:
        try:
            batch = next(data_iter)
            sequences, modality_targets, augmentation = unpack_tfr_batch(batch)
            has_batch = torch.tensor(1, device=device)
        except StopIteration:
            sequences = None
            modality_targets = None
            augmentation = None
            has_batch = torch.tensor(0, device=device)

        if world_size > 1:
            dist.all_reduce(has_batch, op=dist.ReduceOp.MIN)
        if has_batch.item() == 0:
            break
        if sequences is None or modality_targets is None:
            raise RuntimeError("Local loader is exhausted but distributed batch check passed")

        sequences = sequences.to(device, non_blocking=True)
        organism_idx = torch.zeros(sequences.shape[0], dtype=torch.long, device=device)
        reverse_complement = None
        if augmentation is not None:
            reverse_complement = augmentation["reverse_complement"].to(
                device,
                non_blocking=True,
            )

        loss_predictions, metric_predictions = forward_heads(
            model,
            heads_model,
            sequences,
            organism_idx,
            crop_bins,
            use_amp=not args.no_amp,
            return_scaled=args.loss in ("poisson-multinomial", "multinomial"),
            requires_grad=False,
            reverse_complement=reverse_complement,
            strand_pair_by_modality=strand_pair_by_modality,
        )

        batch_loss = torch.tensor(0.0, device=device)
        cropped_targets_by_modality = {}
        for modality, pred in loss_predictions.items():
            target = modality_targets[modality][target_resolution].to(device, non_blocking=True)
            if pred.shape != target.shape:
                raise ValueError(
                    f"{modality}: prediction shape {tuple(pred.shape)} does not match "
                    f"target shape {tuple(target.shape)} after crop_bins={crop_bins}"
                )
            batch_loss = batch_loss + compute_loss(
                pred,
                target,
                loss_name=args.loss,
                head=heads[modality],
                organism_idx=organism_idx,
                positional_weight=eval_args.positional_weight,
                count_weight=eval_args.count_weight,
            )
            update_target_stats(stats_by_modality[modality], metric_predictions[modality], target)
            cropped_targets_by_modality[modality] = target

        if h5_writer is not None:
            h5_writer.append(metric_predictions, cropped_targets_by_modality)

        total_loss += batch_loss.detach().double()
        total_steps += 1
        total_examples += sequences.shape[0]
        steps += 1
        pbar.set_postfix({"loss": f"{batch_loss.item():.4f}"})
        pbar.update(1)
    pbar.close()

    if world_size > 1:
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_steps, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_examples, op=dist.ReduceOp.SUM)
        for stats in stats_by_modality.values():
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)

    metrics_by_modality = {}
    for modality, stats in stats_by_modality.items():
        pearsonr, r2 = compute_target_metrics(stats)
        metrics_by_modality[modality] = {
            "pearsonr": pearsonr,
            "r2": r2,
            "mean_pearsonr": nanmean(pearsonr),
            "mean_r2": nanmean(r2),
        }

    return (
        (total_loss / total_steps.clamp_min(1)).item(),
        metrics_by_modality,
        int(total_examples.item()),
    )


def target_rows(
    dataset: BaskervilleMultiTFRecordDataset,
    metrics_by_modality: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for modality, metrics in metrics_by_modality.items():
        target_rows_for_modality = dataset.target_rows_by_modality[modality]
        for target_i, row in enumerate(target_rows_for_modality):
            rows.append({
                "index": row.get("index", target_i),
                "modality": modality,
                "pearsonr": metrics["pearsonr"][target_i],
                "r2": metrics["r2"][target_i],
                "identifier": row.get("identifier", ""),
                "description": row.get("description", ""),
            })
    return rows


def write_acc(path: Path, rows: list[dict[str, Any]], include_modality: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["index", "pearsonr", "r2", "identifier", "description"]
    if include_modality:
        fieldnames.insert(1, "modality")
    with path.open("w", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
            delimiter="\t",
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            formatted = dict(row)
            formatted["pearsonr"] = f"{row['pearsonr']:.5f}"
            formatted["r2"] = f"{row['r2']:.5f}"
            writer.writerow(formatted)


def write_outputs(
    out_dir: Path,
    dataset: BaskervilleMultiTFRecordDataset,
    metrics_by_modality: dict[str, dict[str, Any]],
) -> None:
    rows = target_rows(dataset, metrics_by_modality)
    include_modality = len(metrics_by_modality) > 1
    write_acc(out_dir / "acc.txt", rows, include_modality=include_modality)
    if include_modality:
        for modality in metrics_by_modality:
            modality_rows = [row for row in rows if row["modality"] == modality]
            write_acc(out_dir / modality / "acc.txt", modality_rows, include_modality=False)


def print_summary(
    label: str,
    loss: float,
    metrics_by_modality: dict[str, dict[str, Any]],
    examples: int,
) -> None:
    all_pearsonr = [
        value
        for metrics in metrics_by_modality.values()
        for value in metrics["pearsonr"]
    ]
    all_r2 = [
        value
        for metrics in metrics_by_modality.values()
        for value in metrics["r2"]
    ]
    print("\n%s Loss:         %7.5f" % (label, loss))
    print("%s PearsonR:     %7.5f" % (label, nanmean(all_pearsonr)))
    print("%s R2:           %7.5f" % (label, nanmean(all_r2)))
    print("%s Examples:     %7d" % (label, examples))
    if len(metrics_by_modality) > 1:
        print("\nModality PearsonR R2")
        for modality, metrics in metrics_by_modality.items():
            print(
                "%-16s %7.5f %7.5f"
                % (modality, metrics["mean_pearsonr"], metrics["mean_r2"])
            )


def main() -> None:
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    checkpoint_path = resolve_checkpoint(args.checkpoint, args.checkpoint_name)
    checkpoint = load_checkpoint(checkpoint_path)
    metadata = load_metadata(checkpoint, checkpoint_path)

    data_dir = resolve_path(args.data_dir, metadata.get("data_dir"), checkpoint_path)
    if data_dir is None:
        raise SystemExit("--data-dir is required when checkpoint metadata omits data_dir")
    pretrained_weights = resolve_path(
        args.pretrained_weights,
        metadata.get("pretrained_weights"),
        checkpoint_path,
    )
    if pretrained_weights is None:
        raise SystemExit(
            "--pretrained-weights is required when checkpoint metadata omits pretrained_weights"
        )

    if args.pooling is None:
        args.pooling = metadata.get("pooling", "mean")
    if args.target_resolution is None:
        args.target_resolution = int(metadata.get("target_resolution", 128))
    if args.loss is None:
        args.loss = metadata.get("loss", "poisson-multinomial")
    if args.batch_size is None:
        args.batch_size = int(metadata.get("batch_size", 1))
    if args.dtype is None:
        args.dtype = metadata.get("dtype", "bfloat16")
    if not args.no_amp and metadata.get("use_amp") is False:
        args.no_amp = True

    rank, world_size, local_rank, device = setup_torchrun(args.device)
    try:
        modalities = resolve_modalities(args, metadata, data_dir)
        dataset = create_tfr_dataset(
            data_dir,
            args.split,
            modalities,
            args.pooling,
            args.target_resolution,
            repeat=False,
            shuffle_files=False,
            num_parallel_reads=args.tfr_num_parallel_reads,
            rank=rank,
            world_size=world_size,
        )
        loader = create_loader(
            dataset,
            args.batch_size,
            args.num_workers,
            prefetch_n=args.prefetch_n,
            persistent_workers=False,
        )
        crop_bins = (
            args.crop_bins
            if args.crop_bins is not None
            else int(metadata.get("crop_bins", dataset.prediction_crop_bins))
        )

        if is_main_process(rank):
            print(
                " ".join(
                    (
                        f"Checkpoint: {checkpoint_path}",
                        f"split={args.split}",
                        f"device={device}",
                        f"local_rank={local_rank}",
                    )
                )
            )
            print(
                " ".join(
                    (
                        f"Data: {data_dir}",
                        f"modalities={','.join(modalities)}",
                        f"batch_size={args.batch_size}",
                        f"workers={loader.num_workers}",
                        f"crop_bins={crop_bins}",
                        f"target_resolution={args.target_resolution}",
                    )
                )
            )

        dtype_policy = (
            DtypePolicy.full_float32()
            if args.dtype == "float32"
            else DtypePolicy.mixed_precision()
        )
        model = AlphaGenome(dtype_policy=dtype_policy)
        model = load_trunk(model, str(pretrained_weights), exclude_heads=True)
        model = remove_all_heads(model).to(device)
        for param in model.parameters():
            param.requires_grad = False

        heads_model = create_heads_model(
            dataset,
            modalities,
            metadata,
            checkpoint,
            device,
            args.target_resolution,
        )
        h5_writer = None
        if args.save:
            h5_writer = H5PredictionWriter(
                args.out_dir,
                modalities,
                {
                    modality: len(dataset.target_indices_by_modality[modality])
                    for modality in modalities
                },
                args.save_dtype,
                rank,
                world_size,
            )
        try:
            loss, metrics_by_modality, examples = evaluate(
                model,
                heads_model,
                loader,
                device,
                args,
                crop_bins,
                args.target_resolution,
                rank,
                world_size,
                h5_writer=h5_writer,
            )
        finally:
            if h5_writer is not None:
                h5_writer.close()

        if is_main_process(rank):
            print_summary(args.label, loss, metrics_by_modality, examples)
            write_outputs(args.out_dir, dataset, metrics_by_modality)
            print(f"\nWrote target statistics to {args.out_dir / 'acc.txt'}")
            if args.save:
                if world_size > 1:
                    print(
                        f"Wrote rank-sharded predictions to {args.out_dir / 'preds.rank*.h5'} "
                        f"and {args.out_dir / 'targets.rank*.h5'}"
                    )
                else:
                    print(
                        f"Wrote predictions to {args.out_dir / 'preds.h5'} "
                        f"and targets to {args.out_dir / 'targets.h5'}"
                    )
    finally:
        cleanup_torchrun()


if __name__ == "__main__":
    main()
