#!/usr/bin/env python
"""Fine-tune frozen AlphaGenome 128 bp heads from Baskerville TFRecords."""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from alphagenome_pytorch import AlphaGenome
from alphagenome_pytorch.config import DtypePolicy
from alphagenome_pytorch.extensions.finetuning.heads import create_finetuning_head
from alphagenome_pytorch.extensions.finetuning.tfrecord_dataset import (
    BaskervilleTFRecordDataset,
    collate_tfr_genomic,
)
from alphagenome_pytorch.extensions.finetuning.transfer import load_trunk, remove_all_heads
from alphagenome_pytorch.losses import multinomial_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a new 128 bp AlphaGenome head from Baskerville TFRecords.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--pretrained-weights", type=Path)
    parser.add_argument("--modality", default="RNA")
    parser.add_argument("--pooling", choices=["mean", "sum"], default="mean")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--loss",
        choices=["poisson-multinomial", "multinomial", "poisson", "mse"],
        default="poisson-multinomial",
        help=(
            "Training loss. 'poisson-multinomial' is the AlphaGenome-style "
            "combined Poisson total-count and multinomial profile loss; "
            "'multinomial' is kept as an alias."
        ),
    )
    parser.add_argument("--positional-weight", type=float, default=5.0)
    parser.add_argument("--count-weight", type=float, default=1.0)
    parser.add_argument("--track-means-samples", type=int, default=16)
    parser.add_argument("--max-train-steps", type=int)
    parser.add_argument("--max-val-steps", type=int)
    parser.add_argument("--output-dir", type=Path, default=Path("finetuning_output/tfr_heads"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', 'cuda', or e.g. 'cuda:0'")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--loader-only", action="store_true", help="Decode one batch and exit.")
    return parser.parse_args()


def create_tfr_dataset(
    data_dir: Path,
    split: str,
    modality: str,
    pooling: str,
    *,
    repeat: bool = False,
    shuffle_files: bool = False,
) -> BaskervilleTFRecordDataset:
    return BaskervilleTFRecordDataset(
        data_dir,
        split=split,  # type: ignore[arg-type]
        modality=modality,
        pooling=pooling,  # type: ignore[arg-type]
        repeat=repeat,
        shuffle_files=shuffle_files,
    )


def create_loader(
    dataset: BaskervilleTFRecordDataset,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_tfr_genomic,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )


@torch.no_grad()
def estimate_track_means(
    data_dir: Path,
    modality: str,
    pooling: str,
    max_samples: int,
    batch_size: int,
    num_workers: int,
) -> torch.Tensor:
    dataset = create_tfr_dataset(data_dir, "train", modality, pooling)
    if max_samples <= 0:
        return torch.ones(1, dataset.n_tracks)

    loader = create_loader(dataset, batch_size=batch_size, num_workers=num_workers)
    sums = torch.zeros(dataset.n_tracks, dtype=torch.float64)
    count = 0

    for sequences, targets in loader:
        del sequences
        target = targets[128].double()
        sums += target.sum(dim=(0, 1))
        count += target.shape[0] * target.shape[1]
        if count >= max_samples * target.shape[1]:
            break

    means = (sums / max(1, count)).float().clamp_min(1e-6)
    return means.unsqueeze(0)


def crop_predictions(pred: torch.Tensor, crop_bins: int) -> torch.Tensor:
    if crop_bins == 0:
        return pred
    if pred.shape[1] <= crop_bins * 2:
        raise ValueError(
            f"Cannot crop {crop_bins} bins from prediction length {pred.shape[1]}"
        )
    return pred[:, crop_bins:-crop_bins, :]


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_arg)

    if device.type == "cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            raise RuntimeError(
                f"Requested {device}, but CUDA is not available to PyTorch. "
                "Use --device cpu or run on a GPU-visible node."
            )
        index = 0 if device.index is None else device.index
        if index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested {device}, but only {torch.cuda.device_count()} CUDA "
                "device(s) are visible."
            )
        torch.cuda.set_device(index)
        return torch.device(f"cuda:{index}")

    return device


def autocast_context(device: torch.device, use_amp: bool):
    if not use_amp or device.type != "cuda":
        return nullcontext()
    index = 0 if device.index is None else device.index
    major = torch.cuda.get_device_properties(index).major
    if major >= 8:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type="cuda", dtype=torch.float16)


def compute_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    loss_name: str,
    head: torch.nn.Module,
    organism_idx: torch.Tensor,
    positional_weight: float,
    count_weight: float,
) -> torch.Tensor:
    if loss_name == "mse":
        return F.mse_loss(pred, target)
    if loss_name == "poisson":
        return F.poisson_nll_loss(
            pred.clamp_min(1e-8),
            target,
            log_input=False,
            full=False,
            reduction="mean",
        )
    if loss_name in ("poisson-multinomial", "multinomial"):
        target_scaled = head.scale(target, organism_idx, resolution=128, channels_last=True)
        mask = torch.ones(
            pred.shape[0],
            1,
            pred.shape[-1],
            dtype=torch.bool,
            device=pred.device,
        )
        loss_dict = multinomial_loss(
            y_pred=pred,
            y_true=target_scaled,
            mask=mask,
            multinomial_resolution=max(1, pred.shape[1] // 8),
            positional_weight=positional_weight,
            count_weight=count_weight,
            channels_last=True,
        )
        return loss_dict["loss"]
    raise ValueError(f"Unknown loss: {loss_name}")


def forward_head(
    model: torch.nn.Module,
    head: torch.nn.Module,
    sequences: torch.Tensor,
    organism_idx: torch.Tensor,
    crop_bins: int,
    use_amp: bool,
    return_scaled: bool,
) -> torch.Tensor:
    amp_context = autocast_context(sequences.device, use_amp)
    with torch.no_grad():
        with amp_context:
            outputs = model(
                sequences,
                organism_idx,
                return_embeddings=True,
                resolutions=(128,),
                channels_last=False,
                embeddings_only=True,
            )
            embeddings = {128: outputs["embeddings_128bp"].detach()}
    with amp_context:
        predictions = head(
            embeddings,
            organism_idx,
            return_scaled=return_scaled,
            channels_last=True,
        )
    return crop_predictions(predictions[128], crop_bins)


def run_epoch(
    model: torch.nn.Module,
    head: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    args: argparse.Namespace,
    crop_bins: int,
    max_steps: int | None,
) -> float:
    training = optimizer is not None
    head.train(training)
    model.eval()
    total_loss = 0.0
    steps = 0
    pbar = tqdm(loader, desc="train" if training else "valid")

    for sequences, targets in pbar:
        sequences = sequences.to(device, non_blocking=True)
        target = targets[128].to(device, non_blocking=True)
        organism_idx = torch.zeros(sequences.shape[0], dtype=torch.long, device=device)

        pred = forward_head(
            model,
            head,
            sequences,
            organism_idx,
            crop_bins,
            use_amp=not args.no_amp,
            return_scaled=args.loss in ("poisson-multinomial", "multinomial"),
        )
        if pred.shape != target.shape:
            raise ValueError(
                f"Prediction shape {tuple(pred.shape)} does not match target "
                f"shape {tuple(target.shape)} after crop_bins={crop_bins}"
            )
        loss = compute_loss(
            pred,
            target,
            loss_name=args.loss,
            head=head,
            organism_idx=organism_idx,
            positional_weight=args.positional_weight,
            count_weight=args.count_weight,
        )

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        steps += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        if max_steps is not None and steps >= max_steps:
            break

    return total_loss / max(1, steps)


def save_checkpoint(
    path: Path,
    head: torch.nn.Module,
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"head_state_dict": head.state_dict(), "metadata": metadata}, path)


def main() -> None:
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    device = resolve_device(args.device)

    train_dataset = create_tfr_dataset(
        args.data_dir,
        "train",
        args.modality,
        args.pooling,
        repeat=False,
        shuffle_files=True,
    )
    val_dataset = create_tfr_dataset(args.data_dir, "valid", args.modality, args.pooling)

    train_loader = create_loader(train_dataset, args.batch_size, args.num_workers)
    val_loader = create_loader(val_dataset, args.batch_size, args.num_workers)

    sample_sequences, sample_targets = next(iter(train_loader))
    print(
        "Loader OK:",
        f"seq={tuple(sample_sequences.shape)}",
        f"target_128={tuple(sample_targets[128].shape)}",
        f"tracks={train_dataset.n_tracks}",
        f"crop_bins={train_dataset.prediction_crop_128bp}",
        f"assay={train_dataset.assay_type}",
    )
    if args.loader_only:
        return

    if args.pretrained_weights is None:
        raise SystemExit("--pretrained-weights is required unless --loader-only is set")

    print(f"Device: {device}")
    dtype_policy = (
        DtypePolicy.full_float32()
        if args.dtype == "float32"
        else DtypePolicy.mixed_precision()
    )

    print("Estimating track means...")
    track_means = estimate_track_means(
        args.data_dir,
        args.modality,
        args.pooling,
        args.track_means_samples,
        args.batch_size,
        args.num_workers,
    )
    print(f"Track means: mean={track_means.mean().item():.6g}")

    model = AlphaGenome(dtype_policy=dtype_policy)
    model = load_trunk(model, str(args.pretrained_weights), exclude_heads=True)
    model = remove_all_heads(model).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    head = create_finetuning_head(
        assay_type=train_dataset.assay_type,
        n_tracks=train_dataset.n_tracks,
        resolutions=(128,),
        num_organisms=1,
        track_means=track_means,
    ).to(device)

    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "data_dir": str(args.data_dir),
        "modality": args.modality,
        "assay_type": train_dataset.assay_type,
        "pooling": args.pooling,
        "n_tracks": train_dataset.n_tracks,
        "track_names": [row.get("identifier", "") for row in train_dataset.target_rows],
        "crop_bins_128bp": train_dataset.prediction_crop_128bp,
        "target_length_128bp": train_dataset.output_length_128bp,
        "loss": args.loss,
        "pretrained_weights": str(args.pretrained_weights),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss = run_epoch(
            model,
            head,
            train_loader,
            device,
            optimizer,
            args,
            train_dataset.prediction_crop_128bp,
            args.max_train_steps,
        )
        val_loss = run_epoch(
            model,
            head,
            val_loader,
            device,
            None,
            args,
            val_dataset.prediction_crop_128bp,
            args.max_val_steps,
        )
        print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        save_checkpoint(output_dir / "last_head.pt", head, {**metadata, "epoch": epoch})
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(output_dir / "best_head.pt", head, {**metadata, "epoch": epoch})


if __name__ == "__main__":
    main()
