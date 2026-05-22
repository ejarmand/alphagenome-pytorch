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
    BaskervilleMultiTFRecordDataset,
    collate_tfr_multimodal,
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
    parser.add_argument(
        "--modality",
        action="append",
        default=None,
        help=(
            "Modality to train. Repeat for multiple heads, or omit/use 'all' "
            "to train all modalities in targets.txt."
        ),
    )
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
    modalities: list[str],
    pooling: str,
    *,
    repeat: bool = False,
    shuffle_files: bool = False,
) -> BaskervilleMultiTFRecordDataset:
    return BaskervilleMultiTFRecordDataset(
        data_dir,
        split=split,  # type: ignore[arg-type]
        modalities=modalities,
        pooling=pooling,  # type: ignore[arg-type]
        repeat=repeat,
        shuffle_files=shuffle_files,
    )


def create_loader(
    dataset: BaskervilleMultiTFRecordDataset,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_tfr_multimodal,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )


@torch.no_grad()
def estimate_track_means(
    data_dir: Path,
    modalities: list[str],
    pooling: str,
    max_samples: int,
    batch_size: int,
    num_workers: int,
) -> dict[str, torch.Tensor]:
    dataset = create_tfr_dataset(data_dir, "train", modalities, pooling)
    if max_samples <= 0:
        return {
            modality: torch.ones(1, len(dataset.target_indices_by_modality[modality]))
            for modality in dataset.modalities
        }

    loader = create_loader(dataset, batch_size=batch_size, num_workers=num_workers)
    sums = {
        modality: torch.zeros(len(indices), dtype=torch.float64)
        for modality, indices in dataset.target_indices_by_modality.items()
    }
    position_counts = {modality: 0 for modality in dataset.modalities}
    samples_seen = 0

    for sequences, modality_targets in loader:
        del sequences
        for modality, targets in modality_targets.items():
            target = targets[128].double()
            sums[modality] += target.sum(dim=(0, 1))
            position_counts[modality] += target.shape[0] * target.shape[1]
        samples_seen += next(iter(modality_targets.values()))[128].shape[0]
        if samples_seen >= max_samples:
            break

    return {
        modality: (
            sums[modality] / max(1, position_counts[modality])
        ).float().clamp_min(1e-6).unsqueeze(0)
        for modality in dataset.modalities
    }


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


def forward_heads(
    model: torch.nn.Module,
    heads: torch.nn.ModuleDict,
    sequences: torch.Tensor,
    organism_idx: torch.Tensor,
    crop_bins: int,
    use_amp: bool,
    return_scaled: bool,
) -> dict[str, torch.Tensor]:
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
    predictions_by_modality = {}
    with amp_context:
        for modality, head in heads.items():
            predictions = head(
                embeddings,
                organism_idx,
                return_scaled=return_scaled,
                channels_last=True,
            )
            predictions_by_modality[modality] = crop_predictions(
                predictions[128],
                crop_bins,
            )
    return predictions_by_modality


def run_epoch(
    model: torch.nn.Module,
    heads: torch.nn.ModuleDict,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    args: argparse.Namespace,
    crop_bins: int,
    max_steps: int | None,
) -> float:
    training = optimizer is not None
    heads.train(training)
    model.eval()
    total_loss = 0.0
    steps = 0
    pbar = tqdm(loader, desc="train" if training else "valid")

    for sequences, modality_targets in pbar:
        sequences = sequences.to(device, non_blocking=True)
        organism_idx = torch.zeros(sequences.shape[0], dtype=torch.long, device=device)

        predictions_by_modality = forward_heads(
            model,
            heads,
            sequences,
            organism_idx,
            crop_bins,
            use_amp=not args.no_amp,
            return_scaled=args.loss in ("poisson-multinomial", "multinomial"),
        )
        loss = torch.tensor(0.0, device=device)
        loss_by_modality = {}
        for modality, pred in predictions_by_modality.items():
            target = modality_targets[modality][128].to(device, non_blocking=True)
            if pred.shape != target.shape:
                raise ValueError(
                    f"{modality}: prediction shape {tuple(pred.shape)} does not "
                    f"match target shape {tuple(target.shape)} after crop_bins={crop_bins}"
                )
            modality_loss = compute_loss(
                pred,
                target,
                loss_name=args.loss,
                head=heads[modality],
                organism_idx=organism_idx,
                positional_weight=args.positional_weight,
                count_weight=args.count_weight,
            )
            loss = loss + modality_loss
            loss_by_modality[modality] = modality_loss.item()

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        steps += 1
        postfix = {"loss": f"{loss.item():.4f}"}
        postfix.update({
            modality: f"{modality_loss:.4f}"
            for modality, modality_loss in loss_by_modality.items()
        })
        pbar.set_postfix(postfix)
        if max_steps is not None and steps >= max_steps:
            break

    return total_loss / max(1, steps)


def save_checkpoint(
    path: Path,
    heads: torch.nn.ModuleDict,
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"heads_state_dict": heads.state_dict(), "metadata": metadata}, path)


def resolve_modalities(args: argparse.Namespace) -> list[str]:
    requested = args.modality or ["all"]
    if len(requested) == 1 and requested[0].lower() == "all":
        return BaskervilleTFRecordDataset.available_modalities(args.data_dir)
    if any(modality.lower() == "all" for modality in requested):
        raise ValueError("Use either --modality all or explicit repeated --modality values")
    return requested


def main() -> None:
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    device = resolve_device(args.device)
    modalities = resolve_modalities(args)

    train_dataset = create_tfr_dataset(
        args.data_dir,
        "train",
        modalities,
        args.pooling,
        repeat=False,
        shuffle_files=True,
    )
    val_dataset = create_tfr_dataset(args.data_dir, "valid", modalities, args.pooling)

    train_loader = create_loader(train_dataset, args.batch_size, args.num_workers)
    val_loader = create_loader(val_dataset, args.batch_size, args.num_workers)

    sample_sequences, sample_targets = next(iter(train_loader))
    print(
        "Loader OK:",
        f"seq={tuple(sample_sequences.shape)}",
        "targets="
        + ",".join(
            f"{modality}:{tuple(targets[128].shape)}"
            for modality, targets in sample_targets.items()
        ),
        f"crop_bins={train_dataset.prediction_crop_128bp}",
        f"modalities={modalities}",
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
        modalities,
        args.pooling,
        args.track_means_samples,
        args.batch_size,
        args.num_workers,
    )
    print(
        "Track means:",
        ", ".join(
            f"{modality}={means.mean().item():.6g}"
            for modality, means in track_means.items()
        ),
    )

    model = AlphaGenome(dtype_policy=dtype_policy)
    model = load_trunk(model, str(args.pretrained_weights), exclude_heads=True)
    model = remove_all_heads(model).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    heads = torch.nn.ModuleDict()
    for modality in modalities:
        heads[modality] = create_finetuning_head(
            assay_type=train_dataset.assay_type_by_modality[modality],
            n_tracks=len(train_dataset.target_indices_by_modality[modality]),
            resolutions=(128,),
            num_organisms=1,
            track_means=track_means[modality],
        )
    heads = heads.to(device)

    optimizer = torch.optim.AdamW(heads.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "data_dir": str(args.data_dir),
        "modalities": modalities,
        "assay_types": train_dataset.assay_type_by_modality,
        "pooling": args.pooling,
        "n_tracks": {
            modality: len(train_dataset.target_indices_by_modality[modality])
            for modality in modalities
        },
        "track_names": {
            modality: [
                row.get("identifier", "")
                for row in train_dataset.target_rows_by_modality[modality]
            ]
            for modality in modalities
        },
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
            heads,
            train_loader,
            device,
            optimizer,
            args,
            train_dataset.prediction_crop_128bp,
            args.max_train_steps,
        )
        val_loss = run_epoch(
            model,
            heads,
            val_loader,
            device,
            None,
            args,
            val_dataset.prediction_crop_128bp,
            args.max_val_steps,
        )
        print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        save_checkpoint(output_dir / "last_heads.pt", heads, {**metadata, "epoch": epoch})
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(output_dir / "best_heads.pt", heads, {**metadata, "epoch": epoch})


if __name__ == "__main__":
    main()
