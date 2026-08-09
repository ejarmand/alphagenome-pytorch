#!/usr/bin/env python

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


BASE_BP = 128


def read_targets(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            row["index"] = int(row["index"])
            rows.append(row)
    return rows


def parse_rna_targets(targets_path):
    rows = read_targets(targets_path)

    rna_rows = [r for r in rows if r["modality"] == "rna"]
    plus_positions = [i for i, r in enumerate(rna_rows) if r["strand"] == "+"]
    minus_positions = [i for i, r in enumerate(rna_rows) if r["strand"] == "-"]

    rna_cols = [r["index"] for r in rna_rows]
    rna_names = [r["identifier"] for r in rna_rows]
    rna_cts = [r["ct"] for r in rna_rows]
    rna_strands = [r["strand"] for r in rna_rows]

    print(f"rna targets: {len(rna_rows)}")
    print(f"rna + targets: {len(plus_positions)}")
    print(f"rna - targets: {len(minus_positions)}")

    return {
        "rna_cols": rna_cols,
        "rna_names": rna_names,
        "rna_cts": rna_cts,
        "rna_strands": rna_strands,
        "plus_positions": plus_positions,
        "minus_positions": minus_positions,
    }


def paired_files(embedding_dir, label_dir):
    embedding_dir = Path(embedding_dir)
    label_dir = Path(label_dir)

    pairs = []
    for emb_path in sorted(embedding_dir.glob("*.npz")):
        label_path = label_dir / emb_path.name
        if label_path.exists():
            pairs.append((emb_path, label_path))

    if not pairs:
        raise FileNotFoundError(f"No matching .npz pairs found in {embedding_dir} and {label_dir}")

    return pairs


class EmbeddingRNADataset(Dataset):
    def __init__(self, embedding_dir, label_dir, rna_cols):
        self.pairs = paired_files(embedding_dir, label_dir)
        self.rna_cols = np.array(rna_cols, dtype=np.int64)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        emb_path, label_path = self.pairs[idx]

        emb_npz = np.load(emb_path)
        if "embeddings_128bp" in emb_npz:
            x = emb_npz["embeddings_128bp"]
        else:
            x = emb_npz[list(emb_npz.keys())[0]]

        if x.ndim == 3 and x.shape[0] == 1:
            x = x[0]

        y_full = np.load(label_path)["y"]
        y = y_full[:, self.rna_cols]

        x = x.astype(np.float32)
        y = y.astype(np.float32)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.clip(y, 0.0, None)

        return torch.from_numpy(x), torch.from_numpy(y)


class RNAPlusMinusDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout, n_rna_targets, plus_positions, minus_positions):
        super().__init__()

        self.register_buffer("plus_positions", torch.tensor(plus_positions, dtype=torch.long))
        self.register_buffer("minus_positions", torch.tensor(minus_positions, dtype=torch.long))
        self.n_rna_targets = n_rna_targets

        self.trunk = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.plus_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(plus_positions)),
            nn.Softplus(),
        )

        self.minus_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(minus_positions)),
            nn.Softplus(),
        )

    def forward(self, x):
        h = self.trunk(x)

        plus = self.plus_head(h)
        minus = self.minus_head(h)

        out = torch.zeros(
            x.shape[0],
            x.shape[1],
            self.n_rna_targets,
            device=x.device,
            dtype=x.dtype,
        )

        out.index_copy_(2, self.plus_positions, plus)
        out.index_copy_(2, self.minus_positions, minus)

        return out


def pool_torch(x, pool_factor):
    if pool_factor == 1:
        return x

    usable = (x.shape[1] // pool_factor) * pool_factor
    x = x[:, :usable, :]
    return x.reshape(x.shape[0], usable // pool_factor, pool_factor, x.shape[2]).mean(dim=2)


def transform_labels(y, transform):
    if transform == "none":
        return y
    if transform == "log1p":
        return torch.log1p(torch.clamp(y, min=0.0))
    raise ValueError(transform)


def inverse_transform_np(y, transform):
    if transform == "none":
        return y
    if transform == "log1p":
        return np.expm1(y)
    raise ValueError(transform)


def mse_loss(pred, y):
    mask = torch.isfinite(pred) & torch.isfinite(y)
    diff = pred[mask] - y[mask]
    if diff.numel() == 0:
        return pred.sum() * 0.0
    return torch.mean(diff * diff)


def pearson_np(a, b):
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)

    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]

    if a.size < 2:
        return np.nan

    if a.std() == 0 or b.std() == 0:
        return np.nan

    return float(np.corrcoef(a, b)[0, 1])


def r2_np(pred, true):
    pred = np.asarray(pred).reshape(-1)
    true = np.asarray(true).reshape(-1)

    mask = np.isfinite(pred) & np.isfinite(true)
    pred = pred[mask]
    true = true[mask]

    if true.size < 2:
        return np.nan

    denom = np.sum((true - true.mean()) ** 2)
    if denom == 0:
        return np.nan

    return float(1.0 - np.sum((true - pred) ** 2) / denom)


@torch.no_grad()
def evaluate(model, loader, device, label_transform, pool_factor, rna_cts, rna_strands):
    model.eval()

    total_loss = 0.0
    n_batches = 0

    preds = []
    trues = []

    for x, y_raw in loader:
        x = x.to(device, non_blocking=True)
        y_raw = y_raw.to(device, non_blocking=True)

        pred = model(x)

        pred_pool = pool_torch(pred, pool_factor)
        y_pool_raw = pool_torch(y_raw, pool_factor)
        y_pool = transform_labels(y_pool_raw, label_transform)

        loss = mse_loss(pred_pool, y_pool)

        total_loss += float(loss.detach().cpu())
        n_batches += 1

        pred_np = pred_pool.detach().cpu().numpy()
        true_np = y_pool.detach().cpu().numpy()

        pred_np = inverse_transform_np(pred_np, label_transform)
        true_np = inverse_transform_np(true_np, label_transform)

        preds.append(pred_np)
        trues.append(true_np)

    pred_all = np.concatenate(preds, axis=0)
    true_all = np.concatenate(trues, axis=0)

    metrics = {}
    metrics["loss"] = total_loss / max(n_batches, 1)
    metrics["rna_pearson_r"] = pearson_np(pred_all, true_all)
    metrics["rna_r2"] = r2_np(pred_all, true_all)

    plus_idxs = [i for i, s in enumerate(rna_strands) if s == "+"]
    minus_idxs = [i for i, s in enumerate(rna_strands) if s == "-"]

    metrics["rna_plus_pearson_r"] = pearson_np(pred_all[:, :, plus_idxs], true_all[:, :, plus_idxs])
    metrics["rna_minus_pearson_r"] = pearson_np(pred_all[:, :, minus_idxs], true_all[:, :, minus_idxs])
    metrics["rna_plus_r2"] = r2_np(pred_all[:, :, plus_idxs], true_all[:, :, plus_idxs])
    metrics["rna_minus_r2"] = r2_np(pred_all[:, :, minus_idxs], true_all[:, :, minus_idxs])

    for ct in sorted(set(rna_cts)):
        idxs = [i for i, x in enumerate(rna_cts) if x == ct]
        metrics[f"rna_pearson_by_ct/{ct}"] = pearson_np(pred_all[:, :, idxs], true_all[:, :, idxs])
        metrics[f"rna_r2_by_ct/{ct}"] = r2_np(pred_all[:, :, idxs], true_all[:, :, idxs])

    return metrics


def run_train_epoch(model, loader, optimizer, device, label_transform, pool_factor):
    model.train()

    total_loss = 0.0
    n_batches = 0
    grad_norms = []

    for x, y_raw in loader:
        x = x.to(device, non_blocking=True)
        y_raw = y_raw.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        pred = model(x)

        pred_pool = pool_torch(pred, pool_factor)
        y_pool_raw = pool_torch(y_raw, pool_factor)
        y_pool = transform_labels(y_pool_raw, label_transform)

        loss = mse_loss(pred_pool, y_pool)
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        grad_norms.append(float(grad_norm.detach().cpu()))
        total_loss += float(loss.detach().cpu())
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "grad_norm": float(np.mean(grad_norms)) if grad_norms else np.nan,
        "grad_norm_max": float(np.max(grad_norms)) if grad_norms else np.nan,
    }


def make_scheduler(optimizer, args):
    if args.scheduler == "none":
        return None

    warmup_epochs = max(args.warmup_epochs, 0)
    total_epochs = max(args.epochs, 1)

    def lr_lambda(epoch_zero_based):
        epoch = epoch_zero_based + 1

        if warmup_epochs > 0 and epoch <= warmup_epochs:
            return epoch / warmup_epochs

        min_factor = args.min_lr / args.lr
        progress = min(max((epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1), 0.0), 1.0)
        return min_factor + 0.5 * (1.0 - min_factor) * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def weight_norms(model):
    model_to_use = model.module if isinstance(model, nn.DataParallel) else model

    values = []
    max_value = 0.0

    for p in model_to_use.parameters():
        if p.requires_grad:
            norm = float(torch.linalg.vector_norm(p.detach()).cpu())
            values.append(norm)
            max_value = max(max_value, norm)

    return {
        "weight_norm": float(np.mean(values)) if values else np.nan,
        "weight_norm_max": max_value,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--targets", required=True)
    parser.add_argument("--train-embeddings", required=True)
    parser.add_argument("--train-labels", required=True)
    parser.add_argument("--valid-embeddings", required=True)
    parser.add_argument("--valid-labels", required=True)
    parser.add_argument("--out", required=True)

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--scheduler", choices=["none", "cosine"], default="cosine")
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=2)

    parser.add_argument("--rna-label-transform", choices=["none", "log1p"], default="log1p")
    parser.add_argument("--pool-bp", type=int, default=512)

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)

    args = parser.parse_args()

    if args.pool_bp % BASE_BP != 0:
        raise ValueError(f"--pool-bp must be divisible by {BASE_BP}")

    pool_factor = args.pool_bp // BASE_BP

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_info = parse_rna_targets(args.targets)

    train_ds = EmbeddingRNADataset(args.train_embeddings, args.train_labels, target_info["rna_cols"])
    valid_ds = EmbeddingRNADataset(args.valid_embeddings, args.valid_labels, target_info["rna_cols"])

    print(f"train pairs: {len(train_ds)}")
    print(f"valid pairs: {len(valid_ds)}")

    x0, y0 = train_ds[0]
    print(f"embedding shape: {tuple(x0.shape)}")
    print(f"rna label shape at 128 bp: {tuple(y0.shape)}")
    print(f"training/eval pool bp: {args.pool_bp}")
    print(f"pool factor: {pool_factor}")
    print(f"label transform: {args.rna_label_transform}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model = RNAPlusMinusDecoder(
        input_dim=x0.shape[-1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        n_rna_targets=len(target_info["rna_cols"]),
        plus_positions=target_info["plus_positions"],
        minus_positions=target_info["minus_positions"],
    ).to(device)

    if torch.cuda.device_count() > 1:
        print(f"using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = make_scheduler(optimizer, args)

    wandb_run = None
    if args.wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config=vars(args),
        )

    best_valid = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            label_transform=args.rna_label_transform,
            pool_factor=pool_factor,
        )

        valid_metrics = evaluate(
            model=model,
            loader=valid_loader,
            device=device,
            label_transform=args.rna_label_transform,
            pool_factor=pool_factor,
            rna_cts=target_info["rna_cts"],
            rna_strands=target_info["rna_strands"],
        )

        norms = weight_norms(model)

        if scheduler is not None:
            scheduler.step()

        lr = optimizer.param_groups[0]["lr"]

        log = {
            "epoch": epoch,
            "lr": lr,
            "pool_bp": args.pool_bp,
            "train/loss": train_metrics["loss"],
            "train/grad_norm": train_metrics["grad_norm"],
            "train/grad_norm_max": train_metrics["grad_norm_max"],
            "valid/loss": valid_metrics["loss"],
            "valid/rna_pearson_r": valid_metrics["rna_pearson_r"],
            "valid/rna_r2": valid_metrics["rna_r2"],
            "valid/rna_plus_pearson_r": valid_metrics["rna_plus_pearson_r"],
            "valid/rna_minus_pearson_r": valid_metrics["rna_minus_pearson_r"],
            "valid/rna_plus_r2": valid_metrics["rna_plus_r2"],
            "valid/rna_minus_r2": valid_metrics["rna_minus_r2"],
            "valid/weight_norm": norms["weight_norm"],
            "valid/weight_norm_max": norms["weight_norm_max"],
        }

        for k, v in valid_metrics.items():
            if k.startswith("rna_pearson_by_ct/"):
                log[f"valid/{k}"] = v
            if k.startswith("rna_r2_by_ct/"):
                log[f"valid/{k}"] = v

        print(
            f"epoch={epoch} "
            f"lr={lr:.8f} "
            f"pool_bp={args.pool_bp} "
            f"train_loss={train_metrics['loss']:.6f} "
            f"valid_loss={valid_metrics['loss']:.6f} "
            f"valid_rna_pearson_r={valid_metrics['rna_pearson_r']:.6f} "
            f"valid_rna_r2={valid_metrics['rna_r2']:.6f} "
            f"valid_rna_plus_pearson_r={valid_metrics['rna_plus_pearson_r']:.6f} "
            f"valid_rna_minus_pearson_r={valid_metrics['rna_minus_pearson_r']:.6f}",
            flush=True,
        )

        if wandb_run is not None:
            wandb.log(log, step=epoch)

        if valid_metrics["loss"] < best_valid:
            best_valid = valid_metrics["loss"]

            model_to_save = model.module if isinstance(model, nn.DataParallel) else model
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model_to_save.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_valid_loss": best_valid,
                "args": vars(args),
                "rna_cols": target_info["rna_cols"],
                "rna_names": target_info["rna_names"],
                "rna_cts": target_info["rna_cts"],
                "rna_strands": target_info["rna_strands"],
                "plus_positions": target_info["plus_positions"],
                "minus_positions": target_info["minus_positions"],
            }

            checkpoint_path = out_dir / "best_rna_decoder.pt"
            torch.save(checkpoint, checkpoint_path)
            print(f"saved best checkpoint: {checkpoint_path}", flush=True)

    print(f"best_valid_loss={best_valid:.6f}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
