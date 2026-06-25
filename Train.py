#!/usr/bin/env python3
"""
Train the TemporalTransformer per windpark on CERRA pseudo-forecasts.

Inputs : 6 CERRA weather variables at the windpark cell.
Target : CERRA `power` (injected real observations) over a 36h window.
Loss   : masked MSE (real obs can have gaps; missing targets are ignored).
"""
import os
import time

import numpy as np
import torch
from tqdm import tqdm

from Transformer import TemporalTransformer
from Loader import loader_prepare
from parser import get_args


def masked_mse(pred, target, mask):
    """MSE over finite-target steps only. mask: (B, T) bool."""
    mask = mask.to(pred.dtype)
    denom = mask.sum().clamp_min(1.0)
    return (((pred - target) ** 2) * mask).sum() / denom


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    losses = []
    for inputs, targets, mask in tqdm(loader, desc="Training", leave=False):
        inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)

        outputs = model(inputs)  # (B, T)
        loss = masked_mse(outputs, targets, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    losses = []
    for inputs, targets, mask in tqdm(loader, desc="Validating", leave=False):
        inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)
        outputs = model(inputs)
        losses.append(masked_mse(outputs, targets, mask).item())
    return float(np.mean(losses)) if losses else float("nan")


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Training CERRA power model | park={args.windpark} | device={device}")
    print(f"zarr={args.zarr_path}")

    train_loader, val_loader, _, _ = loader_prepare(
        windpark=args.windpark,
        zarr_path=args.zarr_path,
        metadata_path=args.metadata_path,
        run_tag=args.run_tag,
        train_range=(args.train_start, args.train_end),
        val_range=(args.val_start, args.val_end),
        test_range=(args.test_start, args.test_end),
        batch_size=args.batch_size,
        target=args.target,
        lead_hours=args.lead_hours,
        freq_hours=args.freq_hours,
        stride=args.stride,
        ensemble=args.ensemble,
    )

    model = TemporalTransformer(
        input_dim=args.input_dim,   # 6
        model_dim=args.model_dim,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
        mlp_mult=args.mlp_mult,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    patience_counter = 0
    start_time = time.time()

    model_dir = os.path.join(args.ckpt_dir, str(args.run_tag))
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(
        model_dir,
        f"{args.run_tag}_dim{args.model_dim}_park{args.windpark}_"
        f"in{args.input_dim}_heads{args.n_heads}_layers{args.num_layers}_"
        f"mlp{args.mlp_mult}_lr{args.lr}_lead{args.lead_hours}_ep{args.epochs}.pt",
    )

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)
        print(f"Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved best model to {save_path}")
        else:
            patience_counter += 1
            print(f"No improvement. Early stop {patience_counter}/{args.patience}")

        if patience_counter >= args.patience:
            print("⏹️ Early stopping triggered.")
            break

    print(f"\nFinished in {time.time() - start_time:.1f}s | best val MSE {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
