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

    print(f"Training cell-level power model | run_tag={args.run_tag} | device={device}")
    print(f"dataset={args.dataset_nc}")

    from pathlib import Path
    train_loader, val_loader, _, geom, input_dim = loader_prepare(
        dataset_nc=Path(args.dataset_nc),
        farms_csv=args.farms_csv,
        specs_csv=args.specs_csv,
        run_tag=args.run_tag,
        batch_size=args.batch_size,
    )
    print(f"input_dim (auto from dataset): {input_dim}")

    # save the cell->farm aggregation geometry next to the checkpoint (needed to score later)
    geom_dir = os.path.join("artifacts", str(args.run_tag))
    os.makedirs(geom_dir, exist_ok=True)
    np.savez(os.path.join(geom_dir, "cell_geom.npz"),
             farms=np.array(geom["farms"]), G=geom["G"], cap_cell=geom["cap_cell"],
             cell_lat=geom["cell_lat"], cell_lon=geom["cell_lon"])

    model = TemporalTransformer(
        input_dim=input_dim,        # auto: 6 wind (+cf_lam) + 3 static
        model_dim=args.model_dim,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
        mlp_mult=args.mlp_mult,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: dim{args.model_dim} x{args.num_layers} layers, dropout {args.dropout}, "
          f"{n_params/1e3:.0f}k params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_loss = float("inf")
    patience_counter = 0
    start_time = time.time()

    model_dir = os.path.join(args.ckpt_dir, str(args.run_tag))
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(
        model_dir,
        f"{args.run_tag}_dim{args.model_dim}_cells_"
        f"in{input_dim}_heads{args.n_heads}_layers{args.num_layers}_"
        f"mlp{args.mlp_mult}_lr{args.lr}_ep{args.epochs}.pt",
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
