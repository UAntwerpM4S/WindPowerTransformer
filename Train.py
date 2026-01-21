import os
import time

import numpy as np
import torch
from tqdm import tqdm

from Transformer import TemporalTransformer
from Loader import loader_prepare
from parser import get_args
from data.stats import ensure_stats


def mse(pred, target):
    diff2 = ((pred - target) ** 2).mean() 
    return diff2

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    losses = []
    for inputs, targets in tqdm(loader, desc="Training", leave=False):
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)  # (B, T)
        loss = mse(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
    return float(np.mean(losses))


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    losses = []
    for inputs, targets in tqdm(loader, desc="Validating", leave=False):
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        loss = mse(outputs, targets)
        losses.append(loss.item())
    return float(np.mean(losses))


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fcs_dir = getattr(args, "fcs_dir", "/mnt/weatherloss/WindPowerTransformer/data/FCS")
    obs_dir = getattr(args, "obs_dir", "/mnt/weatherloss/WindPowerTransformer/data/OBS")

    train_dates = getattr(args, "train_dates", "data/train_dates.pkl")
    val_dates = getattr(args, "val_dates", "data/val_dates.pkl")
    test_dates = getattr(args, "test_dates", "data/test_dates.pkl")

    print(f"Training power model for forecast system: {args.model_name} on device: {device}")
    print(f"Using fcs_dir={fcs_dir}")
    print(f"Using obs_dir={obs_dir}")

    # Compute train-only stats (ws10/ws100) if missing
    ensure_stats(
        date_file=train_dates,
        fcs_dir=fcs_dir,
        model_name=args.model_name,
        windpark=args.windpark,
        ws_vars=("ws10", "ws100"),
    )

    model = TemporalTransformer(
        input_dim=args.input_dim,      # e.g. 6 for [ws10, ws100, sin/cos 10m, sin/cos 100m]
        model_dim=args.model_dim,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
        mlp_mult=args.mlp_mult,
    ).to(device)

    train_loader, val_loader, _ = loader_prepare(
        batch_size=args.batch_size,
        model_name=args.model_name,
        windpark=args.windpark,
        fcs_dir=fcs_dir,
        obs_dir=obs_dir,
        train_dates=train_dates,
        val_dates=val_dates,
        test_dates=test_dates,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    patience_counter = 0
    start_time = time.time()

    ckpt_root = getattr(args, "ckpt_dir", "checkpoints")
    tag = getattr(args, "run_tag", "POWER")

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            model_dir = os.path.join(ckpt_root, str(args.model_name))
            os.makedirs(model_dir, exist_ok=True)

            save_path = os.path.join(
                model_dir,
                f"{tag}_"
                f"dim{args.model_dim}_"
                f"park{args.windpark}_"
                f"in{args.input_dim}_"
                f"heads{args.n_heads}_"
                f"layers{args.num_layers}_"
                f"mlp{args.mlp_mult}_"
                f"lr{args.lr}_"
                f"FULL"
                f"ep{args.epochs}.pt"
            )

            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved best model to {save_path}")

        else:
            patience_counter += 1
            print(f"No improvement. Early stopping counter: {patience_counter}/{args.patience}")

        if patience_counter >= args.patience:
            print("⏹️ Early stopping triggered.")
            break

    print(f"\nTraining finished in {time.time() - start_time:.1f} seconds.")


if __name__ == "__main__":
    main()
