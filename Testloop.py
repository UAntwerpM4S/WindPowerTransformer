#!/usr/bin/env python3
import os
import pickle
import numpy as np
import torch
import xarray as xr
from datetime import datetime 
from tqdm import tqdm
from Loader import loader_prepare
import re 
from Transformer import TemporalTransformer

# ======== USER CONFIG ========
MODEL_NAMES = ["GraphTransformer","GNN","Transformer"]  # folders under checkpoints/
INPUT_DIM   = 6
MODEL_DIM   = 128
N_HEADS     = 4
NUM_LAYERS  = 4
MLP_MULT    = 4
BATCH_SIZE  = 8
OUT_DIR     = "TEST_FCS"
# =============================

os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def safe_name(s: str) -> str:
    # replace spaces and any weird characters with underscores
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")


def parse_windpark_from_ckpt(ckpt_name: str) -> str:
    base = ckpt_name.replace(".pt", "")
    parts = base.split("_")

    # find token like "parkBelwind Phase 1" or "parkNobelwind Offshore Windpark"
    park_idx = next((i for i, p in enumerate(parts) if p.startswith("park")), None)
    if park_idx is None:
        return "UNKNOWN"

    # windpark starts after removing "park" prefix from that token
    first = parts[park_idx][len("park"):]
    windpark_parts = [first] if first else []

    # subsequent tokens until we hit hyperparams
    stop_prefixes = ("in", "dim", "heads", "layers", "mlp", "lr", "ep", "POWER")
    for p in parts[park_idx + 1:]:
        if p.startswith(stop_prefixes):
            break
        windpark_parts.append(p)

    windpark = "_".join(windpark_parts).strip("_")
    return windpark or "UNKNOWN"


@torch.no_grad()
def run_inference_to_arrays(model: torch.nn.Module, test_loader):
    model.eval()
    preds_list, truths_list = [], []
    for X, y in tqdm(test_loader, desc="Inference", leave=False):
        X = X.to(device)
        preds = model(X)           # (B, T)
        preds_list.append(preds.cpu().numpy())
        truths_list.append(y.numpy())
    if not preds_list:
        return None, None
    preds  = np.concatenate(preds_list, axis=0)   # (N, T)
    truths = np.concatenate(truths_list, axis=0)  # (N, T)
    return preds, truths

def main():
    with open("data/test_dates.pkl", "rb") as f:
        all_test_dates = pickle.load(f)

    for model_name in MODEL_NAMES:
        ckpt_dir = os.path.join("checkpoints", model_name)
        if not os.path.isdir(ckpt_dir):
            print(f"⚠️ Missing checkpoint directory: {ckpt_dir}")
            continue

        base_model = TemporalTransformer(
            input_dim=INPUT_DIM,
            model_dim=MODEL_DIM,
            n_heads=N_HEADS,
            num_layers=NUM_LAYERS,
            mlp_mult=MLP_MULT,
        ).to(device)

        # Cache test loaders per windpark so we don't rebuild them repeatedly
        loader_cache = {}

        for ckpt_name in sorted(os.listdir(ckpt_dir)):
            if not ckpt_name.endswith(".pt"):
                continue
            # Optional: enforce matching hyperparams in filename
            if f"dim{MODEL_DIM}_" not in ckpt_name or f"heads{N_HEADS}_" not in ckpt_name or f"layers{NUM_LAYERS}_" not in ckpt_name:
                # Skip checkpoints with different architecture
                continue

            ckpt_path = os.path.join(ckpt_dir, ckpt_name)
            windpark = parse_windpark_from_ckpt(ckpt_name)
            print(f"\n🔍 {model_name} | {ckpt_name} | windpark={windpark}")

            # Load weights
            model = base_model
            state = torch.load(ckpt_path, map_location=device,weights_only=True)
            model.load_state_dict(state)


            # Build / get test loader for this (model, windpark)
            cache_key = (model_name, windpark)
            if cache_key not in loader_cache:
                _, _, test_loader = loader_prepare(
                    batch_size=BATCH_SIZE,
                    model_name=model_name,
                    windpark=windpark,
                    fcs_dir= "/mnt/weatherloss/WindPowerTransformer/data/FCS/" + model_name
                )

            else:
                test_loader = loader_cache[cache_key]

            # Inference
            preds, truths = run_inference_to_arrays(model, test_loader)

            n_samples, T = preds.shape

            date_strs = all_test_dates[:n_samples]
            dates = np.array([np.datetime64(datetime.strptime(s, "%Y%m%d%H%M%S")) for s in date_strs], dtype="datetime64[ns]")

            lead_time = np.arange(T, dtype=np.int32)  # step index (0..T-1)


            # Save NetCDF
            ds = xr.Dataset(
                {
                    "forecast": (("date", "lead_time"), preds.astype(np.float32)),
                    "truth":    (("date", "lead_time"), truths.astype(np.float32)),
                },
                coords={"date": dates, "lead_time": lead_time},
            )

            out_name = f"{model_name}_dim{MODEL_DIM}_heads{N_HEADS}_layers{NUM_LAYERS}_{safe_name(windpark)}.nc"

            out_nc = os.path.join(OUT_DIR, out_name)
            ds.to_netcdf(out_nc)


if __name__ == "__main__":
    main()