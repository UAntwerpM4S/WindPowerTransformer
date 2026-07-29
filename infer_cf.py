#!/usr/bin/env python3
"""INFERENCE (forecast tier): apply each run's post-processor to its own dataset -> CF forecasts.

ONE model per forecasting run. The forecast-tier post-processor conditions on the run's own
`cf_lam` (the LAM's capacity-factor forecast) alongside the 6 wind vars + 3 static features, so it
runs on the SAME dataset_BE_<RUN>.nc the model trained on (feature parity is exact -- both go
through Loader.inference_features / apply_stats). It writes, per run:

    cf_BE_<RUN>.nc
      cf          (init, lead_time, cell)   predicted capacity factor in [0,1]   <- the product
      split       (init)                    train/val/test  (score_cf.py reports on 'test')
      G, cap_cell, cell_lat/lon, valid_time, farm

Predicts EVERY init (cheap); score_cf.py filters to the held-out 'test' split. The model attends
over the full 3..36 h window, which is exactly how the dataset is laid out, so nothing is dropped.

Edit SETTINGS, then:  python infer_cf.py
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import torch

from Transformer import TemporalTransformer
from Loader import inference_features

# -------------------- SETTINGS --------------------
WPOWER_DIR = Path("/mnt/weatherloss/WindPower/data/WPDistr")
CELLS_DIR  = Path("/mnt/weatherloss/WindPowerTransformer/data/cells")     # dataset_BE_<RUN>.nc
OUT_DIR    = Path("/mnt/weatherloss/WindPowerTransformer/data/cf_forecasts")
CKPT_DIR   = Path("checkpoints")

# One post-processor per experiment: (tag, source LAM run, use_cf_lam).
#   tag        = run_tag it was trained under (checkpoints/<tag>/, artifacts/<tag>/); also the
#                output file cf_<REGION>_<tag>.nc
#   source run = which dataset_BE_<run>.nc feeds it (the wind-only variant of a power run reuses
#                that run's dataset, just dropping cf_lam)
#   use_cf_lam = 6 wind + cf_lam + static (True) vs wind-only 6 wind + static (False)
EXPERIMENTS = [
    ("HighCapacityGT",        "HighCapacityGT",     True),
    ("VanillaPowerGT",        "VanillaPowerGT",     True),
    ("VeryHighCapacityGT",    "VeryHighCapacityGT", True),
    ("HighCapacityGT_wo",     "HighCapacityGT",     False),   # wind-only variants
    ("VanillaPowerGT_wo",     "VanillaPowerGT",     False),
    ("VeryHighCapacityGT_wo", "VeryHighCapacityGT", False),
    ("RegularWeather",        "RegularWeather",     False),   # pure weather model (no cf_lam)
]
REGION = "BE"

# architecture — MUST match how the per-run models were trained (Train.py defaults)
MODEL_DIM, N_HEADS, NUM_LAYERS, MLP_MULT = 64, 4, 2, 4
# --------------------------------------------------


def find_ckpt(ckpt_dir: Path, run: str, input_dim: int) -> str:
    cands = sorted(glob.glob(str(ckpt_dir / run / f"*in{input_dim}*.pt")))
    if not cands:
        cands = sorted(glob.glob(str(ckpt_dir / run / "*.pt")))
    if not cands:
        raise FileNotFoundError(f"no checkpoint in {ckpt_dir / run} -- train run_tag={run} first")
    return cands[-1]


@torch.no_grad()
def predict_cf(model, X, device, batch=8192):
    """X (N,L,C,F) -> cf (N,L,C) clamped to [0,1]. Each (init,cell) is one sample."""
    N, L, C, F = X.shape
    flat = X.transpose(0, 2, 1, 3).reshape(N * C, L, F)
    out = np.empty((N * C, L), dtype=np.float32)
    model.eval()
    for s in range(0, flat.shape[0], batch):
        xb = torch.from_numpy(flat[s:s + batch]).to(device)
        out[s:s + batch] = model(xb).cpu().numpy()
    return np.clip(out.reshape(N, C, L).transpose(0, 2, 1), 0.0, 1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region", default=REGION)
    ap.add_argument("--cells-dir", type=Path, default=CELLS_DIR)
    ap.add_argument("--ckpt-dir", type=Path, default=CKPT_DIR)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--wpower-dir", type=Path, default=WPOWER_DIR)
    ap.add_argument("--model_dim", type=int, default=MODEL_DIM)
    ap.add_argument("--n_heads", type=int, default=N_HEADS)
    ap.add_argument("--num_layers", type=int, default=NUM_LAYERS)
    ap.add_argument("--mlp_mult", type=int, default=MLP_MULT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    farms_csv = str(args.wpower_dir / "farms.csv")
    specs_csv = str(args.wpower_dir / "turbine_specs.csv")

    for tag, source_run, use_cf_lam in EXPERIMENTS:
        dataset = args.cells_dir / f"dataset_{args.region}_{source_run}.nc"
        if not dataset.exists():
            print(f"\n{tag}: {dataset} missing -- run the builder first, skipping")
            continue
        print(f"\n{'='*70}\n{tag}  (source {source_run}, cf_lam={use_cf_lam})\n{'='*70}")

        feats = inference_features(dataset, farms_csv, specs_csv, run_tag=tag, use_cf_lam=use_cf_lam)
        X = feats["X"]
        input_dim = X.shape[-1]
        ckpt = find_ckpt(args.ckpt_dir, tag, input_dim)

        model = TemporalTransformer(input_dim=input_dim, model_dim=args.model_dim,
                                    n_heads=args.n_heads, num_layers=args.num_layers,
                                    mlp_mult=args.mlp_mult).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"  {os.path.basename(ckpt)}  | input_dim {input_dim} "
              f"({', '.join(feats['feat_names'])})")

        cf = predict_cf(model, X, device)                                  # (N,L,C) in [0,1]
        C = cf.shape[2]

        out = xr.Dataset(
            {"cf":         (("init", "lead_time", "cell"), cf),
             "split":      ("init", feats["split"]),
             "G":          (("farm", "cell"), feats["G"]),
             "cap_cell":   ("cell", feats["cap_cell"]),
             "valid_time": (("init", "lead_time"), feats["valid"])},
            coords={"init": feats["init"].values, "lead_time": feats["leads"],
                    "cell": np.arange(C, dtype=np.int32),
                    "farm": np.array(feats["farms"], dtype=object),
                    "cell_lat": ("cell", feats["cell_lat"]),
                    "cell_lon": ("cell", feats["cell_lon"])},
            attrs={"tag": tag, "source_run": source_run, "use_cf_lam": int(use_cf_lam),
                   "region": args.region, "checkpoint": os.path.basename(ckpt),
                   "inputs": ", ".join(feats["feat_names"]),
                   "reconstruction": "P(farm,t) = sum_cell G[farm,cell]*cf(cell,t)"})
        out["lead_time"].attrs["units"] = "h"
        enc = {v: {"zlib": True, "complevel": 4} for v in ("cf", "G", "cap_cell")}
        out_path = args.out / f"cf_{args.region}_{tag}.nc"
        tmp = out_path.with_suffix(".nc.tmp")
        out.to_netcdf(tmp, format="NETCDF4", engine="netcdf4", encoding=enc)
        tmp.replace(out_path)

        te = feats["split"] == "test"
        print(f"  wrote {out_path}  (cf mean {cf.mean():.4f} max {cf.max():.4f}; "
              f"{int(te.sum())} test inits)")


if __name__ == "__main__":
    main()
