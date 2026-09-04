#!/usr/bin/env python3
"""Apply the trained post-processor to the run's forecast wind and write cf_<REGION>_<tag>.nc.

One experiment: the transformer trained by run.py on 6 wind variables + static features, applied
to the forecasts in dataset_<REGION>_<run>.nc. Output feeds score_cf.py.

EXPERIMENTS entries are (checkpoint tag, source dataset run, use_cf_lam). use_cf_lam is False
throughout -- this report post-processes wind only, and the checkpoint's input_dim would not
match a 7th channel anyway (guarded below).
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
from Loader import inference_features, kfold_assign

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
    ("RegularWeather", "RegularWeather", False),
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


def load_and_predict(ckpt_dir, tag, feats, args, device):
    """Find the checkpoint for run_tag=tag, load it, predict CF for feats['X'].

    The checkpoint is validated against the feature count BEFORE use. find_ckpt falls back to any
    *.pt when the `in<dim>` glob misses, which is fine within one tier but dangerous across tiers
    (a 6-wind CERRA model must never silently receive a 7-channel forecast input), so the weight
    shape is the authority, not the filename.
    """
    input_dim = feats["X"].shape[-1]
    ckpt = find_ckpt(ckpt_dir, tag, input_dim)
    state = torch.load(ckpt, map_location=device)
    ckpt_dim = int(state["input_proj.weight"].shape[1])
    if ckpt_dim != input_dim:
        raise SystemExit(
            f"input_dim mismatch: checkpoint {os.path.basename(ckpt)} expects {ckpt_dim} "
            f"features, this dataset builds {input_dim} ({', '.join(feats['feat_names'])}).\n"
            f"  A CERRA-tier converter is 6 wind + 3 static = 9; a forecast-tier model with "
            f"cf_lam is 10. Set use_cf_lam=False for the CERRA rows in EXPERIMENTS.")
    model = TemporalTransformer(input_dim=input_dim, model_dim=args.model_dim,
                                n_heads=args.n_heads, num_layers=args.num_layers,
                                mlp_mult=args.mlp_mult).to(device)
    model.load_state_dict(state)
    return predict_cf(model, feats["X"], device), os.path.basename(ckpt)


def check_cells_match(ckpt_dir, tag, feats, strict=True):
    """The checkpoint's training cells must be the SAME physical cells as this dataset's.

    Only matters across tiers: the CERRA converter is trained on cells indexed into the full
    72,668-cell CERRA grid, then applied to a forecast file whose grid is a SUBSET of it. The
    cells themselves are the same 15 Belgian cells and each carries its own static features, so
    ordering is not load-bearing -- but a genuine mismatch (a farm cell missing from the forecast
    cutout, or a different region) would corrupt the farm aggregation silently. Train.py saves the
    training geometry to artifacts/<tag>/cell_geom.npz; compare against it.
    """
    p = os.path.join("artifacts", str(tag), "cell_geom.npz")
    if not os.path.exists(p):
        print(f"  ! no {p} -- cannot verify cell geometry (train {tag} to write it)")
        return
    g = np.load(p, allow_pickle=True)
    n_ck, n_ds = len(g["cell_lat"]), len(feats["cell_lat"])
    if n_ck != n_ds:
        msg = f"cell count differs: checkpoint {tag} trained on {n_ck}, dataset has {n_ds}"
    else:
        dlat = np.abs(g["cell_lat"] - feats["cell_lat"]).max()
        dlon = np.abs(g["cell_lon"] - feats["cell_lon"]).max()
        dcap = np.abs(g["cap_cell"] - feats["cap_cell"]).max()
        if max(dlat, dlon) > 1e-3 or dcap > 1e-3:
            msg = (f"cells differ: max |dlat|={dlat:.5f} |dlon|={dlon:.5f} deg, "
                   f"|dcap|={dcap:.4f} MW")
        else:
            print(f"  cells verified: {n_ds} cells match the checkpoint "
                  f"(max offset {max(dlat, dlon):.1e} deg)")
            return
    if strict:
        raise SystemExit(f"CELL MISMATCH for {tag} on this dataset -- {msg}.\n"
                         f"  Re-extract with the same region and turbine set, or pass "
                         f"--no-strict-cells to score anyway.")
    print(f"  ! CELL MISMATCH ({msg}) -- continuing because --no-strict-cells")


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
    ap.add_argument("--no-strict-cells", dest="strict_cells", action="store_false",
                    help="warn instead of aborting when the dataset's cells do not match the "
                         "cells the checkpoint was trained on")
    ap.add_argument("--only", nargs="+", default=None, metavar="TAG",
                    help="run only these EXPERIMENTS tags (e.g. --only RegularWeather)")
    ap.add_argument("--kfold", type=int, default=0,
                    help="assemble out-of-fold predictions from K per-fold models <tag>_f0..f{K-1}")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    farms_csv = str(args.wpower_dir / "farms.csv")
    specs_csv = str(args.wpower_dir / "turbine_specs.csv")

    experiments = [e for e in EXPERIMENTS if not args.only or e[0] in args.only]
    if not experiments:
        raise SystemExit(f"--only {args.only} matched no EXPERIMENTS tag")
    for tag, source_run, use_cf_lam in experiments:
        dataset = args.cells_dir / f"dataset_{args.region}_{source_run}.nc"
        if not dataset.exists():
            print(f"\n{tag}: {dataset} missing -- run the builder first, skipping")
            continue
        print(f"\n{'='*70}\n{tag}  (source {source_run}, cf_lam={use_cf_lam}"
              f"{f', {args.kfold}-fold OOF' if args.kfold else ''})\n{'='*70}")

        if args.kfold:
            # assemble out-of-fold predictions: each fold model predicts only its own held-out fold
            oof = fold = feats = None
            for f in range(args.kfold):
                ff = inference_features(dataset, farms_csv, specs_csv,
                                        run_tag=f"{tag}_f{f}", use_cf_lam=use_cf_lam)
                if oof is None:
                    oof = np.full(ff["X"].shape[:3], np.nan, dtype=np.float32)
                    fold = kfold_assign(ff["init"], args.kfold)
                    feats = ff
                cf_f, name = load_and_predict(args.ckpt_dir, f"{tag}_f{f}", ff, args, device)
                m = fold == f
                oof[m] = cf_f[m]
                print(f"  fold {f}: {int(m.sum())} inits <- {name}")
            cf = oof
            split_coord = np.array(["test"] * len(feats["init"]), dtype=object)  # all OOF
            ckpt_desc = f"{tag}_f0..f{args.kfold - 1} (OOF)"
        else:
            feats = inference_features(dataset, farms_csv, specs_csv, run_tag=tag,
                                       use_cf_lam=use_cf_lam)
            cf, ckpt_desc = load_and_predict(args.ckpt_dir, tag, feats, args, device)
            split_coord = feats["split"]
            print(f"  {ckpt_desc}  | input_dim {feats['X'].shape[-1]} "
                  f"({', '.join(feats['feat_names'])})")
        check_cells_match(args.ckpt_dir, tag, feats, strict=args.strict_cells)
        C = cf.shape[2]

        out = xr.Dataset(
            {"cf":         (("init", "lead_time", "cell"), cf),
             "split":      ("init", split_coord),
             "G":          (("farm", "cell"), feats["G"]),
             "cap_cell":   ("cell", feats["cap_cell"]),
             "valid_time": (("init", "lead_time"), feats["valid"])},
            coords={"init": feats["init"].values, "lead_time": feats["leads"],
                    "cell": np.arange(C, dtype=np.int32),
                    "farm": np.array(feats["farms"], dtype=object),
                    "cell_lat": ("cell", feats["cell_lat"]),
                    "cell_lon": ("cell", feats["cell_lon"])},
            attrs={"tag": tag, "source_run": source_run, "use_cf_lam": int(use_cf_lam),
                   "tier": "cerra" if tag != source_run else "forecast",
                   "region": args.region, "checkpoint": ckpt_desc,
                   "inputs": ", ".join(feats["feat_names"]),
                   "reconstruction": "P(farm,t) = sum_cell G[farm,cell]*cf(cell,t)"})
        out["lead_time"].attrs["units"] = "h"
        enc = {v: {"zlib": True, "complevel": 4} for v in ("cf", "G", "cap_cell")}
        stem = tag if tag == source_run else f"{tag}@{source_run}"
        out_path = args.out / f"cf_{args.region}_{stem}.nc"
        tmp = out_path.with_suffix(".nc.tmp")
        out.to_netcdf(tmp, format="NETCDF4", engine="netcdf4", encoding=enc)
        tmp.replace(out_path)

        te = split_coord == "test"
        print(f"  wrote {out_path}  (cf mean {np.nanmean(cf):.4f} max {np.nanmax(cf):.4f}; "
              f"{int(te.sum())} test inits)")


if __name__ == "__main__":
    main()
