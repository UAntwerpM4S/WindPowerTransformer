#!/usr/bin/env python3
"""INFERENCE: apply the trained cell-level model to forecast wind -> capacity-factor forecasts.

Reads the LAM inference directories directly (the same FORECAST_DIRS score_power_configs.py uses),
pulls the 6 wind variables at the region's cells, builds the 3 static features
[capacity, turbinecount, rated_ws], standardises with the TRAIN stats saved at training time,
runs the model over each init's full lead window, CLAMPS to [0,1], and writes ONE file per run:

    cf_<REGION>_<RUN>.nc
      cf          (init, lead_time, cell)   predicted capacity factor  <- the product
      G           (farm, cell)              capacity of farm f in cell c   (reconstruction op)
      cap_cell    (cell)                    total capacity per cell
      cell_lat/lon, valid_time

score_cf.py then reconstructs per-farm power  P(farm,t) = sum_cell G[farm,cell]*cf(cell,t)  and
scores it next to the LAM's direct forecast, the power curve, and persistence.

The model attends over the whole 3..36 h window, so an init is written ONLY if it covers every
lead (same rule as extract_cells.py); partial windows would shift the positional encoding.

Edit the SETTINGS block, then:  python infer_cf.py
Override runs/checkpoint on the CLI:  python infer_cf.py --runs HighCapacityGT=/path --ckpt ...
"""
from __future__ import annotations

import argparse
import os
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import torch
from scipy.spatial import cKDTree

from Transformer import TemporalTransformer
from Loader import WIND_VARS, STANDARDIZE, farm_rated_ws, _interp2d

# -------------------- SETTINGS --------------------
WPOWER_DIR = Path("/mnt/weatherloss/WindPower/data/WPDistr")

FORECAST_DIRS = {
    "HighCapacityGT": Path("/mnt/weatherloss/WindPower/inference/WPDistr/HighCapacityGT"),
    "VanillaPowerGT": Path("/mnt/weatherloss/WindPower/inference/WPDistr/VanillaPowerGT"),
}
REGION     = "BE"
LEAD_HOURS = list(range(3, 37, 3))                 # 3..36 h
OUT_DIR    = Path("/mnt/weatherloss/WindPowerTransformer/data/cf_forecasts")

CKPT     = "checkpoints/CERRAcell/CERRAcell_dim64_cells_in9_heads4_layers2_mlp4_lr0.0003_ep40.pt"
RUN_TAG  = "CERRAcell"                              # which artifacts/<tag>/cell_feature_stats.pkl
MODEL_DIM, N_HEADS, NUM_LAYERS, MLP_MULT = 64, 4, 2, 4   # MUST match the checkpoint

MATCH_TOL_KM = 1.0
# --------------------------------------------------

FORECAST_RE = re.compile(r"forecast_(\d{14})")


def to_180(lon):
    lon = np.asarray(lon, dtype=float)
    return ((lon + 180.0) % 360.0) - 180.0


def parse_init(path: Path) -> pd.Timestamp:
    return pd.to_datetime(FORECAST_RE.search(path.name).group(1), format="%Y%m%d%H%M%S")


def build_reconstruction(fc_lat, fc_lon, turbines, farms):
    """Verbatim from score_power_configs.py, plus per-cell turbine count (a static feature)."""
    coslat = np.cos(np.radians(float(fc_lat.mean())))
    tree = cKDTree(np.c_[to_180(fc_lon) * coslat, fc_lat])
    _, cell = tree.query(np.c_[to_180(turbines["longitude"]) * coslat,
                               turbines["latitude"].to_numpy()], k=1)
    t = turbines.assign(cell=cell.astype(int))
    cell_idx = np.sort(t["cell"].unique())
    cpos = {int(c): j for j, c in enumerate(cell_idx)}
    fpos = {f: i for i, f in enumerate(farms)}
    G = np.zeros((len(farms), cell_idx.size), dtype=np.float64)
    for (farm, c), cap in t.groupby(["farm", "cell"])["capacity_mw"].sum().items():
        G[fpos[farm], cpos[int(c)]] = cap
    cap_cell = t.groupby("cell")["capacity_mw"].sum().reindex(cell_idx).to_numpy()
    count_cell = t.groupby("cell").size().reindex(cell_idx).fillna(0).to_numpy().astype(np.float64)
    return cell_idx, G, cap_cell, count_cell


def grid_key(lat, lon):
    return (lat.size, round(float(lat[0]), 4), round(float(lon[-1]), 4))


def locate_cells(lat, lon, cell_lat, cell_lon):
    coslat = np.cos(np.radians(float(lat.mean())))
    tree = cKDTree(np.c_[to_180(lon) * coslat, lat])
    d, idx = tree.query(np.c_[to_180(cell_lon) * coslat, cell_lat], k=1)
    if (d * 111.32).max() > MATCH_TOL_KM:
        bad = int(np.argmax(d))
        raise ValueError(f"cell {bad} ({cell_lat[bad]:.4f},{cell_lon[bad]:.4f}) not in this grid")
    return idx.astype(int)


def static_features(G, cap_cell, count_cell, farms, farms_df, specs):
    """[capacity, turbinecount, capacity-weighted rated_ws] per cell — matches Loader.cell_static."""
    rws = farm_rated_ws(farms_df, specs)
    rws_vec = np.array([rws[f] for f in farms])
    with np.errstate(invalid="ignore"):
        ratedws_cell = (G * rws_vec[:, None]).sum(0) / G.sum(0)
    return np.stack([cap_cell, count_cell, ratedws_cell], axis=1).astype(np.float32)


def standardize(dyn, static, stats):
    """dyn (N,L,C,6) wind, static (C,3) — standardise ws + static in place with train stats."""
    feat = list(WIND_VARS)
    for v in STANDARDIZE:
        j = feat.index(v)
        dyn[:, :, :, j] = (dyn[:, :, :, j] - stats[v]["mean"]) / stats[v]["std"]
    static = static.copy()
    for j, v in enumerate(["capacity", "turbinecount", "rated_ws"]):
        static[:, j] = (np.nan_to_num(static[:, j], nan=stats[v]["mean"])
                        - stats[v]["mean"]) / stats[v]["std"]
    return dyn, static


@torch.no_grad()
def run_model(model, dyn, static, device):
    """dyn (N,L,C,6) + static (C,3) -> cf (N,L,C) clamped to [0,1]. One sample per (init,cell)."""
    N, L, C, _ = dyn.shape
    stat_b = np.broadcast_to(static[None, None], (N, L, C, static.shape[1]))
    X = np.concatenate([dyn, stat_b], axis=-1).astype(np.float32)          # (N,L,C,9)
    flat = X.transpose(0, 2, 1, 3).reshape(N * C, L, X.shape[-1])
    for s in range(flat.shape[0]):                                         # gap-fill per series
        if not np.isfinite(flat[s]).all():
            flat[s] = _interp2d(flat[s])
    out = np.empty((N * C, L), dtype=np.float32)
    model.eval()
    for s in range(0, flat.shape[0], 8192):
        xb = torch.from_numpy(flat[s:s + 8192]).to(device)
        out[s:s + 8192] = model(xb).cpu().numpy()
    return np.clip(out.reshape(N, C, L).transpose(0, 2, 1), 0.0, 1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=None, metavar="LABEL=DIR")
    ap.add_argument("--region", default=REGION, choices=["BE", "UK", "all"])
    ap.add_argument("--leads", type=int, nargs="+", default=LEAD_HOURS)
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--run_tag", default=RUN_TAG)
    ap.add_argument("--model_dim", type=int, default=MODEL_DIM)
    ap.add_argument("--n_heads", type=int, default=N_HEADS)
    ap.add_argument("--num_layers", type=int, default=NUM_LAYERS)
    ap.add_argument("--mlp_mult", type=int, default=MLP_MULT)
    ap.add_argument("--wpower-dir", type=Path, default=WPOWER_DIR)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--limit", type=int, default=None, help="only the first N inits (smoke test)")
    args = ap.parse_args()

    runs = {}
    if args.runs:
        for r in args.runs:
            label, d = r.split("=", 1)
            runs[label] = Path(d)
    else:
        runs = dict(FORECAST_DIRS)
    args.out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(os.path.join("artifacts", args.run_tag, "cell_feature_stats.pkl"), "rb") as f:
        stats = pickle.load(f)
    model = TemporalTransformer(input_dim=9, model_dim=args.model_dim, n_heads=args.n_heads,
                                num_layers=args.num_layers, mlp_mult=args.mlp_mult).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    print(f"loaded {args.ckpt}  ({sum(p.numel() for p in model.parameters())/1e3:.0f}k params)")

    farms_df = pd.read_csv(args.wpower_dir / "farms.csv")
    turbines = pd.read_csv(args.wpower_dir / "turbines.csv")
    specs = pd.read_csv(args.wpower_dir / "turbine_specs.csv")
    specs = specs.rename(columns={specs.columns[0]: "turbine_type"}).set_index("turbine_type")
    farms = (farms_df.farm.tolist() if args.region == "all"
             else farms_df[farms_df.region == args.region].farm.tolist())
    turbines = turbines[turbines.farm.isin(farms)]
    L = len(args.leads)
    print(f"{args.region}: {len(farms)} farms, "
          f"{farms_df.set_index('farm').loc[farms,'capacity_mw'].sum():.0f} MW")

    for label, d in runs.items():
        files = sorted(d.glob("forecast_*.nc"))
        if args.limit:
            files = files[:args.limit]
        if not files:
            print(f"\n{label}: no forecast_*.nc in {d} -- skipping")
            continue
        print(f"\n{'='*70}\n{label}: {len(files)} files\n{'='*70}")

        # canonical cells from the first file that has all wind vars
        cell_lat = cell_lon = G = cap_cell = count_cell = None
        for f in files:
            with xr.open_dataset(f) as ds:
                if all(v in ds for v in WIND_VARS):
                    cell_idx, G, cap_cell, count_cell = build_reconstruction(
                        ds["latitude"].values, ds["longitude"].values, turbines, farms)
                    cell_lat = ds["latitude"].values[cell_idx].astype(np.float64)
                    cell_lon = to_180(ds["longitude"].values[cell_idx])
                    break
        if G is None:
            print(f"  no file has all of {WIND_VARS} -- skipping")
            continue
        C = cell_idx.size
        static = static_features(G, cap_cell, count_cell, farms, farms_df, specs)
        print(f"  {C} cells, {cap_cell.sum():.0f} MW")

        idx_cache, inits, blocks, dropped = {}, [], [], 0
        for i, f in enumerate(files):
            init = parse_init(f)
            with xr.open_dataset(f) as ds:
                if not all(v in ds for v in WIND_VARS):
                    dropped += 1
                    continue
                key = grid_key(ds["latitude"].values, ds["longitude"].values)
                if key not in idx_cache:
                    idx_cache[key] = locate_cells(ds["latitude"].values, ds["longitude"].values,
                                                  cell_lat, cell_lon)
                cidx = idx_cache[key]
                fc_times = pd.DatetimeIndex(ds["time"].values)
                t2i = {t: j for j, t in enumerate(fc_times)}
                rows = [t2i.get(init + pd.Timedelta(hours=lh)) for lh in args.leads]
                if any(r is None for r in rows):
                    dropped += 1
                    continue
                rows = np.asarray(rows, dtype=int)
                blk = np.stack([ds[v].values[np.ix_(rows, cidx)] for v in WIND_VARS], axis=-1)
            inits.append(init)
            blocks.append(blk.astype(np.float32))                         # (L, C, 6)
            if i and i % 500 == 0:
                print(f"  read {i}/{len(files)} ({len(inits)} kept)", flush=True)

        if not inits:
            print("  nothing usable -- skipping")
            continue
        if dropped:
            print(f"  dropped {dropped} init(s) missing a wind var or a lead")

        dyn = np.stack(blocks)                                            # (N,L,C,6)
        dyn, static_z = standardize(dyn, static, stats)
        cf = run_model(model, dyn, static_z, device)                     # (N,L,C) in [0,1]

        init_ix = pd.DatetimeIndex(inits)
        leads = np.asarray(args.leads, dtype=np.int32)
        valid = (init_ix.values[:, None]
                 + leads.astype("timedelta64[h]")[None, :]).astype("datetime64[ns]")
        out = xr.Dataset(
            {"cf":         (("init", "lead_time", "cell"), cf),
             "G":          (("farm", "cell"), G),
             "cap_cell":   ("cell", cap_cell),
             "valid_time": (("init", "lead_time"), valid)},
            coords={"init": init_ix.values, "lead_time": leads,
                    "cell": np.arange(C, dtype=np.int32),
                    "farm": np.array(farms, dtype=object),
                    "cell_lat": ("cell", cell_lat), "cell_lon": ("cell", cell_lon)},
            attrs={"run": label, "region": args.region, "checkpoint": os.path.basename(args.ckpt),
                   "reconstruction": "P(farm,t) = sum_cell G[farm,cell]*cf(cell,t)",
                   "note": "cf is the model's predicted capacity factor on FORECAST wind, in [0,1]"})
        out["lead_time"].attrs["units"] = "h"
        enc = {v: {"zlib": True, "complevel": 4} for v in ("cf", "G", "cap_cell")}
        out_path = args.out / f"cf_{args.region}_{label}.nc"
        tmp = out_path.with_suffix(".nc.tmp")
        out.to_netcdf(tmp, format="NETCDF4", engine="netcdf4", encoding=enc)
        tmp.replace(out_path)
        print(f"  wrote {out_path}  ({len(inits)} inits x {L} leads x {C} cells)")
        print(f"  cf: mean {cf.mean():.4f}  sd {cf.std():.4f}  max {cf.max():.4f}")


if __name__ == "__main__":
    main()
