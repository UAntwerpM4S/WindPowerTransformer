#!/usr/bin/env python3
"""
Test CERRA-trained transformers on ONE weather model's real forecasts.

Each forecast_<init>.nc is already one 13-step pseudo-forecast:
    dims (time=13, values=<cells>), with time = init, init+3h, ..., init+36h
    data_vars include the 6 inputs by name + per-cell latitude/longitude.

Per windpark:
  - map the park to its nearest forecast-grid cell,
  - pull the 6 weather inputs over the 13 steps -> X=(13,6),
  - standardize ws10/ws100 with the CERRA TRAIN stats (artifacts/CERRA/<park>),
  - predict power with checkpoints/CERRA/...<park>...pt,
  - score against observed `power` from the CERRA zarr at the valid times.

Output: TEST_FCS_<MODEL>/<park>.nc with forecast/truth over (init, lead_time),
same format as TEST_CERRA so the two are directly comparable.
"""
import os
import re
import glob
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import xarray as xr
from tqdm import tqdm

from Transformer import TemporalTransformer
from Loader import (
    DEFAULT_FEATURES, STANDARDIZE_VARS, stats_path,
    _open_cerra, _cerra_cell_series, _interp2d_time_lastdim,
)

# ======== CONFIG ========
MODEL = "VanillaPowerGT"          # the weather model whose forecasts we test
RUN_TAG = "CERRA"                 # checkpoint / stats tag from training
INPUT_DIM = 6
MODEL_DIM = 128
N_HEADS = 4
NUM_LAYERS = 4
MLP_MULT = 4
BATCH_SIZE = 64

FCS_BASE = "/mnt/weatherloss/WindPower/inference/WindAI"
ZARR_PATH = "/mnt/weatherloss/WindPower/data/WindAI/Anemoidatasets/New_Cerra_A_large.zarr"
METADATA_PATH = "/mnt/weatherloss/WindPower/data/NorthSea/Power/windfarm_metadata.csv"

TEST_START = "2024-08-01"
TEST_END = "2025-07-31 23:00"

OUT_DIR = "TEST_FCS"   # verify.py reads {OUT_DIR}/{MODEL}_dim{D}_heads{H}_layers{L}_*.nc
# ========================

FEATURES = list(DEFAULT_FEATURES)
TS_RE = re.compile(r"forecast_(\d{14})\.nc$")

windparks = [
    "Belwind Phase 1",
    "Thorntonbank - C-Power - Area NE",
    "Thorntonbank - C-Power - Area SW",
    "Mermaid Offshore WP",
    "Nobelwind Offshore Windpark",
    "Norther Offshore WP",
    "Northwester 2",
    "Northwind",
    "Rentel Offshore WP",
    "Seastar Offshore WP",
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_")


def find_ckpt(windpark: str):
    pat = os.path.join(
        "checkpoints", RUN_TAG,
        f"{RUN_TAG}_dim{MODEL_DIM}_park{windpark}_in{INPUT_DIM}_"
        f"heads{N_HEADS}_layers{NUM_LAYERS}_*.pt",
    )
    matches = sorted(glob.glob(pat))
    return matches[0] if matches else None


def load_stats(windpark: str) -> dict:
    p = stats_path(RUN_TAG, windpark)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing train stats {p}. Train this park first.")
    with open(p, "rb") as f:
        return pickle.load(f)


def list_forecasts():
    files = []
    for f in glob.glob(os.path.join(FCS_BASE, MODEL, "forecast_*.nc")):
        m = TS_RE.search(os.path.basename(f))
        if not m:
            continue
        init = datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
        if pd.Timestamp(TEST_START) <= pd.Timestamp(init) <= pd.Timestamp(TEST_END):
            files.append((init, f))
    return [f for _, f in sorted(files)]


def nearest_cell(lats, lons, lat0, lon0):
    """Return (cell_index, distance_in_degrees) of the closest grid cell."""
    d2 = (lats - lat0) ** 2 + (lons - lon0) ** 2
    i = int(np.argmin(d2))
    return i, float(np.sqrt(d2[i]))


def standardize_and_impute(X, stats):
    """X: (N, T, F) -> standardize ws10/ws100, then impute NaNs per sample."""
    X = X.astype(np.float32).copy()
    for v in STANDARDIZE_VARS:
        if v in FEATURES:
            fpos = FEATURES.index(v)
            mu, sd = float(stats[v]["mean"]), float(stats[v]["std"])
            X[:, :, fpos] = (X[:, :, fpos] - mu) / max(sd, 1e-6)
    for i in range(X.shape[0]):
        X[i] = _interp2d_time_lastdim(X[i])
    return X


@torch.no_grad()
def predict(model, X):
    """X: (N, T, F) -> (N, T)."""
    model.eval()
    out = []
    for s in range(0, len(X), BATCH_SIZE):
        xb = torch.from_numpy(X[s:s + BATCH_SIZE]).to(device)
        out.append(model(xb).cpu().numpy())
    return np.concatenate(out, 0) if out else np.zeros((0, X.shape[1]), np.float32)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    files = list_forecasts()
    if not files:
        raise SystemExit(f"No forecast files for {MODEL} in {TEST_START}..{TEST_END}")
    print(f"{MODEL}: {len(files)} forecast inits in test range")

    # forecast-grid cell per park (from the first file's lat/lon + metadata)
    meta = pd.read_csv(METADATA_PATH).set_index("farm")
    with xr.open_dataset(files[0]) as ds0:
        flat = ds0["latitude"].values
        flon = ds0["longitude"].values
        n_steps = ds0.sizes["time"]
    lead_time = np.arange(n_steps, dtype=np.int32)  # step index; verify.py *3 -> hours

    print("park -> nearest forecast cell:")
    park_cell = {}
    for wp in windparks:
        idx, dist = nearest_cell(
            flat, flon,
            float(meta.loc[wp, "cerra_grid_lat"]),
            float(meta.loc[wp, "cerra_grid_lon"]),
        )
        park_cell[wp] = idx
        print(f"  {wp:<34} cell={idx:>6}  dist={dist:.6f}° (~{dist * 111:.2f} km)")

    # observed power series per park (truth) from the CERRA zarr
    obs_series = {}
    with _open_cerra(ZARR_PATH) as dsz:
        var_names = list(dsz.attrs["variables"])
        pidx = var_names.index("power")
        for wp in windparks:
            dates, _, series = _cerra_cell_series(dsz, METADATA_PATH, wp, ensemble=0)
            obs_series[wp] = pd.Series(series[:, pidx], index=dates.tz_localize(None))

    # accumulate inputs/truth per park over all forecast files (single I/O pass)
    acc = {wp: {"X": [], "truth": [], "init": []} for wp in windparks}
    for fp in tqdm(files, desc="reading forecasts"):
        with xr.open_dataset(fp) as ds:
            arrs = {v: ds[v].values for v in FEATURES}      # each (T, cells)
            tidx = pd.DatetimeIndex(ds["time"].values)      # (T,) naive
        for wp in windparks:
            c = park_cell[wp]
            X = np.stack([arrs[v][:, c] for v in FEATURES], axis=-1)  # (T, F)
            y = obs_series[wp].reindex(tidx).to_numpy(dtype=np.float32)  # (T,)
            acc[wp]["X"].append(X.astype(np.float32))
            acc[wp]["truth"].append(y)
            acc[wp]["init"].append(tidx[0])

    # per-park: standardize, predict, save
    for wp in windparks:
        ckpt = find_ckpt(wp)
        if ckpt is None:
            print(f"⚠️ No checkpoint for {wp}, skipping.")
            continue

        X = np.stack(acc[wp]["X"])                    # (N, T, F)
        truth = np.stack(acc[wp]["truth"])            # (N, T)
        init_dates = np.array(acc[wp]["init"], dtype="datetime64[ns]")

        X = standardize_and_impute(X, load_stats(wp))

        model = TemporalTransformer(
            input_dim=INPUT_DIM, model_dim=MODEL_DIM,
            n_heads=N_HEADS, num_layers=NUM_LAYERS, mlp_mult=MLP_MULT,
        ).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))

        preds = predict(model, X)                     # (N, T)

        out = xr.Dataset(
            {
                "forecast": (("date", "lead_time"), preds.astype(np.float32)),
                "truth": (("date", "lead_time"), truth.astype(np.float32)),
            },
            coords={"date": init_dates, "lead_time": lead_time},
            attrs={"windpark": wp, "weather_model": MODEL, "run_tag": RUN_TAG},
        )
        out_nc = os.path.join(
            OUT_DIR,
            f"{MODEL}_dim{MODEL_DIM}_heads{N_HEADS}_layers{NUM_LAYERS}_{safe_name(wp)}.nc",
        )
        out.to_netcdf(out_nc)
        print(f"💾 {out_nc}  ({len(init_dates)} inits)")


if __name__ == "__main__":
    main()
