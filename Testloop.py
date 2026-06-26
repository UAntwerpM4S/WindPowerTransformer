#!/usr/bin/env python3
"""
Test CERRA-trained transformers on the real forecasts of one or more weather
models.

The transformer checkpoints are trained on CERRA and are model-independent, so
they (and the CERRA observed-power truth) are loaded ONCE and reused across every
weather model in MODELS.

Each forecast_<init>.nc is already one 13-step pseudo-forecast:
    dims (time=13, values=<cells>), time = init, init+3h, ..., init+36h
    data_vars include the 6 inputs by name + per-cell latitude/longitude.

Per model, per windpark:
  - map the park to its nearest forecast-grid cell,
  - pull the 6 inputs over the 13 steps -> X=(13,6),
  - standardize ws10/ws100 with the CERRA TRAIN stats (artifacts/CERRA/<park>),
  - predict power with checkpoints/CERRA/...<park>...pt,
  - score against observed `power` from the CERRA zarr at the valid times.

Output: TEST_FCS/<MODEL>_dim{D}_heads{H}_layers{L}_<park>.nc  (verify.py format).
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
    _interp2d_time_lastdim, load_obs_csv,
)

# ======== CONFIG ========
MODELS = [
    "RegularWeather",
    "VanillaPowerGT",
    "VanillaPowerTF",
    "WindHeavyTinyPower",
    "WindHeavyVanillaPower",
    "WindWeather",
]
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


def list_forecasts(model: str):
    files = []
    for f in glob.glob(os.path.join(FCS_BASE, model, "forecast_*.nc")):
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


def build_obs_series():
    """Per-farm observed power (truth) from the obs CSV. Model-independent."""
    obs_df = load_obs_csv()
    obs = {}
    for wp in windparks:
        if wp in obs_df.columns:
            s = obs_df[wp]
            obs[wp] = pd.Series(s.to_numpy(), index=s.index.tz_localize(None))
        else:
            print(f"⚠️ farm {wp!r} not in obs CSV — truth will be all-NaN.")
            obs[wp] = pd.Series(dtype="float64")
    return obs


def load_park_models():
    """Load each park's checkpoint + stats once (shared across all weather models)."""
    state = {}
    for wp in windparks:
        ckpt = find_ckpt(wp)
        if ckpt is None:
            print(f"⚠️ No checkpoint for {wp}, it will be skipped.")
            continue
        model = TemporalTransformer(
            input_dim=INPUT_DIM, model_dim=MODEL_DIM,
            n_heads=N_HEADS, num_layers=NUM_LAYERS, mlp_mult=MLP_MULT,
        ).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        state[wp] = {"model": model, "stats": load_stats(wp)}
    return state


def run_model(model_name, park_state, obs_series, meta):
    files = list_forecasts(model_name)
    if not files:
        print(f"⚠️ No forecast files for {model_name} in test range, skipping.")
        return
    print(f"\n=== {model_name}: {len(files)} forecast inits ===")

    with xr.open_dataset(files[0]) as ds0:
        flat = ds0["latitude"].values
        flon = ds0["longitude"].values
        n_steps = ds0.sizes["time"]
    lead_time = np.arange(n_steps, dtype=np.int32)  # step index; verify.py *3 -> hours

    park_cell, max_dist = {}, 0.0
    for wp in park_state:
        idx, dist = nearest_cell(flat, flon,
                                 float(meta.loc[wp, "cerra_grid_lat"]),
                                 float(meta.loc[wp, "cerra_grid_lon"]))
        park_cell[wp] = idx
        max_dist = max(max_dist, dist)
    print(f"  mapped {len(park_cell)} parks to forecast cells (max dist {max_dist:.6f}°)")

    acc = {wp: {"X": [], "truth": [], "init": []} for wp in park_state}
    for fp in tqdm(files, desc=f"reading {model_name}"):
        with xr.open_dataset(fp) as ds:
            arrs = {v: ds[v].values for v in FEATURES}      # each (T, cells)
            tidx = pd.DatetimeIndex(ds["time"].values)
        for wp in park_state:
            c = park_cell[wp]
            X = np.stack([arrs[v][:, c] for v in FEATURES], axis=-1)
            y = obs_series[wp].reindex(tidx).to_numpy(dtype=np.float32)
            acc[wp]["X"].append(X.astype(np.float32))
            acc[wp]["truth"].append(y)
            acc[wp]["init"].append(tidx[0])

    for wp in park_state:
        X = standardize_and_impute(np.stack(acc[wp]["X"]), park_state[wp]["stats"])
        truth = np.stack(acc[wp]["truth"])
        init_dates = np.array(acc[wp]["init"], dtype="datetime64[ns]")
        preds = predict(park_state[wp]["model"], X)

        out = xr.Dataset(
            {
                "forecast": (("date", "lead_time"), preds.astype(np.float32)),
                "truth": (("date", "lead_time"), truth.astype(np.float32)),
            },
            coords={"date": init_dates, "lead_time": lead_time},
            attrs={"windpark": wp, "weather_model": model_name, "run_tag": RUN_TAG},
        )
        out_nc = os.path.join(
            OUT_DIR,
            f"{model_name}_dim{MODEL_DIM}_heads{N_HEADS}_layers{NUM_LAYERS}_{safe_name(wp)}.nc",
        )
        out.to_netcdf(out_nc)
    print(f"  💾 wrote {len(park_state)} parks -> {OUT_DIR}/{model_name}_*.nc")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    meta = pd.read_csv(METADATA_PATH).set_index("farm")

    park_state = load_park_models()
    if not park_state:
        raise SystemExit("No checkpoints found under checkpoints/CERRA — train first.")
    obs_series = build_obs_series()

    for model_name in MODELS:
        run_model(model_name, park_state, obs_series, meta)


if __name__ == "__main__":
    main()
