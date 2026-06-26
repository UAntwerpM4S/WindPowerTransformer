#!/usr/bin/env python3
"""
PowerCurve baseline for ONE weather model, on the same test set as Testloop.py.

For each forecast_<init>.nc (a 13-step window) we take ws100 at each windpark's
forecast-grid cell, push it through an analytic turbine power curve weighted by
that farm's turbine counts, and sum to a farm-total power forecast. Truth is the
observed `power` from the CERRA zarr at the valid times (identical to Testloop).

Output: PC_FCS/<MODEL>/<MODEL>_PowerCurve_<park>.nc with forecast/truth over
(date, lead_time = step index 0..12), so verify.py can plot it against the
transformer (TEST_FCS/...) on the same axes.
"""
import os
import re
import glob
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from Loader import load_obs_csv

# ======== CONFIG ========
MODELS = [                        # must match the MODELS tested in Testloop.py
    "RegularWeather",
    "VanillaPowerGT",
    "VanillaPowerTF",
    "WindHeavyTinyPower",
    "WindHeavyVanillaPower",
    "WindWeather",
]

FCS_BASE = "/mnt/weatherloss/WindPower/inference/WindAI"
ZARR_PATH = "/mnt/weatherloss/WindPower/data/WindAI/Anemoidatasets/New_Cerra_A_large.zarr"

POWER_META_DIR = Path("/mnt/weatherloss/WindPower/data/NorthSea/Power")
COUNTS_PATH = POWER_META_DIR / "wind_farm_turbine_counts.csv"
SPECS_PATH = POWER_META_DIR / "turbine_specs.csv"
METADATA_PATH = POWER_META_DIR / "windfarm_metadata.csv"

TEST_START = "2024-08-01"
TEST_END = "2025-07-31 23:00"

WS_VAR = "ws100"
OUT_DIR = Path("PC_FCS")
# ========================

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


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_")


class TurbineSpec:
    def __init__(self, cut_in, rated_ws, cut_out, rated_power_mw):
        self.cut_in = float(cut_in)
        self.rated_ws = float(rated_ws)
        self.cut_out = float(cut_out)
        self.rated_power_mw = float(rated_power_mw)


def power_curve(ws: np.ndarray, spec: TurbineSpec) -> np.ndarray:
    ws = np.asarray(ws, dtype=float)
    out = np.zeros_like(ws, dtype=np.float32)
    ramp = (ws >= spec.cut_in) & (ws < spec.rated_ws)
    denom = (spec.rated_ws**3 - spec.cut_in**3)
    a = 1 / denom
    b = spec.cut_in**3 / denom
    out[ramp] = spec.rated_power_mw * (a * ws[ramp]**3 - b)
    out[(ws >= spec.rated_ws) & (ws < spec.cut_out)] = spec.rated_power_mw
    return out


def load_specs(path: Path) -> dict:
    df = pd.read_csv(path)
    key = "turbine_type (name-capacity-type)"
    return {
        str(r[key]): TurbineSpec(r["cut_in_ms"], r["rated_ws_ms"], r["cut_out_ms"], r["rated_power_mw"])
        for _, r in df.iterrows()
    }


def load_counts(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).set_index("farm")
    cols = [c for c in df.columns if c.lower() != "total"]
    return df[cols].astype(float)


def list_forecasts(model):
    files = []
    for f in glob.glob(os.path.join(FCS_BASE, model, "forecast_*.nc")):
        m = TS_RE.search(os.path.basename(f))
        if not m:
            continue
        init = datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
        if pd.Timestamp(TEST_START) <= pd.Timestamp(init) <= pd.Timestamp(TEST_END):
            files.append((init, f))
    return [f for _, f in sorted(files)]


def nearest_cell(lats, lons, lat0, lon0) -> int:
    return int(np.argmin((lats - lat0) ** 2 + (lons - lon0) ** 2))


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


def run_model(model, specs, counts, turbine_types, obs_series, meta):
    files = list_forecasts(model)
    if not files:
        print(f"⚠️ No forecast files for {model} in test range, skipping.")
        return
    print(f"\n=== {model}: {len(files)} forecast inits ===")

    out_dir = OUT_DIR / model
    out_dir.mkdir(parents=True, exist_ok=True)

    with xr.open_dataset(files[0]) as ds0:
        flat = ds0["latitude"].values
        flon = ds0["longitude"].values
        n_steps = ds0.sizes["time"]
    park_cell = {
        wp: nearest_cell(flat, flon,
                         float(meta.loc[wp, "cerra_grid_lat"]),
                         float(meta.loc[wp, "cerra_grid_lon"]))
        for wp in windparks
    }
    lead_time = np.arange(n_steps, dtype=np.int32)  # step index; verify.py *3 -> hours

    acc = {wp: {"fc": [], "truth": [], "init": []} for wp in windparks}
    for fp in tqdm(files, desc=f"power curve {model}"):
        with xr.open_dataset(fp) as ds:
            ws = ds[WS_VAR].values                      # (T, cells)
            tidx = pd.DatetimeIndex(ds["time"].values)  # (T,) naive
        for wp in windparks:
            ws_wp = ws[:, park_cell[wp]]                 # (T,)
            total = np.zeros(n_steps, dtype=np.float32)
            row = counts.loc[wp]
            for tname in turbine_types:
                n = float(row[tname])
                if n > 0:
                    total += power_curve(ws_wp, specs[tname]) * n
            acc[wp]["fc"].append(total)
            acc[wp]["truth"].append(obs_series[wp].reindex(tidx).to_numpy(dtype=np.float32))
            acc[wp]["init"].append(tidx[0])

    for wp in windparks:
        fc = np.stack(acc[wp]["fc"])                     # (N, T)
        truth = np.stack(acc[wp]["truth"])               # (N, T)
        init_dates = np.array(acc[wp]["init"], dtype="datetime64[ns]")

        ds_out = xr.Dataset(
            {
                "forecast": (("date", "lead_time"), fc.astype(np.float32)),
                "truth": (("date", "lead_time"), truth.astype(np.float32)),
            },
            coords={"date": init_dates, "lead_time": lead_time},
            attrs={"windpark": wp, "weather_model": model, "ws_variable": WS_VAR},
        )
        ds_out.to_netcdf(out_dir / f"{model}_PowerCurve_{safe_name(wp)}.nc")
    print(f"  💾 wrote {len(windparks)} parks -> {out_dir}/{model}_PowerCurve_*.nc")


def main():
    specs = load_specs(SPECS_PATH)
    counts_df = load_counts(COUNTS_PATH)
    turbine_types = list(counts_df.columns)
    counts = counts_df.reindex(windparks).fillna(0.0)  # (parks x types)

    meta = pd.read_csv(METADATA_PATH).set_index("farm")
    obs_series = build_obs_series()

    for model in MODELS:
        run_model(model, specs, counts, turbine_types, obs_series, meta)


if __name__ == "__main__":
    main()
