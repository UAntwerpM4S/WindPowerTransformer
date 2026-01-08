from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# ---- paths ----
FORECAST_DIR = Path("/mnt/weatherloss/WindPower/inference/CI/GraphTransformer")
METADATA_CSV = Path("/mnt/weatherloss/WindPower/data/NorthSea/Power/windfarm_metadata.csv")
OUT_DIR = Path("/mnt/weatherloss/WindPowerTransformer/data/FCS")

MODEL_NAME = "GraphTransformer"

# Variables we want in the output files
VARS = [
    "ws10",
    "ws100",
    "wdir10_sin",
    "wdir10_cos",
    "wdir_sin100",
    "wdir_cos100",
]

FNAME_RE = re.compile(r"forecast_(\d{14})\.nc$")  # forecast_YYYYmmddHHMMSS.nc


def load_belgian_farms(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["region"].astype(str).str.lower() == "belgium"].copy()
    if df.empty:
        raise RuntimeError("No Belgium rows found in windfarm_metadata.csv")

    # Prefer cell coordinates if available (these match the model grid cell)
    if "lat_cell" in df.columns and "lon_cell" in df.columns:
        df["lat_use"] = df["lat_cell"].astype(float)
        df["lon_use"] = df["lon_cell"].astype(float)
    else:
        df["lat_use"] = df["lat"].astype(float)
        df["lon_use"] = df["lon"].astype(float)

    # One row per farm
    df = df.drop_duplicates(subset=["farm"], keep="first").reset_index(drop=True)
    df["farm"] = df["farm"].astype(str)
    return df


def nearest_values_index(lat_vals: np.ndarray, lon_vals: np.ndarray, lat0: float, lon0: float) -> int:
    """
    Nearest neighbor on lat/lon.
    Uses simple squared distance in degrees (fine for small regional domain).
    """
    d2 = (lat_vals - lat0) ** 2 + (lon_vals - lon0) ** 2
    return int(np.argmin(d2))


def extract_one_forecast(fc_path: Path, farms: pd.DataFrame) -> Path:
    m = FNAME_RE.search(fc_path.name)
    if not m:
        raise ValueError(f"Unrecognized forecast filename format: {fc_path.name}")
    init_str = m.group(1)  # YYYYmmddHHMMSS

    out_path = OUT_DIR / f"fcs.{init_str}.nc"
    if out_path.exists():
        return out_path

    ds = xr.open_dataset(fc_path)

    # Basic checks
    missing = [v for v in VARS if v not in ds.variables]
    if missing:
        raise KeyError(f"{fc_path.name} is missing variables: {missing}")
    if "values" not in ds.dims or "time" not in ds.dims:
        raise ValueError(f"{fc_path.name} expected dims (time, values) but got {ds.dims}")

    lat = ds["latitude"].values.astype(np.float64)   # (values,)
    lon = ds["longitude"].values.astype(np.float64)  # (values,)

    # Match each wind farm to nearest 'values' index
    idxs = []
    names = []
    for _, r in farms.iterrows():
        i = nearest_values_index(lat, lon, float(r["lat_use"]), float(r["lon_use"]))
        idxs.append(i)
        names.append(r["farm"])

    # Subset and reshape: (time, windpark)
    sub = ds[VARS].isel(values=xr.DataArray(idxs, dims="windpark"))
    sub = sub.assign_coords(windpark=("windpark", np.array(names, dtype=object)))

    # Build lead_time (hours since init time)
    init_time = sub["time"].values[0]
    lead_hours = ((sub["time"].values - init_time) / np.timedelta64(1, "h")).astype(np.float32)

    sub = sub.assign_coords(lead_time=("time", lead_hours)).swap_dims({"time": "lead_time"}).drop_vars("time")

    # Add model dimension to match your loader’s .sel(model=..., windpark=...)
    sub = sub.expand_dims(model=[MODEL_NAME])

    # Ensure dimension order is consistent
    sub = sub.transpose("model", "windpark", "lead_time")

    # Write
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Optional compression (requires netcdf4)
    encoding = {v: {"zlib": True, "complevel": 4} for v in VARS}
    tmp = OUT_DIR / f".tmp_fcs.{init_str}.nc"
    sub.to_netcdf(tmp, format="NETCDF4", engine="netcdf4", encoding=encoding)
    os.replace(tmp, out_path)

    return out_path


def main():
    farms = load_belgian_farms(METADATA_CSV)

    fc_files = sorted(FORECAST_DIR.glob("forecast_*.nc"))
    if not fc_files:
        raise RuntimeError(f"No forecast_*.nc files found in {FORECAST_DIR}")

    written = 0
    for fc in fc_files:
        out = extract_one_forecast(fc, farms)
        written += 1
        print(f"Wrote: {out}")

    print(f"Done. Processed {written} forecast files into {OUT_DIR}")


if __name__ == "__main__":
    main()
