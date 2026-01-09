from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


INFERENCE_BASE_DIR = Path("/mnt/weatherloss/WindPower/inference/CI")
METADATA_CSV = Path("/mnt/weatherloss/WindPower/data/NorthSea/Power/windfarm_metadata.csv")
OUT_BASE_DIR = Path("/mnt/weatherloss/WindPowerTransformer/data/FCS")

MODELS = {
    "GraphTransformer": "GraphTransformer",
    "Transformer": "Transformer",
    "GNN": "GNN",
}

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

# ---------------------------
# Date window (EDIT THESE)
# Interpreted as UTC-like naive datetimes matching your filenames.
# Inclusive filtering by default.
# ---------------------------
START_DT = datetime.strptime("20240801000000", "%Y%m%d%H%M%S")
END_DT   = datetime.strptime("20250728210000", "%Y%m%d%H%M%S")


def load_belgian_farms(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["region"].astype(str).str.lower() == "belgium"].copy()

    if "lat_cell" in df.columns and "lon_cell" in df.columns:
        df["lat_use"] = df["lat_cell"].astype(float)
        df["lon_use"] = df["lon_cell"].astype(float)
    else:
        df["lat_use"] = df["lat"].astype(float)
        df["lon_use"] = df["lon"].astype(float)

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


def parse_init_dt_from_filename(p: Path) -> datetime | None:
    m = FNAME_RE.search(p.name)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y%m%d%H%M%S")


def in_window(dt: datetime, start: datetime, end: datetime, inclusive: bool = True) -> bool:
    if inclusive:
        return start <= dt <= end
    return start <= dt < end


def build_subset(ds: xr.Dataset, farms: pd.DataFrame, model_name: str) -> xr.Dataset:
    # Basic checks (kept lightweight, but you can re-add strict errors if you want)
    # missing = [v for v in VARS if v not in ds.variables]

    lat = ds["latitude"].values.astype(np.float64)   # (values,)
    lon = ds["longitude"].values.astype(np.float64)  # (values,)

    # Match each wind farm to nearest 'values' index
    idxs: list[int] = []
    names: list[str] = []
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
    sub = sub.expand_dims(model=[model_name])

    # Ensure dimension order is consistent
    sub = sub.transpose("model", "windpark", "lead_time")

    return sub


def extract_one_forecast(
    fc_path: Path,
    farms: pd.DataFrame,
    model_name: str,
    out_dir: Path,
) -> Path:
    m = FNAME_RE.search(fc_path.name)
    if not m:
        raise ValueError(f"Unrecognized forecast filename format: {fc_path.name}")
    init_str = m.group(1)  # YYYYmmddHHMMSS

    out_path = out_dir / f"fcs.{init_str}.nc"
    if out_path.exists():
        return out_path

    ds = xr.open_dataset(fc_path)
    try:
        sub = build_subset(ds, farms, model_name)
    finally:
        ds.close()

    out_dir.mkdir(parents=True, exist_ok=True)

    encoding = {v: {"zlib": True, "complevel": 4} for v in VARS}
    tmp = out_dir / f".tmp_fcs.{init_str}.nc"
    sub.to_netcdf(tmp, format="NETCDF4", engine="netcdf4", encoding=encoding)
    os.replace(tmp, out_path)

    return out_path


def process_model(
    model_name: str,
    model_dirname: str,
    farms: pd.DataFrame,
    start_dt: datetime,
    end_dt: datetime,
) -> int:
    forecast_dir = INFERENCE_BASE_DIR / model_dirname
    out_dir = OUT_BASE_DIR / model_name

    all_files = sorted(forecast_dir.glob("forecast_*.nc"))

    # Filter by init datetime parsed from filename
    fc_files: list[Path] = []
    skipped_badname = 0
    skipped_outside = 0

    for p in all_files:
        dt = parse_init_dt_from_filename(p)
        if dt is None:
            skipped_badname += 1
            continue
        if in_window(dt, start_dt, end_dt, inclusive=True):
            fc_files.append(p)
        else:
            skipped_outside += 1

    print(
        f"[{model_name}] Found {len(all_files)} files, "
        f"using {len(fc_files)} within {start_dt}..{end_dt} (inclusive). "
        f"Skipped {skipped_outside} outside window, {skipped_badname} bad names."
    )

    written = 0
    for fc in fc_files:
        out = extract_one_forecast(fc, farms, model_name=model_name, out_dir=out_dir)
        written += 1
        print(f"[{model_name}] Wrote: {out}")

    print(f"[{model_name}] Done. Processed {written} forecast files into {out_dir}")
    return written


def main():
    farms = load_belgian_farms(METADATA_CSV)

    total = 0
    for model_name, model_dirname in MODELS.items():
        total += process_model(
            model_name,
            model_dirname,
            farms,
            start_dt=START_DT,
            end_dt=END_DT,
        )

    print(f"All done. Total processed files: {total}. Output base: {OUT_BASE_DIR}")


if __name__ == "__main__":
    main()
