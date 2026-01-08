from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

FORECAST_DIR = Path("/mnt/weatherloss/WindPower/inference/CI/GraphTransformer")
POWER_DIR = Path("/mnt/weatherloss/WindPower/data/NorthSea/Power")
METADATA_CSV = POWER_DIR / "windfarm_metadata.csv"
OBS_CSV = POWER_DIR / "BE_UK_offshore_per_unit_3H_meanMW_shifted.csv"

OUT_DIR = Path("/mnt/weatherloss/WindPowerTransformer/data/OBS")

FNAME_RE = re.compile(r"forecast_(\d{14})\.nc$")  # forecast_YYYYmmddHHMMSS.nc


def load_belgian_farm_names(metadata_csv: Path) -> list[str]:
    df = pd.read_csv(metadata_csv)
    df = df[df["region"].astype(str).str.lower() == "belgium"].copy()
    if df.empty:
        raise RuntimeError("No Belgium rows found in windfarm_metadata.csv")
    df = df.drop_duplicates(subset=["farm"], keep="first")
    return df["farm"].astype(str).tolist()


def load_obs_csv(obs_csv: Path) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by UTC timestamps, with farm columns.
    """
    df = pd.read_csv(obs_csv)

    if "time" not in df.columns:
        raise ValueError(f"Observation CSV must contain a 'time' column: {obs_csv}")

    # Parse time as UTC-aware timestamps
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time").sort_index()

    # Ensure numeric columns
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def extract_init_str(fc_path: Path) -> str:
    m = FNAME_RE.search(fc_path.name)
    if not m:
        raise ValueError(f"Unrecognized forecast filename format: {fc_path.name}")
    return m.group(1)


def make_one_obs_file(fc_path: Path, obs_df: pd.DataFrame, belgian_farms: list[str]) -> Path:
    init_str = extract_init_str(fc_path)
    out_path = OUT_DIR / f"obs.{init_str}.nc"
    if out_path.exists():
        return out_path

    fc = xr.open_dataset(fc_path)

    if "time" not in fc.coords and "time" not in fc.variables:
        raise ValueError(f"{fc_path.name} has no 'time' coordinate")

    # Forecast times (naive datetime64[ns]) -> pandas UTC timestamps
    fc_times = pd.to_datetime(fc["time"].values)
    # Treat them as UTC (your CSV is explicitly +00:00)
    fc_times = fc_times.tz_localize("UTC")

    init_time = fc_times[0]
    lead_hours = ((fc_times - init_time) / pd.Timedelta(hours=1)).astype(np.float32).to_numpy()

    # Pull observations at exactly those times (reindex keeps all fc_times; missing -> NaN)
    obs_at_times = obs_df.reindex(fc_times)

    # Ensure we have all requested farm columns; missing farms -> NaN column
    data = []
    for farm in belgian_farms:
        if farm in obs_at_times.columns:
            series = obs_at_times[farm].to_numpy(dtype=np.float32)
        else:
            series = np.full(len(fc_times), np.nan, dtype=np.float32)
        data.append(series)

    # Stack to (windpark, lead_time)
    wp = np.stack(data, axis=0).astype(np.float32)

    ds_out = xr.Dataset(
        data_vars={
            "WP": (("windpark", "lead_time"), wp),
        },
        coords={
            "windpark": ("windpark", np.array(belgian_farms, dtype=object)),
            "lead_time": ("lead_time", lead_hours),
        },
        attrs={
            "description": "Wind power observations aligned to forecast lead times",
            "source_csv": str(OBS_CSV),
            "lead_time_units": "hours since forecast init time",
            "forecast_file": str(fc_path),
        },
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Optional compression (requires netcdf4)
    encoding = {"WP": {"zlib": True, "complevel": 4}}
    tmp = OUT_DIR / f".tmp_obs.{init_str}.nc"
    ds_out.to_netcdf(tmp, format="NETCDF4", engine="netcdf4", encoding=encoding)
    os.replace(tmp, out_path)

    return out_path


def main():
    belgian_farms = load_belgian_farm_names(METADATA_CSV)
    obs_df = load_obs_csv(OBS_CSV)

    fc_files = sorted(FORECAST_DIR.glob("forecast_*.nc"))
    if not fc_files:
        raise RuntimeError(f"No forecast_*.nc files found in {FORECAST_DIR}")

    written = 0
    for fc_path in fc_files:
        out = make_one_obs_file(fc_path, obs_df, belgian_farms)
        written += 1
        print(f"Wrote: {out}")

    print(f"Done. Wrote {written} obs files to: {OUT_DIR}")


if __name__ == "__main__":
    main()
