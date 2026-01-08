#!/usr/bin/env python3
import re
import pickle
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# ===================== CONFIG =====================
MODEL_NAME = "GraphTransformer"

FCS_DIR = Path("data/FCS")
OBS_DIR = Path("data/OBS")
TEST_DATES_PKL = Path("data/test_dates.pkl")

POWER_META_DIR = Path("/mnt/weatherloss/WindPower/data/NorthSea/Power")
COUNTS_PATH = POWER_META_DIR / "wind_farm_turbine_counts.csv"
SPECS_PATH  = POWER_META_DIR / "turbine_specs.csv"

OUT_DIR = Path("forecast_netcdfs_powercurve_obs")  # choose a writable dir
OUT_PREFIX = f"{MODEL_NAME}_PowerCurve_"

WS_VAR = "ws100"   # FCS wind speed
OBS_VAR = "WP"     # OBS truth variable
# ================================================


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_")


def ts14_to_datetime64(ts: str) -> np.datetime64:
    # YYYYMMDDHHMMSS -> datetime64[ns]
    dt = datetime.strptime(ts, "%Y%m%d%H%M%S")
    return np.datetime64(dt, "ns")


class TurbineSpec:
    def __init__(self, cut_in, rated_ws, cut_out, rated_power_mw):
        self.cut_in = float(cut_in)
        self.rated_ws = float(rated_ws)
        self.cut_out = float(cut_out)
        self.rated_power_mw = float(rated_power_mw)


def power_curve(ws: np.ndarray, spec: TurbineSpec) -> np.ndarray:
    """
    Piecewise cubic ramp to rated, then constant to cutout.
    Returns MW per turbine. Works on ws shape (...).
    """
    ws = np.asarray(ws, dtype=float)
    out = np.zeros_like(ws, dtype=np.float32)

    ramp = (ws >= spec.cut_in) & (ws < spec.rated_ws)
    out[ramp] = spec.rated_power_mw * ((ws[ramp] - spec.cut_in) / (spec.rated_ws - spec.cut_in)) ** 3

    rated = (ws >= spec.rated_ws) & (ws < spec.cut_out)
    out[rated] = spec.rated_power_mw

    return out


def load_specs(path: Path) -> dict[str, TurbineSpec]:
    df = pd.read_csv(path)
    key_col = "turbine_type (name-capacity-type)"
    if key_col not in df.columns:
        raise ValueError(f"{path} must contain column '{key_col}'")
    return {
        str(row[key_col]): TurbineSpec(
            cut_in=row["cut_in_ms"],
            rated_ws=row["rated_ws_ms"],
            cut_out=row["cut_out_ms"],
            rated_power_mw=row["rated_power_mw"],
        )
        for _, row in df.iterrows()
    }


def load_counts(path: Path) -> pd.DataFrame:
    """
    Index: farm name (must match windpark strings)
    Columns: turbine types (must match specs keys)
    """
    df = pd.read_csv(path)
    if "farm" not in df.columns:
        raise ValueError(f"{path} must contain a 'farm' column")
    df = df.set_index("farm")
    cols = [c for c in df.columns if c.lower() != "total"]
    return df[cols].astype(float)


def select_model_if_present(da: xr.DataArray, model_name: str) -> xr.DataArray:
    if "model" in da.dims:
        if "model" in da.coords and model_name in da["model"].values:
            return da.sel(model=model_name)
        return da.isel(model=0)
    return da


def assert_has_dims(da: xr.DataArray, required: tuple[str, ...], where: str):
    for d in required:
        if d not in da.dims:
            raise ValueError(f"{where}: expected dim '{d}' in {da.dims}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(TEST_DATES_PKL, "rb") as f:
        test_dates = pickle.load(f)
    if not test_dates:
        raise ValueError("test_dates.pkl is empty")

    specs = load_specs(SPECS_PATH)
    counts_df = load_counts(COUNTS_PATH)

    # Validate that all turbine types in counts exist in specs
    missing_specs = [t for t in counts_df.columns if t not in specs]
    if missing_specs:
        raise ValueError(
            "These turbine types are in wind_farm_turbine_counts.csv but missing in turbine_specs.csv:\n"
            + "\n".join(f"  - {t}" for t in missing_specs)
        )

    # Peek one file to get windpark ordering + lead length
    f0 = FCS_DIR / f"fcs.{test_dates[0]}.nc"
    if not f0.exists():
        raise FileNotFoundError(f"Missing first test FCS file: {f0}")

    with xr.open_dataset(f0) as ds0:
        if WS_VAR not in ds0:
            raise KeyError(f"{f0} missing variable '{WS_VAR}'")
        ws0 = select_model_if_present(ds0[WS_VAR], MODEL_NAME)

        assert_has_dims(ws0, ("windpark", "lead_time"), f"FCS {f0}")
        windparks = [str(x) for x in ws0["windpark"].values]
        lead_vals_ref = ws0["lead_time"].values
        T = int(ws0.sizes["lead_time"])

    # Ensure we have counts for all windparks we will process
    # (Fail fast with a helpful message if not.)
    missing_counts = [wp for wp in windparks if wp not in counts_df.index]
    if missing_counts:
        raise ValueError(
            "These windparks exist in FCS but are missing from wind_farm_turbine_counts.csv:\n"
            + "\n".join(f"  - {w}" for w in missing_counts)
        )

    # Build counts matrix aligned to FCS windpark order: (W, n_types)
    counts_aligned = counts_df.reindex(windparks)
    if counts_aligned.isnull().any().any():
        bad = counts_aligned[counts_aligned.isnull().any(axis=1)].index.tolist()
        raise ValueError(f"Counts reindex produced NaNs for windparks: {bad}")
    counts_mat = counts_aligned.to_numpy(dtype=np.float32)  # (W, K)

    turbine_types = list(counts_aligned.columns)  # K
    W = len(windparks)

    # Allocate arrays for all dates, all parks
    N = len(test_dates)
    forecasts_all = np.zeros((N, W, T), dtype=np.float32)
    truths_all    = np.zeros((N, W, T), dtype=np.float32)
    dates_all     = np.array([ts14_to_datetime64(ts) for ts in test_dates], dtype="datetime64[ns]")

    for i, ts in enumerate(tqdm(test_dates, desc="Building forecast+truth")):
        fcs_path = FCS_DIR / f"fcs.{ts}.nc"
        obs_path = OBS_DIR / f"obs.{ts}.nc"
        if not fcs_path.exists():
            raise FileNotFoundError(f"Missing FCS file: {fcs_path}")
        if not obs_path.exists():
            raise FileNotFoundError(f"Missing OBS file: {obs_path}")

        # --- FCS ---
        with xr.open_dataset(fcs_path) as ds_fcs:
            if WS_VAR not in ds_fcs:
                raise KeyError(f"{fcs_path} missing variable '{WS_VAR}'")

            ws = select_model_if_present(ds_fcs[WS_VAR], MODEL_NAME)
            assert_has_dims(ws, ("windpark", "lead_time"), f"FCS {fcs_path}")

            # Enforce same ordering and lead length
            wp_now = [str(x) for x in ws["windpark"].values]
            if wp_now != windparks:
                raise ValueError(f"{fcs_path}: windpark ordering differs from first file")
            if int(ws.sizes["lead_time"]) != T:
                raise ValueError(f"{fcs_path}: lead_time length differs ({ws.sizes['lead_time']} vs {T})")

            ws_np = ws.transpose("windpark", "lead_time").values.astype(np.float32)  # (W, T)

            # Compute power forecast (W, T) by summing turbine types
            total = np.zeros((W, T), dtype=np.float32)
            for k, tname in enumerate(turbine_types):
                spec = specs[tname]
                pc = power_curve(ws_np, spec)           # (W, T) MW per turbine
                total += pc * counts_mat[:, k:k+1]      # broadcast (W,1)

            forecasts_all[i, :, :] = total

        # --- OBS ---
        with xr.open_dataset(obs_path) as ds_obs:
            if OBS_VAR not in ds_obs:
                raise KeyError(f"{obs_path} missing variable '{OBS_VAR}'")

            wp = ds_obs[OBS_VAR]
            assert_has_dims(wp, ("windpark", "lead_time"), f"OBS {obs_path}")

            wp_now = [str(x) for x in wp["windpark"].values]
            if wp_now != windparks:
                raise ValueError(f"{obs_path}: windpark ordering differs from FCS/first file")
            if int(wp.sizes["lead_time"]) != T:
                raise ValueError(f"{obs_path}: lead_time length differs ({wp.sizes['lead_time']} vs {T})")

            truths_all[i, :, :] = wp.transpose("windpark", "lead_time").values.astype(np.float32)

    # Output: one file per windpark, like your forecast_netcdfs schema
    lead_time = np.arange(T, dtype=np.int32)  # step index 0..T-1 (matches your existing files)

    for wi, farm in enumerate(windparks):
        ds_out = xr.Dataset(
            {
                "forecast": (("date", "lead_time"), forecasts_all[:, wi, :]),
                "truth":    (("date", "lead_time"), truths_all[:, wi, :]),
            },
            coords={"date": dates_all, "lead_time": lead_time},
            attrs={
                "windpark": farm,
                "model_name": MODEL_NAME,
                "ws_variable": WS_VAR,
                "obs_variable": OBS_VAR,
                "lead_time_original_values": np.array(lead_vals_ref).astype(str).tolist(),
                "note": "forecast is power (MW) computed from FCS ws100 via simple power curve + turbine counts; truth from OBS WP",
            },
        )

        out_path = OUT_DIR / f"{OUT_PREFIX}{safe_name(farm)}.nc"
        ds_out.to_netcdf(out_path)
        ds_out.close()
        print(f"✅ Wrote {out_path}")

    print("🎉 Done.")


if __name__ == "__main__":
    main()
