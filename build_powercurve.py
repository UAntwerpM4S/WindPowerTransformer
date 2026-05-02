#!/usr/bin/env python3
import re
import pickle
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

MODEL_NAMES = ["VanillaPower", "NoPower", "Synthetic"]

FCS_BASE_DIR   = Path("data/FCS")
OBS_DIR        = Path("data/OBS")
TEST_DATES_PKL = Path("data/test_dates.pkl")

POWER_META_DIR = Path("/mnt/weatherloss/WindPower/data/NorthSea/Power")
COUNTS_PATH    = POWER_META_DIR / "wind_farm_turbine_counts.csv"
SPECS_PATH     = POWER_META_DIR / "turbine_specs.csv"
METADATA_PATH  = POWER_META_DIR / "windfarm_metadata.csv"

CERRA_PATH = Path("/mnt/weatherloss/WindPower/data/EGU26/Anemoidatasets/New_Cerra_A_large.zarr")

OUT_DIR = Path("PC_FCS")

WS_VAR  = "ws100"
OBS_VAR = "WP"


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_")


def ts14_to_datetime64(ts: str) -> np.datetime64:
    return np.datetime64(datetime.strptime(ts, "%Y%m%d%H%M%S"), "ns")


def ts14_to_utc(ts: str) -> pd.Timestamp:
    return pd.Timestamp(datetime.strptime(ts, "%Y%m%d%H%M%S")).tz_localize("UTC")


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


def load_specs(path: Path) -> dict[str, TurbineSpec]:
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


def select_model_if_present(da: xr.DataArray, model_name: str) -> xr.DataArray:
    return da.sel(model=model_name) if "model" in da.dims else da


def lead_vals_to_hours(lead_vals: np.ndarray) -> list[int]:
    sample = lead_vals[0]
    if isinstance(sample, np.timedelta64):
        return [int(v / np.timedelta64(1, "h")) for v in lead_vals]
    if hasattr(sample, "total_seconds"):
        return [int(v.total_seconds() / 3600) for v in lead_vals]
    return [int(v) for v in lead_vals]


def build_windpark_cerra_indices(
    metadata_path: Path,
    cerra_lat: np.ndarray,
    cerra_lon: np.ndarray,
    windparks: list[str],
) -> list[int | None]:
    """Map each windpark name to a flat CERRA cell index (None if not found in metadata)."""
    meta = pd.read_csv(metadata_path)
    cerra_keys = {
        (round(la, 6), round(lo, 6)): i
        for i, (la, lo) in enumerate(zip(cerra_lat, cerra_lon))
    }
    farm_to_cerra: dict[str, int] = {}
    for _, row in meta.iterrows():
        key = (round(row["cerra_grid_lat"], 6), round(row["cerra_grid_lon"], 6))
        if key in cerra_keys:
            farm_to_cerra[str(row["farm"])] = cerra_keys[key]
    return [farm_to_cerra.get(wp) for wp in windparks]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    test_dates = pickle.load(open(TEST_DATES_PKL, "rb"))
    specs = load_specs(SPECS_PATH)
    counts_df = load_counts(COUNTS_PATH)

    ds_cerra = xr.open_zarr(CERRA_PATH, consolidated=False)
    cerra_vars = list(ds_cerra.attrs["variables"])
    cerra_lat = ds_cerra["latitudes"].values
    cerra_lon = ds_cerra["longitudes"].values
    cerra_dates = pd.to_datetime(ds_cerra["dates"].values).tz_localize("UTC")
    cerra_date_to_idx = {d: i for i, d in enumerate(cerra_dates)}
    cerra_power_idx = cerra_vars.index("power")

    for model_name in MODEL_NAMES:
        fcs_dir = FCS_BASE_DIR / model_name
        out_dir = OUT_DIR / model_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_prefix = f"{model_name}_PowerCurve_"

        f0 = fcs_dir / f"fcs.{test_dates[0]}.nc"
        with xr.open_dataset(f0) as ds0:
            ws0 = select_model_if_present(ds0[WS_VAR], model_name)
            windparks = [str(x) for x in ws0["windpark"].values]
            lead_vals_ref = ws0["lead_time"].values
            T = int(ws0.sizes["lead_time"])

        counts_mat = counts_df.reindex(windparks).fillna(0).to_numpy(dtype=np.float32)
        turbine_types = list(counts_df.columns)
        W = len(windparks)

        lead_hours = lead_vals_to_hours(lead_vals_ref)
        cerra_cell_indices = build_windpark_cerra_indices(METADATA_PATH, cerra_lat, cerra_lon, windparks)
        valid_wis = np.array([wi for wi, ci in enumerate(cerra_cell_indices) if ci is not None], dtype=int)
        valid_cis = np.array([ci for ci in cerra_cell_indices if ci is not None], dtype=int)
        n_matched = len(valid_wis)
        print(f"[{model_name}] {n_matched}/{W} windparks matched to CERRA cells")

        needed_valid = sorted({
            ts14_to_utc(ts) + pd.Timedelta(hours=lh)
            for ts in test_dates for lh in lead_hours
        })
        needed_in_cerra = [vt for vt in needed_valid if vt in cerra_date_to_idx]
        if needed_in_cerra:
            cerra_bulk = ds_cerra["data"].isel(
                time=[cerra_date_to_idx[vt] for vt in needed_in_cerra],
                variable=cerra_power_idx,
                ensemble=0,
            ).values  # (n_valid_times, n_cells)
            cerra_cache = {vt.isoformat(): cerra_bulk[i] for i, vt in enumerate(needed_in_cerra)}
        else:
            cerra_cache = {}

        N = len(test_dates)
        forecasts_all = np.zeros((N, W, T), dtype=np.float32)
        truths_all    = np.zeros((N, W, T), dtype=np.float32)
        cerra_all     = np.full((N, W, T), np.nan, dtype=np.float32)
        dates_all     = np.array([ts14_to_datetime64(ts) for ts in test_dates], dtype="datetime64[ns]")

        for i, ts in enumerate(tqdm(test_dates, desc=f"[{model_name}]")):
            fcs_path = fcs_dir / f"fcs.{ts}.nc"
            obs_path = OBS_DIR / f"obs.{ts}.nc"

            with xr.open_dataset(fcs_path) as ds_fcs:
                ws = select_model_if_present(ds_fcs[WS_VAR], model_name)
                ws_np = ws.transpose("windpark", "lead_time").values.astype(np.float32)

            total = np.zeros((W, T), dtype=np.float32)
            for k, tname in enumerate(turbine_types):
                pc = power_curve(ws_np, specs[tname])
                total += pc * counts_mat[:, k:k+1]
            forecasts_all[i] = total

            with xr.open_dataset(obs_path) as ds_obs:
                truths_all[i] = ds_obs[OBS_VAR].transpose("windpark", "lead_time").values.astype(np.float32)

            if cerra_cache and n_matched > 0:
                init_utc = ts14_to_utc(ts)
                for t_idx, lh in enumerate(lead_hours):
                    vt_iso = (init_utc + pd.Timedelta(hours=lh)).isoformat()
                    if vt_iso in cerra_cache:
                        cerra_all[i, valid_wis, t_idx] = cerra_cache[vt_iso][valid_cis]

        lead_time = np.arange(T, dtype=np.int32)

        for wi, farm in enumerate(windparks):
            ds_out = xr.Dataset(
                {
                    "forecast":    (("date", "lead_time"), forecasts_all[:, wi, :]),
                    "truth":       (("date", "lead_time"), truths_all[:, wi, :]),
                    "truth_cerra": (("date", "lead_time"), cerra_all[:, wi, :]),
                },
                coords={"date": dates_all, "lead_time": lead_time},
                attrs={
                    "windpark": farm,
                    "model_name": model_name,
                    "ws_variable": WS_VAR,
                    "obs_variable": OBS_VAR,
                    "lead_time_original_values": np.array(lead_vals_ref).astype(str).tolist(),
                },
            )
            ds_out.to_netcdf(out_dir / f"{out_prefix}{safe_name(farm)}.nc")
            ds_out.close()

    ds_cerra.close()


if __name__ == "__main__":
    main()
