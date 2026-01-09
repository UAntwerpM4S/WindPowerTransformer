#!/usr/bin/env python3
import re
import pickle
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

MODEL_NAMES = ["GraphTransformer", "GNN", "Transformer"]

FCS_BASE_DIR   = Path("data/FCS")   
OBS_DIR        = Path("data/OBS")    
TEST_DATES_PKL = Path("data/test_dates.pkl")

POWER_META_DIR = Path("/mnt/weatherloss/WindPower/data/NorthSea/Power")
COUNTS_PATH = POWER_META_DIR / "wind_farm_turbine_counts.csv"
SPECS_PATH  = POWER_META_DIR / "turbine_specs.csv"

OUT_DIR = Path("PC_FCS")

WS_VAR  = "ws100"
OBS_VAR = "WP"


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_")


def ts14_to_datetime64(ts: str) -> np.datetime64:
    return np.datetime64(datetime.strptime(ts, "%Y%m%d%H%M%S"), "ns")


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
    out[ramp] = spec.rated_power_mw * ((ws[ramp] - spec.cut_in) / (spec.rated_ws - spec.cut_in)) ** 3
    rated = (ws >= spec.rated_ws) & (ws < spec.cut_out)
    out[rated] = spec.rated_power_mw
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


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    test_dates = pickle.load(open(TEST_DATES_PKL, "rb"))
    specs = load_specs(SPECS_PATH)
    counts_df = load_counts(COUNTS_PATH)

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

        counts_mat = counts_df.reindex(windparks).to_numpy(dtype=np.float32)
        turbine_types = list(counts_df.columns)
        W = len(windparks)

        N = len(test_dates)
        forecasts_all = np.zeros((N, W, T), dtype=np.float32)
        truths_all    = np.zeros((N, W, T), dtype=np.float32)
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

        lead_time = np.arange(T, dtype=np.int32)

        for wi, farm in enumerate(windparks):
            ds_out = xr.Dataset(
                {
                    "forecast": (("date", "lead_time"), forecasts_all[:, wi, :]),
                    "truth":    (("date", "lead_time"), truths_all[:, wi, :]),
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


if __name__ == "__main__":
    main()
