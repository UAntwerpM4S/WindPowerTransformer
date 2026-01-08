# stats.py
import os
import pickle
import numpy as np
import xarray as xr


def _hourly_mask(ds):
    return (ds["lead_time"] % 1.0 == 0)


def stats_path(model_name: str, windpark: str) -> str:
    """Where we persist stats for this (model, park)."""
    return os.path.join("artifacts", model_name, windpark, "feature_stats.pkl")


def compute_ws_stats_for_pair(
    date_file="train_dates.pkl",
    fcs_dir="CERRA",              # <- point this to your new fcs directory
    model_name="GraphTransformer",
    windpark="Belwind Phase 1",
    ws_vars=("ws10", "ws100"),    # <- NEW: compute separate stats for both
):
    """
    Compute ws10/ws100 mean/std from TRAIN split only, for a (model, windpark).

    Expects per-init files: fcs.<init>.nc with dims (model, windpark, lead_time)
    and variables ws10/ws100.
    """
    with open(date_file, "rb") as f:
        train_inits = pickle.load(f)

    vals = {v: [] for v in ws_vars}

    for init in train_inits:
        fpath = os.path.join(fcs_dir, f"fcs.{init}.nc")
        if not os.path.exists(fpath):
            continue

        ds = xr.open_dataset(fpath).sel(model=model_name, windpark=windpark).squeeze()
        ds = ds.sel(lead_time=_hourly_mask(ds))

        for v in ws_vars:
            if v not in ds:
                continue
            arr = ds[v].values.astype(np.float32)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                vals[v].append(arr)

    stats = {}
    for v in ws_vars:
        if not vals[v]:
            raise RuntimeError(f"No finite values for {v} for {model_name} / {windpark} in train split.")
        allv = np.concatenate(vals[v])
        mu = float(allv.mean())
        sd = float(allv.std(ddof=0))
        if sd < 1e-6:
            sd = 1.0
        stats[v] = {"mean": mu, "std": sd}

    out_path = stats_path(model_name, windpark)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(stats, f)

    print(f"Saved stats -> {out_path}: {stats}")
    return out_path


def ensure_stats(date_file, fcs_dir, model_name, windpark, ws_vars=("ws10", "ws100")):
    """Compute stats if missing; return the path either way."""
    p = stats_path(model_name, windpark)
    if not os.path.exists(p):
        return compute_ws_stats_for_pair(date_file, fcs_dir, model_name, windpark, ws_vars=ws_vars)
    return p
