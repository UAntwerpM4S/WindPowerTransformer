"""
CERRA-backed dataset for the WindPowerTransformer.

The continuous CERRA reanalysis zarr (3-hourly) is treated as a stream of
*pseudo-forecasts*: each CERRA timestamp is an "init", and the next
`lead_hours` worth of steps form the lead-time axis. One sample is

    X : (T, F)  float32   -- F weather inputs over the 36h window
    y : (T,)    float32   -- target power over the window
    m : (T,)    bool      -- finite-target mask (real obs can have gaps)

with T = lead_hours // freq_hours + 1 (e.g. 36h at 3-hourly -> 13 steps:
0,3,6,...,36h).

The zarr is the Anemoi format used by build_powercurve.py:
    data        (time, variable, ensemble, cell)
    latitudes   (cell,)
    longitudes  (cell,)
    dates       (time,)
    attrs["variables"] -> ordered list of variable names

IMPORTANT: the zarr's `power` variable holds the injected real observations and
is used as the training target. The six weather inputs are taken at the
windpark's CERRA grid cell (same cell mapping as build_powercurve.py).
"""

import os
import pickle

import numpy as np
import pandas as pd
import xarray as xr
import torch
from torch.utils.data import Dataset, DataLoader


DEFAULT_FEATURES = (
    "ws10", "ws100",
    "wdir10_sin", "wdir10_cos",
    "wdir100_sin", "wdir100_cos",
)
STANDARDIZE_VARS = ("ws10", "ws100")  # sin/cos are already in [-1, 1]

# Per-farm observed power (the training target). One column per farm, UTC index.
OBS_CSV = "/mnt/weatherloss/WindPower/data/NorthSea/Power/BE_UK_offshore_per_unit_3H_meanMW_shifted.csv"


def load_obs_csv(path=OBS_CSV) -> pd.DataFrame:
    """Per-farm observed power, indexed by UTC timestamp, columns = farm names."""
    df = pd.read_csv(path)
    if "time" not in df.columns:
        raise ValueError(f"Observation CSV must contain a 'time' column: {path}")
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time").sort_index()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def farm_obs_aligned(obs_df: pd.DataFrame, farm: str, dates_utc) -> np.ndarray:
    """Per-farm obs aligned to dates_utc (tz-aware UTC); NaN where missing."""
    if farm not in obs_df.columns:
        print(f"⚠️ farm {farm!r} not in obs CSV columns — target will be all-NaN.")
        return np.full(len(dates_utc), np.nan, dtype=np.float32)
    return obs_df[farm].reindex(dates_utc).to_numpy(dtype=np.float32)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _interp1d_nan_to_num(arr: np.ndarray) -> np.ndarray:
    """Linear-interpolate along time; nearest-boundary fill; zeros if all NaN."""
    x = np.arange(arr.shape[0])
    m = np.isfinite(arr)
    if not m.any():
        return np.zeros_like(arr, dtype=np.float32)
    out = arr.astype(np.float32).copy()
    if not m.all():
        out[~m] = np.interp(x[~m], x[m], out[m])
    return out


def _interp2d_time_lastdim(arr2d: np.ndarray) -> np.ndarray:
    """Interpolate each feature series along the time axis (T, F)."""
    arr2d = np.asarray(arr2d, dtype=np.float32)
    out = np.empty_like(arr2d)
    for f in range(arr2d.shape[1]):
        out[:, f] = _interp1d_nan_to_num(arr2d[:, f])
    return out


def windpark_cerra_index(metadata_path, cerra_lat, cerra_lon, windpark) -> int:
    """Map a single windpark name to its flat CERRA cell index.

    Mirrors build_windpark_cerra_indices() in build_powercurve.py.
    """
    meta = pd.read_csv(metadata_path)
    cerra_keys = {
        (round(float(la), 6), round(float(lo), 6)): i
        for i, (la, lo) in enumerate(zip(cerra_lat, cerra_lon))
    }
    for _, row in meta.iterrows():
        if str(row["farm"]) != str(windpark):
            continue
        key = (round(float(row["cerra_grid_lat"]), 6), round(float(row["cerra_grid_lon"]), 6))
        if key in cerra_keys:
            return cerra_keys[key]
        raise KeyError(
            f"Windpark {windpark!r} found in metadata but its CERRA grid "
            f"({key}) does not match any CERRA cell."
        )
    raise KeyError(f"Windpark {windpark!r} not found in {metadata_path}")


def _open_cerra(zarr_path):
    return xr.open_zarr(zarr_path, consolidated=False)


def _cerra_cell_series(ds, metadata_path, windpark, ensemble):
    """Return (dates_utc, var_names, series) for one windpark cell.

    series : (T_all, V) float32 array of ALL zarr variables at that cell.
    """
    var_names = list(ds.attrs["variables"])
    cerra_lat = ds["latitudes"].values
    cerra_lon = ds["longitudes"].values
    cell_idx = windpark_cerra_index(metadata_path, cerra_lat, cerra_lon, windpark)

    # one read of every variable at this cell: dims -> (time, variable)
    series = (
        ds["data"]
        .isel(ensemble=ensemble, cell=cell_idx)
        .transpose("time", "variable")
        .values.astype(np.float32)
    )
    dates = pd.to_datetime(ds["dates"].values)
    if dates.tz is None:
        dates = dates.tz_localize("UTC")
    else:
        dates = dates.tz_convert("UTC")
    return dates, var_names, series


def _col_indices(var_names, requested):
    missing = [v for v in requested if v not in var_names]
    if missing:
        raise KeyError(
            f"Variables {missing} not in CERRA zarr. Available: {var_names}"
        )
    return [var_names.index(v) for v in requested]


# --------------------------------------------------------------------------- #
# stats (train split only)
# --------------------------------------------------------------------------- #
def stats_path(run_tag, windpark) -> str:
    return os.path.join("artifacts", run_tag, windpark, "cerra_feature_stats.pkl")


def ensure_stats(
    zarr_path,
    metadata_path,
    windpark,
    train_start,
    train_end,
    run_tag,
    ws_vars=STANDARDIZE_VARS,
    ensemble=0,
):
    """Compute mean/std of `ws_vars` over the TRAIN date range only (per park)."""
    path = stats_path(run_tag, windpark)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    with _open_cerra(zarr_path) as ds:
        dates, var_names, series = _cerra_cell_series(ds, metadata_path, windpark, ensemble)

    t0 = pd.Timestamp(train_start, tz="UTC")
    t1 = pd.Timestamp(train_end, tz="UTC")
    in_train = (dates >= t0) & (dates <= t1)

    idx = _col_indices(var_names, ws_vars)
    stats = {}
    for v, ci in zip(ws_vars, idx):
        vals = series[in_train, ci]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            raise RuntimeError(f"No finite {v} values for {windpark} in train range.")
        mu = float(vals.mean())
        sd = float(vals.std(ddof=0))
        stats[v] = {"mean": mu, "std": sd if sd > 1e-6 else 1.0}

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(stats, f)
    print(f"Saved CERRA stats -> {path}: {stats}")
    return stats


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #
class CerraWindowDataset(Dataset):
    def __init__(
        self,
        zarr_path,
        metadata_path,
        windpark,
        start,
        end,
        stats,
        obs_df,
        features=DEFAULT_FEATURES,
        lead_hours=36,
        freq_hours=3,
        stride=1,
        ensemble=0,
    ):
        if lead_hours % freq_hours != 0:
            raise ValueError(f"lead_hours ({lead_hours}) must be divisible by freq_hours ({freq_hours})")
        self.features = list(features)
        self.lead_hours = int(lead_hours)
        self.freq_hours = int(freq_hours)
        self.n_steps = lead_hours // freq_hours + 1  # inclusive 0..lead_hours

        with _open_cerra(zarr_path) as ds:
            dates, var_names, series = _cerra_cell_series(
                ds, metadata_path, windpark, ensemble
            )

        feat_idx = _col_indices(var_names, self.features)

        self.X_all = series[:, feat_idx].astype(np.float32)        # (T_all, F)
        # target = per-farm observed power (real obs, NaN where missing)
        self.y_all = farm_obs_aligned(obs_df, windpark, dates)     # (T_all,)
        self.dates = dates

        # standardize ws10/ws100 in place using train stats
        for v in STANDARDIZE_VARS:
            if v in self.features:
                fpos = self.features.index(v)
                mu = float(stats[v]["mean"])
                sd = float(stats[v]["std"])
                self.X_all[:, fpos] = (self.X_all[:, fpos] - mu) / max(sd, 1e-6)

        # contiguity guard: a window is valid only if its n_steps span exactly
        # lead_hours (CERRA is regular 3-hourly, but be safe at gaps).
        step = self.dates.values.astype("datetime64[h]")
        T_all = len(self.dates)
        t0 = pd.Timestamp(start, tz="UTC")
        t1 = pd.Timestamp(end, tz="UTC")
        in_split = (self.dates >= t0) & (self.dates <= t1)

        starts = []
        for i in range(0, T_all - self.n_steps + 1, stride):
            if not in_split[i]:
                continue
            span = (step[i + self.n_steps - 1] - step[i]).astype("timedelta64[h]").astype(int)
            if span == self.lead_hours:  # exact regular spacing across the window
                starts.append(i)
        self.starts = np.asarray(starts, dtype=np.int64)
        self.init_dates = self.dates[self.starts]

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s = int(self.starts[idx])
        e = s + self.n_steps

        X = self.X_all[s:e].copy()           # (T, F)
        y = self.y_all[s:e].copy()           # (T,)

        mask = np.isfinite(y)
        X = _interp2d_time_lastdim(X)        # model never sees NaN inputs
        y = np.where(mask, y, 0.0).astype(np.float32)

        return (
            torch.from_numpy(X),
            torch.from_numpy(y),
            torch.from_numpy(mask),
        )


# --------------------------------------------------------------------------- #
# prepare
# --------------------------------------------------------------------------- #
def loader_prepare(
    windpark,
    zarr_path,
    metadata_path,
    run_tag,
    train_range,
    val_range,
    test_range,
    batch_size=8,
    features=DEFAULT_FEATURES,
    obs_csv=OBS_CSV,
    lead_hours=36,
    freq_hours=3,
    stride=1,
    ensemble=0,
    num_workers_train=4,
    num_workers_eval=2,
):
    """Build train/val/test loaders for one windpark from the CERRA zarr.

    Inputs come from the CERRA zarr; the target is per-farm observed power
    from the obs CSV.
    """
    stats = ensure_stats(
        zarr_path, metadata_path, windpark,
        train_start=train_range[0], train_end=train_range[1],
        run_tag=run_tag, ensemble=ensemble,
    )
    obs_df = load_obs_csv(obs_csv)

    def make(rng):
        return CerraWindowDataset(
            zarr_path, metadata_path, windpark,
            start=rng[0], end=rng[1], stats=stats, obs_df=obs_df,
            features=features,
            lead_hours=lead_hours, freq_hours=freq_hours,
            stride=stride, ensemble=ensemble,
        )

    train_set = make(train_range)
    val_set = make(val_range)
    test_set = make(test_range)

    print(
        f"[{windpark}] windows -> train={len(train_set)} "
        f"val={len(val_set)} test={len(test_set)}"
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers_train
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers_eval
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers_eval
    )
    return train_loader, val_loader, test_loader, test_set
