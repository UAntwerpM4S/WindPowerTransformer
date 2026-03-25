import os
import pickle
import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset, DataLoader

from data.stats import stats_path


def _interp1d_nan_to_num(arr: np.ndarray) -> np.ndarray:
    """Linear interpolate along time; nearest boundary fill; zeros if all NaN."""
    x = np.arange(arr.shape[0])
    m = np.isfinite(arr)
    if not m.any():
        return np.zeros_like(arr, dtype=np.float32)
    out = arr.astype(np.float32).copy()
    if not m.all():
        out[~m] = np.interp(x[~m], x[m], out[m])
    return out


def _interp2d_time_lastdim(arr2d: np.ndarray) -> np.ndarray:
    """Interpolate each feature series along time axis (T,F)."""
    arr2d = np.asarray(arr2d, dtype=np.float32)
    T, F = arr2d.shape
    out = np.empty_like(arr2d)
    for f in range(F):
        out[:, f] = _interp1d_nan_to_num(arr2d[:, f])
    return out


class WindRampDataset(Dataset):
    """
    Expects files:
      fcs.<init>.nc with dims (model, windpark, lead_time) and vars:
        ws10, ws100, wdir10_sin, wdir10_cos, wdir_sin100, wdir_cos100
      obs.<init>.nc with dims (windpark, lead_time) and var:
        WP

    Returns:
        X:    (T, F) float32 — inputs (imputed)
        y:    (T,)    float32 — target WP
    """

    def __init__(
        self,
        date_file,
        fcs_dir="/mnt/weatherloss/WindPowerTransformer/data/FCS",
        obs_dir="/mnt/weatherloss/WindPowerTransformer/data/OBS",
        model_name="GraphTransformer",
        windpark="Belwind Phase 1",
        features=("ws10", "ws100", "wdir10_sin", "wdir10_cos", "wdir100_sin", "wdir100_cos"),
        keep_hourly_only=True,  # keeps 0,3,6,... too since they are integers
    ):
        with open(date_file, "rb") as f:
            all_dates = pickle.load(f)

        self.fcs_dir = fcs_dir
        self.obs_dir = obs_dir
        self.model_name = model_name
        self.windpark = windpark
        self.features = list(features)
        self.keep_hourly_only = keep_hourly_only

        # Filter out dates where files are missing
        self.dates = [
            d for d in all_dates
            if os.path.exists(os.path.join(fcs_dir, f"fcs.{d}.nc"))
            and os.path.exists(os.path.join(obs_dir, f"obs.{d}.nc"))
        ]
        if len(self.dates) < len(all_dates):
            print(f"Warning: skipped {len(all_dates) - len(self.dates)} dates with missing files.")

        # Load train-only stats for THIS (model, park)
        p = stats_path(model_name, windpark)
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Stats not found: {p}. Train.py should call stats.ensure_stats() before creating loaders."
            )
        with open(p, "rb") as fh:
            self.stats = pickle.load(fh)

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, idx):
        init_str = self.dates[idx]

        fcs_path = os.path.join(self.fcs_dir, f"fcs.{init_str}.nc")
        obs_path = os.path.join(self.obs_dir, f"obs.{init_str}.nc")

        with xr.open_dataset(fcs_path) as fcs_ds, xr.open_dataset(obs_path) as obs_ds:
            fcs = fcs_ds.sel(model=self.model_name, windpark=self.windpark).squeeze()
            obs = obs_ds.sel(windpark=self.windpark).squeeze()

            # Keep only integer-hour lead times (keeps 3-hourly too)
            if self.keep_hourly_only:
                hourly = (fcs["lead_time"] % 1.0 == 0)
                fcs = fcs.sel(lead_time=hourly)
                obs = obs.sel(lead_time=hourly)

            # Build features (standardize ws10/ws100; sin/cos are already [-1, 1])
            feat_list = []
            for var in self.features:
                arr = fcs[var].values.astype(np.float32).copy()  # (T,)

                if var in ("ws10", "ws100"):
                    mu = float(self.stats[var]["mean"])
                    sd = float(self.stats[var]["std"])
                    arr = (arr - mu) / max(sd, 1e-6)

                feat_list.append(arr)

            X = np.stack(feat_list, axis=-1)  # (T, F)

            y = obs["WP"].values.astype(np.float32).copy()  # (T,)

        target_finite = np.isfinite(y)
        inputs_finite = np.all(np.isfinite(X), axis=-1)

        # Impute inputs so the model never sees NaNs/Infs
        X = _interp2d_time_lastdim(X)

        return (
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )


def loader_prepare(
    batch_size=16,
    model_name="GraphTransformer",
    windpark="Belwind Phase 1",
    fcs_dir="/mnt/weatherloss/WindPowerTransformer/data/FCS",
    obs_dir="/mnt/weatherloss/WindPowerTransformer/data/OBS",
    train_dates="data/train_dates.pkl",
    val_dates="data/val_dates.pkl",
    test_dates="data/test_dates.pkl",
    num_workers_train=4,
    num_workers_eval=2,
):
    train_set = WindRampDataset(
        train_dates,
        fcs_dir=fcs_dir,
        obs_dir=obs_dir,
        model_name=model_name,
        windpark=windpark,
    )
    val_set = WindRampDataset(
        val_dates,
        fcs_dir=fcs_dir,
        obs_dir=obs_dir,
        model_name=model_name,
        windpark=windpark,
    )
    test_set = WindRampDataset(
        test_dates,
        fcs_dir=fcs_dir,
        obs_dir=obs_dir,
        model_name=model_name,
        windpark=windpark,
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers_train)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers_eval)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers_eval)

    return train_loader, val_loader, test_loader
