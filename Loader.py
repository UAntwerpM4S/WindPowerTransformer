"""
Loader for the cell-level PowerTransformer — reads dataset_*.nc from build_targets.py.

ONE loader for BOTH tiers (the only difference is whether cf_lam is present):
  - CERRA-truth baseline : dataset_BE_CERRA.nc            inputs = 6 wind          (+2 static)
  - forecast post-proc   : dataset_BE_<RUN>.nc            inputs = 6 wind + cf_lam (+2 static)

A dataset_*.nc (built by extract_cells[_cerra].py -> build_targets.py) holds, per init and lead:
    ws100 ws10 wdir100_{cos,sin} wdir10_{cos,sin}   [+ cf_lam]     (init, lead_time, cell)
    cf_obs                                                         (init, lead_time, cell)  TARGET
    G[farm, cell]  cap_cell[cell]                                  aggregation geometry
    split[init] in {train, val, test, unused}

Sample = one (init, cell): a 36h lead-window with X (L, F), y (L,) capacityfactor, m (L,) mask.
F = 6 wind (+ cf_lam if present) + 2 static cell features (capacity, rated_ws), so ONE shared
model specialises per cell. Static features are derived from G + specs (order-independent, they
travel with the cell), so a model trained on the CERRA cells applies unchanged to forecast cells.

Standardisation (ws10/ws100/capacity/rated_ws) is fit on the TRAIN split only.
"""

import os
import pickle
import re

import numpy as np
import pandas as pd
import xarray as xr
import torch
from torch.utils.data import Dataset, DataLoader

WIND_VARS = ("ws100", "ws10", "wdir100_cos", "wdir100_sin", "wdir10_cos", "wdir10_sin")
STANDARDIZE = ("ws100", "ws10")               # among the wind vars; sin/cos already in [-1,1]
FLEET_RE = re.compile(r"\s*(\d+)\s*x\s*(.+?)\s*$")


# --------------------------------------------------------------------------- #
# static per-cell features from the file's G + specs (order-independent)
# --------------------------------------------------------------------------- #
def farm_rated_ws(farms_df, specs) -> dict:
    """Capacity-weighted mean rated wind speed per farm, from the fleet string + specs."""
    out = {}
    for _, r in farms_df.iterrows():
        num = den = 0.0
        for chunk in str(r["fleet"]).split(";"):
            m = FLEET_RE.match(chunk)
            if not m:
                continue
            cnt, ttype = int(m.group(1)), m.group(2)
            if ttype not in specs.index:
                continue
            mw = float(specs.loc[ttype, "rated_power_mw"])
            num += cnt * mw * float(specs.loc[ttype, "rated_ws_ms"])
            den += cnt * mw
        out[r["farm"]] = num / den if den > 0 else np.nan
    return out


def cell_static(ds, farms_df, specs):
    """Static features per cell — the LAM's forcings + rated_ws:
        [capacity, turbinecount, capacity-weighted rated_ws].

    capacity and turbinecount come from the file; rated_ws = sum_f G[f,c]*rws_f / sum_f G[f,c]
    from the file's G and the farm rated speeds. All are per-cell physical values, independent of
    cell ordering, so a model trained on the CERRA cells applies unchanged to forecast cells.
    """
    G = ds["G"].values                                    # (F, C)
    cap_cell = ds["cap_cell"].values                      # (C,)
    count_cell = ds["turbinecount"].values                # (C,)
    farms = [str(f) for f in ds["farm"].values]
    rws = farm_rated_ws(farms_df, specs)
    rws_vec = np.array([rws[f] for f in farms])           # (F,)
    with np.errstate(invalid="ignore"):
        ratedws_cell = (G * rws_vec[:, None]).sum(0) / G.sum(0)
    static = np.stack([cap_cell, count_cell, ratedws_cell], axis=1).astype(np.float32)
    return static, ["capacity", "turbinecount", "rated_ws"]


# --------------------------------------------------------------------------- #
def _interp1d(arr):
    x = np.arange(arr.shape[0]); m = np.isfinite(arr)
    if not m.any():
        return np.zeros_like(arr, dtype=np.float32)
    out = arr.astype(np.float32).copy()
    if not m.all():
        out[~m] = np.interp(x[~m], x[m], out[m])
    return out


def _interp2d(a):
    a = np.asarray(a, np.float32); out = np.empty_like(a)
    for f in range(a.shape[1]):
        out[:, f] = _interp1d(a[:, f])
    return out


# --------------------------------------------------------------------------- #
# stats (train split only)
# --------------------------------------------------------------------------- #
def ensure_stats(inputs, feat_names, static, static_names, split, run_tag):
    """mean/std over the TRAIN inits for the standardised features. inputs (N,L,C,Fin)."""
    path = os.path.join("artifacts", run_tag, "cell_feature_stats.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    tr = split == "train"
    stats = {}
    for v in STANDARDIZE:
        if v in feat_names:
            vals = inputs[tr][:, :, :, feat_names.index(v)].ravel()
            vals = vals[np.isfinite(vals)]
            stats[v] = {"mean": float(vals.mean()), "std": max(float(vals.std()), 1e-6)}
    for j, v in enumerate(static_names):        # static: standardise capacity & rated_ws
        vals = static[:, j][np.isfinite(static[:, j])]
        stats[v] = {"mean": float(vals.mean()), "std": max(float(vals.std()), 1e-6)}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(stats, f)
    print(f"Saved cell stats -> {path}")
    return stats


def apply_stats(inputs, feat_names, static, static_names, stats):
    """Standardise ws + static with the given (train-fit) stats. Used by BOTH training and
    inference so the feature scaling is identical. inputs standardised in place; static copied."""
    for v in STANDARDIZE:
        if v in feat_names:
            j = feat_names.index(v)
            inputs[:, :, :, j] = (inputs[:, :, :, j] - stats[v]["mean"]) / stats[v]["std"]
    static = static.copy()
    for j, v in enumerate(static_names):
        static[:, j] = (np.nan_to_num(static[:, j], nan=stats[v]["mean"])
                        - stats[v]["mean"]) / stats[v]["std"]
    return inputs, static


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #
class CellDataset(Dataset):
    def __init__(self, inputs, cf_obs, static, feat_names, static_names, split, which, stats):
        """inputs (N,L,C,Fin_dyn) standardised; static (C,Fstat) standardised; target (N,L,C)."""
        self.X = inputs                     # already standardised (dynamic features)
        self.static = static                # (C, Fstat) standardised
        self.cf = cf_obs                    # (N, L, C)
        sel = np.where(split == which)[0]   # init indices in this split
        C = inputs.shape[2]
        # (init, cell) pairs whose target has at least one finite step
        self.index = [(i, c) for i in sel for c in range(C)
                      if np.isfinite(self.cf[i, :, c]).any()]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        i, c = self.index[k]
        dyn = self.X[i, :, c, :]                             # (L, Fdyn)
        stat = np.broadcast_to(self.static[c], (dyn.shape[0], self.static.shape[1]))
        X = np.concatenate([dyn, stat], axis=1).astype(np.float32)
        y = self.cf[i, :, c]                                 # (L,)
        m = np.isfinite(y)
        X = _interp2d(X)
        y = np.where(m, y, 0.0).astype(np.float32)
        return torch.from_numpy(X), torch.from_numpy(y), torch.from_numpy(m)


# --------------------------------------------------------------------------- #
def loader_prepare(dataset_nc, farms_csv, specs_csv, run_tag, batch_size=64,
                   num_workers_train=4, num_workers_eval=2, use_cf_lam=True):
    ds = xr.open_dataset(dataset_nc)
    feat_names = list(WIND_VARS) + (["cf_lam"] if use_cf_lam and "cf_lam" in ds else [])
    N, L, C = ds.sizes["init"], ds.sizes["lead_time"], ds.sizes["cell"]
    inputs = np.stack([ds[v].values for v in feat_names], axis=-1).astype(np.float32)  # (N,L,C,Fdyn)
    cf_obs = ds["cf_obs"].values.astype(np.float32)          # (N, L, C)
    split = ds["split"].values.astype(str)                   # (N,)

    farms_df = pd.read_csv(farms_csv)
    specs = pd.read_csv(specs_csv)
    specs = specs.rename(columns={specs.columns[0]: "turbine_type"}).set_index("turbine_type")
    static, static_names = cell_static(ds, farms_df, specs)  # (C, 2)
    geom = {"G": ds["G"].values, "cap_cell": ds["cap_cell"].values,
            "farms": [str(f) for f in ds["farm"].values],
            "cell_lat": ds["cell_lat"].values, "cell_lon": ds["cell_lon"].values,
            "leads": ds["lead_time"].values}
    ds.close()

    print(f"{dataset_nc.name if hasattr(dataset_nc,'name') else dataset_nc}: "
          f"{N} inits x {L} leads x {C} cells | features {feat_names} + static {static_names}")
    for nm in ("train", "val", "test"):
        print(f"  {nm}: {(split == nm).sum()} inits")

    stats = ensure_stats(inputs, feat_names, static, static_names, split, run_tag)
    inputs, static = apply_stats(inputs, feat_names, static, static_names, stats)

    def make(which):
        return CellDataset(inputs, cf_obs, static, feat_names, static_names, split, which, stats)

    tr, va, te = make("train"), make("val"), make("test")
    print(f"cell-windows -> train={len(tr)} val={len(va)} test={len(te)}")
    dl = lambda d, s, nw: DataLoader(d, batch_size=batch_size, shuffle=s, num_workers=nw)
    input_dim = inputs.shape[-1] + static.shape[1]
    return (dl(tr, True, num_workers_train), dl(va, False, num_workers_eval),
            dl(te, False, num_workers_eval), geom, input_dim)


# --------------------------------------------------------------------------- #
# inference: the full standardised grid for a dataset, using the SAVED train stats
# --------------------------------------------------------------------------- #
def inference_features(dataset_nc, farms_csv, specs_csv, run_tag, use_cf_lam=True):
    """Standardised model input (N, L, C, input_dim) for a whole dataset.

    Same feature construction as loader_prepare (WIND_VARS [+cf_lam] + 3 static, ws & static
    standardised) but with the stats LOADED from artifacts/<run_tag>/ rather than refit, and
    returned as the full gridded array + coords + geometry so infer_cf.py can run the model and
    aggregate. This is the single source of feature parity between training and inference.

    use_cf_lam=False drops the cf_lam channel even when the dataset has it (the wind-only variant),
    so the SAME dataset trains/scores both tiers -- must match how run_tag was trained.
    """
    ds = xr.open_dataset(dataset_nc)
    feat_names = list(WIND_VARS) + (["cf_lam"] if use_cf_lam and "cf_lam" in ds else [])
    inputs = np.stack([ds[v].values for v in feat_names], axis=-1).astype(np.float32)  # (N,L,C,Fd)

    farms_df = pd.read_csv(farms_csv)
    specs = pd.read_csv(specs_csv)
    specs = specs.rename(columns={specs.columns[0]: "turbine_type"}).set_index("turbine_type")
    static, static_names = cell_static(ds, farms_df, specs)

    path = os.path.join("artifacts", str(run_tag), "cell_feature_stats.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing train stats {path} -- train run_tag={run_tag} first")
    with open(path, "rb") as f:
        stats = pickle.load(f)
    inputs, static = apply_stats(inputs, feat_names, static, static_names, stats)

    N, L, C = inputs.shape[:3]
    stat_b = np.broadcast_to(static[None, None], (N, L, C, static.shape[1]))
    X = np.concatenate([inputs, stat_b], axis=-1).astype(np.float32)         # (N,L,C,input_dim)
    for i in range(N):                                                       # gap-fill per series
        for c in range(C):
            if not np.isfinite(X[i, :, c, :]).all():
                X[i, :, c, :] = _interp2d(X[i, :, c, :])

    out = dict(
        X=X, feat_names=feat_names + static_names,
        split=ds["split"].values.astype(str),
        init=pd.DatetimeIndex(ds["init"].values),
        leads=np.asarray(ds["lead_time"].values, dtype=int),
        valid=ds["valid_time"].values,
        farms=[str(f) for f in ds["farm"].values],
        G=ds["G"].values, cap_cell=ds["cap_cell"].values,
        cell_lat=ds["cell_lat"].values, cell_lon=ds["cell_lon"].values,
    )
    ds.close()
    return out
