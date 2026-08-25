#!/usr/bin/env python3
"""Step 1 (CERRA-truth tier): the 6 wind variables at the Belgian cells, from CERRA reanalysis.

The forecast tier (extract_cells.py) consolidates many forecast_*.nc into cells_BE_<RUN>.nc with
7 inputs (6 wind + cf_lam). The CERRA-truth baseline is different: there is only ONE continuous
CERRA zarr and there is NO LAM power channel, so this writes the SAME file format but wind-only
(no cf_lam). build_targets.py then attaches cf_obs and the split; the loader consumes the result.

CERRA is treated as a stream of pseudo-forecasts: each CERRA timestamp is an "init", and the next
`--leads` hours are the lead axis (valid_time = init + lead). The wind at a valid time is just the
CERRA truth there, so the model learns the physical wind->capacityfactor mapping. Cells, G and
cap_cell come from the SAME build_reconstruction() as extract_cells.py / score_power_configs.py,
so the 15 cells are identical across tiers.

Output: cells_<REGION>_CERRA.nc
    dims   (init, lead_time, cell)  + (farm, cell)
    inputs ws100 ws10 wdir100_cos wdir100_sin wdir10_cos wdir10_sin        (NO cf_lam)
    geom   G[farm, cell]  cap_cell[cell]  turbinecount[cell]  cell_lat  cell_lon  valid_time

Usage:
  python extract_cells_cerra.py                              # BE, 2020-01-01 .. 2025-07-31
  python extract_cells_cerra.py --start 2020-01-01 --end 2024-01-31
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

# -------------------- SETTINGS --------------------
ZARR_PATH  = Path("/mnt/weatherloss/WindPower/data/WindAI/Anemoidatasets/New_Cerra_A_large.zarr")
WPOWER_DIR = Path("/mnt/weatherloss/WindPower/data/WPDistr")
REGION     = "BE"
LEAD_HOURS = list(range(3, 37, 3))     # 3..36 h, matching extract_cells.py
FREQ_HOURS = 3
OUT_DIR    = Path("/mnt/weatherloss/WindPowerTransformer/data/cells")
WIND_VARS  = ["ws100", "ws10", "wdir100_cos", "wdir100_sin", "wdir10_cos", "wdir10_sin"]
# --------------------------------------------------


def to_180(lon):
    lon = np.asarray(lon, dtype=float)
    return ((lon + 180.0) % 360.0) - 180.0


def build_reconstruction(cerra_lat, cerra_lon, turbines, farms):
    """Assign turbines to CERRA cells; G[f,c] = capacity of farm f in cell c. Same as extract_cells."""
    coslat = np.cos(np.radians(float(cerra_lat.mean())))
    tree = cKDTree(np.c_[to_180(cerra_lon) * coslat, cerra_lat])
    _, cell = tree.query(np.c_[to_180(turbines["longitude"]) * coslat,
                               turbines["latitude"].to_numpy()], k=1)
    t = turbines.assign(cell=cell.astype(int))
    cell_idx = np.sort(t["cell"].unique())
    cpos = {int(c): j for j, c in enumerate(cell_idx)}
    fpos = {f: i for i, f in enumerate(farms)}
    G = np.zeros((len(farms), cell_idx.size), dtype=np.float64)
    for (farm, c), cap in t.groupby(["farm", "cell"])["capacity_mw"].sum().items():
        G[fpos[farm], cpos[int(c)]] = cap
    cap_cell = t.groupby("cell")["capacity_mw"].sum().reindex(cell_idx).to_numpy()
    return cell_idx, G, cap_cell


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zarr", type=Path, default=ZARR_PATH)
    ap.add_argument("--wpower-dir", type=Path, default=WPOWER_DIR)
    ap.add_argument("--region", default=REGION, choices=["BE", "UK", "all"])
    ap.add_argument("--leads", type=int, nargs="+", default=LEAD_HOURS)
    ap.add_argument("--freq-hours", type=int, default=FREQ_HOURS)
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2025-07-31 21:00")
    ap.add_argument("--ensemble", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if any(lh % args.freq_hours for lh in args.leads):
        raise SystemExit(f"every lead must be a multiple of freq_hours ({args.freq_hours})")

    farms_df = pd.read_csv(args.wpower_dir / "farms.csv")
    turbines = pd.read_csv(args.wpower_dir / "turbines.csv")
    farms = (farms_df.farm.tolist() if args.region == "all"
             else farms_df[farms_df.region == args.region].farm.tolist())
    turbines = turbines[turbines.farm.isin(farms)]
    cap = farms_df.set_index("farm").loc[farms, "capacity_mw"]
    print(f"{args.region}: {len(farms)} farms, {cap.sum():.1f} MW, {len(turbines)} turbines")

    ds = xr.open_zarr(args.zarr, consolidated=False)
    var_names = list(ds.attrs["variables"])
    missing = [v for v in WIND_VARS if v not in var_names]
    if missing:
        raise SystemExit(f"CERRA zarr lacks {missing}")
    cerra_lat = np.asarray(ds["latitudes"]).ravel()
    cerra_lon = to_180(np.asarray(ds["longitudes"]).ravel())
    dates = pd.DatetimeIndex(pd.to_datetime(ds["dates"].values))
    dates = dates.tz_localize(None) if dates.tz is not None else dates   # naive UTC, like extract_cells

    cell_idx, G, cap_cell = build_reconstruction(cerra_lat, cerra_lon, turbines, farms)
    C = cell_idx.size
    cell_lat = cerra_lat[cell_idx].astype(np.float64)
    cell_lon = cerra_lon[cell_idx].astype(np.float64)
    shared = int(((G > 0).sum(0) > 1).sum())
    print(f"  {C} cells, {cap_cell.sum():.1f} MW, {shared} shared")

    # turbines per cell (static forcing, like the LAM's turbinecount). Reassign here since we
    # have the turbine table; store it so the loader gets it straight from the file.
    coslat = np.cos(np.radians(float(cerra_lat.mean())))
    _tree = cKDTree(np.c_[to_180(cerra_lon) * coslat, cerra_lat])
    _, _tc = _tree.query(np.c_[to_180(turbines["longitude"]) * coslat,
                               turbines["latitude"].to_numpy()], k=1)
    count_cell = (pd.Series(_tc).value_counts().reindex(cell_idx).fillna(0)
                  .to_numpy().astype(np.float64))

    # wind at the 15 cells for ALL times, once (small: T x C x 6)
    fidx = [var_names.index(v) for v in WIND_VARS]
    da = (ds["data"].isel(ensemble=args.ensemble)
          .isel(cell=xr.DataArray(cell_idx, dims="c")).transpose("time", "variable", "c"))
    wind_all = np.transpose(da.values[:, fidx, :].astype(np.float32), (0, 2, 1))  # (T, C, 6)
    ds.close()

    # sliding pseudo-forecast windows: init i -> valid steps i + lead/freq
    steps = [lh // args.freq_hours for lh in args.leads]     # e.g. 1..12
    L, max_step = len(args.leads), max(steps)
    t0 = pd.Timestamp(args.start); t1 = pd.Timestamp(args.end)
    T_all = len(dates)
    step_h = dates.values.astype("datetime64[h]")

    inits, blocks, valids = [], [], []
    for i in range(T_all - max_step):
        if not (t0 <= dates[i] <= t1):
            continue
        vt_idx = [i + s for s in steps]
        # require regular 3-hourly spacing across the whole window (guards CERRA gaps)
        span = (step_h[vt_idx[-1]] - step_h[i]).astype("timedelta64[h]").astype(int)
        if span != max(args.leads):
            continue
        inits.append(dates[i])
        blocks.append(wind_all[vt_idx])                      # (L, C, 6)
        valids.append(dates.values[vt_idx])

    if not inits:
        raise SystemExit("no usable init windows in the requested range")
    arr = np.stack(blocks)                                   # (N, L, C, 6)
    N = arr.shape[0]
    init_ix = pd.DatetimeIndex(inits)
    valid = np.stack(valids)                                 # (N, L) datetime64

    ds_out = xr.Dataset(
        {name: (("init", "lead_time", "cell"), arr[:, :, :, v]) for v, name in enumerate(WIND_VARS)}
        | {
            "G":          (("farm", "cell"), G),
            "cap_cell":   ("cell", cap_cell),
            "turbinecount": ("cell", count_cell),
            "valid_time": (("init", "lead_time"), valid),
        },
        coords={
            "init": init_ix.values,
            "lead_time": np.asarray(args.leads, dtype=np.int32),
            "cell": np.arange(C, dtype=np.int32),
            "farm": np.array(farms, dtype=object),
            "cell_lat": ("cell", cell_lat),
            "cell_lon": ("cell", cell_lon),
        },
        attrs={
            "description": f"CERRA-truth wind (6 vars) at the {args.region} cells, per init and lead",
            "run": "CERRA",
            "source": str(args.zarr),
            "region": args.region,
            "power_var": "NONE (CERRA truth has no LAM power channel; wind-only baseline)",
            "time_convention": "init and valid_time are naive UTC. power_obs at t is the mean MW "
                               "over [t, t+3h) and LEADS the field by ~1.5 h -- handled downstream.",
            "reconstruction": "P_pred(farm,t) = sum_cell G[farm,cell] * CF_pred(cell,t)",
        },
    )
    ds_out["lead_time"].attrs["units"] = "h"
    ds_out["G"].attrs["units"] = "MW"
    ds_out["cap_cell"].attrs["units"] = "MW"

    out_path = args.out / f"cells_{args.region}_CERRA.nc"
    enc = {v: {"zlib": True, "complevel": 4}
           for v in list(WIND_VARS) + ["G", "cap_cell", "turbinecount"]}
    tmp = out_path.with_suffix(".nc.tmp")
    ds_out.to_netcdf(tmp, format="NETCDF4", engine="netcdf4", encoding=enc)
    tmp.replace(out_path)
    print(f"  wrote {out_path}")
    print(f"  {N} inits x {L} leads x {C} cells x {len(WIND_VARS)} wind vars "
          f"({init_ix.min():%Y-%m-%d} .. {init_ix.max():%Y-%m-%d})")


if __name__ == "__main__":
    main()
