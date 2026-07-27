#!/usr/bin/env python3
"""Step 1: extract the 7 PowerTransformer input variables at the Belgian CERRA cells.

The LAM writes one forecast_YYYYmmddHHMMSS.nc per init time, on the inner cutout grid
(~24k-27k cells).  Belgium's 10 farms occupy 15 of those cells (12 shared between farms).
This script pulls, for every init and every requested lead time, the 6 wind variables plus
the LAM's own power channel at exactly those 15 cells, and writes ONE consolidated NetCDF
per run:

    dims   (init, lead_time, cell)          + (farm, cell) for the reconstruction operator
    inputs ws100 ws10 wdir100_cos wdir100_sin wdir10_cos wdir10_sin cf_lam
    geometry  G[farm, cell]  cap_cell[cell]  cell_lat  cell_lon

`cf_lam` is always a CAPACITY FACTOR: a run whose power variable is raw MW/cell is divided
by cap_cell here, so downstream code never has to care which it was.

`G` and `cap_cell` come from `build_reconstruction()` in WindAI/data/Wpower/score_power_configs.py
and are carried in the file so the aggregation step is self-contained:

    P_pred(farm, t) = sum_cell G[farm, cell] * CF_pred(cell, t)

Cells are identified by LAT/LON, never by index: the forecast grid is a subset of the full
CERRA grid and the subset can differ between files.  The canonical 15 cells are built once
from a reference grid, then looked up per file by nearest neighbour with a hard tolerance.

Usage:
  python extract_cells.py                                  # all runs in FORECAST_DIRS, BE
  python extract_cells.py --runs HighCapacityGT=/path/dir
  python extract_cells.py --limit 20                       # quick smoke test
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

# -------------------- SETTINGS --------------------
WPOWER_DIR = Path("/mnt/weatherloss/WindPower/data/WPDistr")

FORECAST_DIRS = {
    "HighCapacityGT": Path("/mnt/weatherloss/WindPower/inference/WPDistr/HighCapacityGT"),
    "VanillaPowerGT": Path("/mnt/weatherloss/WindPower/inference/WPDistr/VanillaPowerGT"),
}
POWER_VAR  = "capacityfactor"          # or "power" (MW/cell) -> converted to CF here
REGION     = "BE"
LEAD_HOURS = list(range(3, 37, 3))     # 3..36 h, matching score_power_configs.py
OUT_DIR    = Path("/mnt/weatherloss/WindPowerTransformer/data/cells")

WIND_VARS = ["ws100", "ws10", "wdir100_cos", "wdir100_sin", "wdir10_cos", "wdir10_sin"]

# A canonical cell must be found in each file's grid within this distance.  The grids are
# subsets of one CERRA grid (~5.5 km spacing), so a genuine match is exact to rounding; a
# miss means the cell is absent from that file's cutout and the file must not be used.
MATCH_TOL_KM = 1.0
# --------------------------------------------------

FORECAST_RE = re.compile(r"forecast_(\d{14})")


def to_180(lon):
    lon = np.asarray(lon, dtype=float)
    return ((lon + 180.0) % 360.0) - 180.0


def parse_init(path: Path) -> pd.Timestamp:
    return pd.to_datetime(FORECAST_RE.search(path.name).group(1), format="%Y%m%d%H%M%S")


def build_reconstruction(fc_lat, fc_lon, turbines, farms):
    """Assign turbines to forecast cells and build the reconstruction operators.

    Verbatim from WindAI/data/Wpower/score_power_configs.py so the cells here are the same
    cells the scorer uses.

    Returns:
      cell_idx : forecast-cell indices that hold turbines
      G        : (n_farms, n_cells) with G[f,j] = capacity of farm f in cell_idx[j]
      cap_cell : (n_cells,) total capacity per cell
    """
    coslat = np.cos(np.radians(float(fc_lat.mean())))
    tree = cKDTree(np.c_[to_180(fc_lon) * coslat, fc_lat])
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


def grid_key(lat, lon):
    """Cheap signature of a forecast grid, so the cell lookup is done once per distinct grid."""
    return (lat.size, round(float(lat[0]), 4), round(float(lon[-1]), 4))


def locate_cells(lat, lon, cell_lat, cell_lon):
    """Indices of the canonical cells in THIS file's grid, matched by lat/lon."""
    coslat = np.cos(np.radians(float(lat.mean())))
    tree = cKDTree(np.c_[to_180(lon) * coslat, lat])
    d, idx = tree.query(np.c_[to_180(cell_lon) * coslat, cell_lat], k=1)
    d_km = d * 111.32                      # degrees -> km (the x axis is already cos-scaled)
    if d_km.max() > MATCH_TOL_KM:
        bad = int(np.argmax(d_km))
        raise ValueError(
            f"cell {bad} (lat {cell_lat[bad]:.4f}, lon {cell_lon[bad]:.4f}) is "
            f"{d_km.max():.2f} km from the nearest grid point (tol {MATCH_TOL_KM} km): "
            f"this file's cutout does not contain that cell")
    return idx.astype(int)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=None, metavar="LABEL=DIR",
                    help="override FORECAST_DIRS: one or more LABEL=/path/to/inference/dir")
    ap.add_argument("--wpower-dir", type=Path, default=WPOWER_DIR,
                    help="where farms.csv / turbines.csv live")
    ap.add_argument("--region", default=REGION, choices=["BE", "UK", "all"])
    ap.add_argument("--power-var", default=POWER_VAR,
                    help="the LAM power channel: 'capacityfactor', or 'power' for MW/cell")
    ap.add_argument("--leads", type=int, nargs="+", default=LEAD_HOURS)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--limit", type=int, default=None, help="only the first N inits (smoke test)")
    args = ap.parse_args()

    if args.runs:
        runs: dict[str, Path] = {}
        for r in args.runs:
            if "=" not in r:
                raise SystemExit(f"--runs takes LABEL=DIR, got {r!r}")
            label, d = r.split("=", 1)
            runs[label] = Path(d)
    else:
        runs = dict(FORECAST_DIRS)
    args.out.mkdir(parents=True, exist_ok=True)

    farms_df = pd.read_csv(args.wpower_dir / "farms.csv")
    turbines = pd.read_csv(args.wpower_dir / "turbines.csv")
    farms = (farms_df.farm.tolist() if args.region == "all"
             else farms_df[farms_df.region == args.region].farm.tolist())
    turbines = turbines[turbines.farm.isin(farms)]
    cap = farms_df.set_index("farm").loc[farms, "capacity_mw"]
    print(f"{args.region}: {len(farms)} farms, {cap.sum():.1f} MW, {len(turbines)} turbines")

    in_vars = WIND_VARS + [args.power_var]
    L = len(args.leads)

    for label, d in runs.items():
        files = sorted(d.glob("forecast_*.nc"))
        if not files:
            print(f"\n{label:16s} NO forecast_*.nc in {d} -- skipping")
            continue
        if args.limit:
            files = files[:args.limit]
        print(f"\n{'=' * 78}\n{label}: {len(files)} forecast files in {d}\n{'=' * 78}")

        # ---- canonical cells from the first file that has every variable ----
        ref = None
        for f in files:
            with xr.open_dataset(f) as ds:
                if all(v in ds for v in in_vars):
                    cell_idx, G, cap_cell = build_reconstruction(
                        ds["latitude"].values, ds["longitude"].values, turbines, farms)
                    cell_lat = ds["latitude"].values[cell_idx].astype(np.float64)
                    cell_lon = to_180(ds["longitude"].values[cell_idx])
                    ref = f
                    break
        if ref is None:
            missing = sorted(set(in_vars) - set(xr.open_dataset(files[0]).variables))
            print(f"  no file has all of {in_vars} (first file lacks {missing}) -- skipping run")
            continue

        C = cell_idx.size
        shared = int((G > 0).sum(0).max())
        print(f"  reference grid: {ref.name}")
        print(f"  {C} cells, {cap_cell.sum():.1f} MW, up to {shared} farm(s) per cell, "
              f"{int((( G > 0).sum(0) > 1).sum())} shared")

        # ---- accumulate ----
        idx_cache: dict = {}
        inits: list[pd.Timestamp] = []
        blocks: list[np.ndarray] = []          # each (L, C, V)
        n_missing_var = n_missing_lead = 0

        for i, f in enumerate(files):
            init = parse_init(f)
            with xr.open_dataset(f) as ds:
                if not all(v in ds for v in in_vars):
                    n_missing_var += 1
                    continue

                key = grid_key(ds["latitude"].values, ds["longitude"].values)
                if key not in idx_cache:
                    idx_cache[key] = locate_cells(ds["latitude"].values,
                                                  ds["longitude"].values, cell_lat, cell_lon)
                cidx = idx_cache[key]

                fc_times = pd.DatetimeIndex(ds["time"].values)
                t2i = {t: j for j, t in enumerate(fc_times)}
                rows = [t2i.get(init + pd.Timedelta(hours=lh)) for lh in args.leads]
                if any(r is None for r in rows):
                    # drop the init entirely: every series must be built on one identical sample
                    n_missing_lead += 1
                    continue
                rows = np.asarray(rows, dtype=int)

                blk = np.empty((L, C, len(in_vars)), dtype=np.float32)
                for v, name in enumerate(in_vars):
                    blk[:, :, v] = ds[name].values[np.ix_(rows, cidx)]

            inits.append(init)
            blocks.append(blk)
            if i and i % 500 == 0:
                print(f"  read {i}/{len(files)} ({len(inits)} kept)", flush=True)

        if n_missing_var:
            print(f"  dropped {n_missing_var} init(s) missing a variable "
                  f"(heterogeneous inference dir -- check with check_forecast_vars.py)")
        if n_missing_lead:
            print(f"  dropped {n_missing_lead} init(s) not covering every lead in {args.leads}")
        if not inits:
            print("  nothing usable -- skipping run")
            continue

        arr = np.stack(blocks)                                   # (N, L, C, V)
        N = arr.shape[0]
        init_ix = pd.DatetimeIndex(inits)

        # the LAM power channel -> capacity factor, whichever way it was written
        cf = arr[:, :, :, -1]
        if args.power_var != "capacityfactor":
            cf = np.divide(cf, cap_cell[None, None, :].astype(np.float32),
                           out=np.full_like(cf, np.nan),
                           where=cap_cell[None, None, :] > 0)

        leads = np.asarray(args.leads, dtype=np.int32)
        valid = (init_ix.values[:, None]
                 + (leads.astype("timedelta64[h]"))[None, :]).astype("datetime64[ns]")

        ds_out = xr.Dataset(
            {name: (("init", "lead_time", "cell"), arr[:, :, :, v])
             for v, name in enumerate(WIND_VARS)}
            | {
                "cf_lam":     (("init", "lead_time", "cell"), cf),
                "G":          (("farm", "cell"), G),
                "cap_cell":   ("cell", cap_cell),
                "valid_time": (("init", "lead_time"), valid),
            },
            coords={
                "init": init_ix.values,
                "lead_time": leads,
                "cell": np.arange(C, dtype=np.int32),
                "farm": np.array(farms, dtype=object),
                "cell_lat": ("cell", cell_lat),
                "cell_lon": ("cell", cell_lon),
            },
            attrs={
                "description": "PowerTransformer inputs: 7 variables at the "
                               f"{args.region} CERRA cells, per init and lead time",
                "run": label,
                "source_dir": str(d),
                "region": args.region,
                "power_var": args.power_var,
                "reference_file": ref.name,
                "n_inits_dropped_missing_var": n_missing_var,
                "n_inits_dropped_missing_lead": n_missing_lead,
                "time_convention": "init and valid_time are naive UTC. power_obs at t is the "
                                   "mean MW over [t, t+3h) and LEADS the instantaneous forecast "
                                   "field by ~1.5 h -- handle when pairing with observations.",
                "reconstruction": "P_pred(farm,t) = sum_cell G[farm,cell] * CF_pred(cell,t)",
            },
        )
        # NB: not "hours since init" -- CF would read that back as a time encoding and the file
        # would fail to reopen.
        ds_out["lead_time"].attrs["units"] = "h"
        ds_out["lead_time"].attrs["long_name"] = "forecast lead time after the init"
        ds_out["G"].attrs["units"] = "MW"
        ds_out["G"].attrs["long_name"] = "capacity of farm f inside cell c"
        ds_out["cap_cell"].attrs["units"] = "MW"
        ds_out["cf_lam"].attrs["long_name"] = "LAM direct power forecast as a capacity factor"

        out_path = args.out / f"cells_{args.region}_{label}.nc"
        enc = {v: {"zlib": True, "complevel": 4}
               for v in list(WIND_VARS) + ["cf_lam", "G", "cap_cell"]}
        tmp = out_path.with_suffix(".nc.tmp")
        ds_out.to_netcdf(tmp, format="NETCDF4", engine="netcdf4", encoding=enc)
        tmp.replace(out_path)

        finite = np.isfinite(arr).all(axis=(1, 2, 3)).sum()
        print(f"  wrote {out_path}")
        print(f"  {N} inits x {L} leads x {C} cells x {len(in_vars)} vars   "
              f"({init_ix.min():%Y-%m-%d} .. {init_ix.max():%Y-%m-%d}, "
              f"{finite}/{N} inits fully finite)")
        print(f"  cf_lam: mean {np.nanmean(cf):.4f}  sd {np.nanstd(cf):.4f}  "
              f"min {np.nanmin(cf):.4f}  max {np.nanmax(cf):.4f}")


if __name__ == "__main__":
    main()
