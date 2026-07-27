#!/usr/bin/env python3
"""Step 2: attach the cell-level observed target to the extracted forecast inputs.

Reads `cells_<REGION>_<RUN>.nc` from extract_cells.py, distributes the observed farm power
onto the same cells, and writes ONE self-contained training file:

    dataset_<REGION>_<RUN>.nc
      inputs   ws100 ws10 wdir100_{cos,sin} wdir10_{cos,sin} cf_lam   (init, lead_time, cell)
      target   cf_obs                                                 (init, lead_time, cell)
      scoring  power_obs                                              (init, lead_time, farm)
      geometry G[farm, cell]  cap_cell[cell]  cell_lat  cell_lon
      split    split[init] in {train, val, test, unused}

The distribution is the one from WindAI/data/Wpower/build_power.py, expressed through the G
that extract_cells.py already stored (G[f,c] = capacity of farm f in cell c):

    share(f, c)  = G[f, c] / sum_c G[f, c]                    (each farm's shares sum to 1)
    power(c, t)  = SUM_f  P_obs(f, t) * share(f, c)           (extensive: SUM over farms)
    cf_obs(c, t) = power(c, t) / cap_cell(c)                  (in [0, 1])

NaN rule, same as build_power.py: power is extensive, so a cell is NaN at t if ANY farm
contributing to it is NaN at t -- you cannot sum a known and an unknown.  Belgium's cells are
heavily shared (12 of 15), so this NaNs more than you might expect; the valid fraction is printed.

TIME CONVENTION.  power_obs at t is the mean MW over [t, t+3h) and leads the instantaneous
forecast field by ~1.5 h.  This script pairs obs at the valid time with the forecast at that
valid time, applying NO shift -- exactly what build_power.py and score_power_configs.py do, so
these targets stay comparable with the published LAM numbers.  `--obs-shift-hours` exists to
test the alternative; it is not the default for that reason.

SPLIT LEAKAGE.  Inits are 3-hourly but each forecast covers 36 h, so inits on either side of a
split boundary share valid times.  `--gap-hours` (default 36) drops the inits inside that window
so no validation or test valid time was seen during training.

Usage:
  python build_targets.py                                    # every cells_*.nc in --cells-dir
  python build_targets.py --cells cells_BE_HighCapacityGT.nc
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# -------------------- SETTINGS --------------------
WPOWER_DIR = Path("/mnt/weatherloss/WindPower/data/WPDistr")
CELLS_DIR  = Path("/mnt/weatherloss/WindPowerTransformer/data/cells")

# split by INIT month, inclusive (from data/splitdates.py)
TRAIN_MONTHS = ((2024, 8), (2025, 3))
VAL_MONTHS   = ((2025, 4), (2025, 5))
TEST_MONTHS  = ((2025, 6), (2025, 7))
GAP_HOURS    = 36            # forecast length: the overlap a split boundary has to clear

CF_MAX = 1.05                # same guard as build_power.py: above this, capacity/obs disagree
# --------------------------------------------------

WIND_VARS = ["ws100", "ws10", "wdir100_cos", "wdir100_sin", "wdir10_cos", "wdir10_sin"]


def paint_power(P: np.ndarray, share: np.ndarray) -> np.ndarray:
    """P (T, F) farm power with NaNs, share (F, C) capacity shares -> power (T, C).

    Extensive SUM over farms with the strict NaN rule: a cell is NaN at t if any farm feeding
    it (share > 0) is NaN at t.  Mirrors build_power.py's paint_power(drop_nan=False), which
    takes W (C, F); here the operator arrives as share (F, C), so it is used untransposed.
    """
    finite = np.isfinite(P)
    power = np.where(finite, P, 0.0) @ share                     # (T, C), NaN farms count as 0
    nan_hits = (~finite).astype(float) @ (share > 0)             # (T, C) NaN contributors per cell
    return np.where(nan_hits > 0, np.nan, power)


def month_in(idx: pd.DatetimeIndex, lo: tuple[int, int], hi: tuple[int, int]) -> np.ndarray:
    ym = idx.year * 12 + idx.month
    return (ym >= lo[0] * 12 + lo[1]) & (ym <= hi[0] * 12 + hi[1])


def assign_split(inits: pd.DatetimeIndex, train, val, test, gap_hours: int) -> np.ndarray:
    """Label each init train/val/test/unused, then drop inits that straddle a boundary.

    An init is dropped if its forecast window [init, init + gap] reaches into a different
    split's window, which would leak valid times across the boundary.
    """
    split = np.full(len(inits), "unused", dtype=object)
    for name, rng in (("train", train), ("val", val), ("test", test)):
        split[month_in(inits, *rng)] = name

    gap = pd.Timedelta(hours=gap_hours)
    keep = np.ones(len(inits), dtype=bool)
    for i, (t0, s) in enumerate(zip(inits, split)):
        if s == "unused":
            continue
        # any init whose own forecast window overlaps this one, but in another split
        near = (inits > t0 - gap) & (inits < t0 + gap)
        if (split[near] != s).any():
            keep[i] = False
    split[~keep] = "unused"
    return split


def build_one(cells_path: Path, obs: pd.DataFrame, args) -> None:
    print(f"\n{'=' * 78}\n{cells_path.name}\n{'=' * 78}")
    ds = xr.open_dataset(cells_path)

    farms = [str(f) for f in ds["farm"].values]
    G = ds["G"].values                                            # (F, C) MW
    cap_cell = ds["cap_cell"].values                              # (C,)
    N, L, C = ds.sizes["init"], ds.sizes["lead_time"], ds.sizes["cell"]
    F = len(farms)

    if not np.allclose(cap_cell, G.sum(0)):
        raise ValueError("cap_cell != G.sum(0): the cells file is inconsistent")
    missing = [f for f in farms if f not in obs.columns]
    if missing:
        raise ValueError(f"farms absent from power_obs.csv: {missing}")

    farm_cap = G.sum(1, keepdims=True)                            # (F, 1)
    if (farm_cap <= 0).any():
        raise ValueError("a farm has zero capacity in G -- no cell holds its turbines")
    share = G / farm_cap                                          # (F, C), rows sum to 1
    if not np.allclose(share.sum(1), 1.0):
        raise ValueError("capacity shares do not sum to 1 per farm")

    # ---- observations on the (init, lead) valid-time grid ----
    valid = pd.DatetimeIndex(ds["valid_time"].values.ravel())
    if args.obs_shift_hours:
        valid = valid + pd.Timedelta(hours=args.obs_shift_hours)
    P_flat = obs.reindex(valid)[farms].to_numpy(dtype=float)      # (N*L, F)
    n_hit = int(np.isfinite(P_flat).any(1).sum())
    if n_hit == 0:
        raise ValueError("no forecast valid time matches power_obs.csv -- check the periods")
    print(f"  {n_hit}/{len(valid)} valid times have >=1 reporting farm")

    power_cells = paint_power(P_flat, share)                      # (N*L, C)
    with np.errstate(invalid="ignore", divide="ignore"):
        cf_obs = power_cells / cap_cell[None, :]

    cf_max = float(np.nanmax(cf_obs)) if np.isfinite(cf_obs).any() else np.nan
    if cf_max > CF_MAX:
        raise ValueError(f"observed capacity factor {cf_max:.3f} > {CF_MAX} -- capacity/obs mismatch")
    print(f"  cf_obs: valid fraction {np.isfinite(cf_obs).mean():.4f}  max {cf_max:.3f}  "
          f"mean {np.nanmean(cf_obs):.4f}  sd {np.nanstd(cf_obs):.4f}")

    cf_obs = cf_obs.reshape(N, L, C).astype(np.float32)
    P_farm = P_flat.reshape(N, L, F).astype(np.float32)

    # a quick read on what the post-processor has to beat
    cf_lam = ds["cf_lam"].values
    both = np.isfinite(cf_obs) & np.isfinite(cf_lam)
    if both.any():
        o, p = cf_obs[both], cf_lam[both]
        print(f"  cell-level LAM vs obs: bias {np.mean(p - o):+.4f}  MAE {np.mean(np.abs(p - o)):.4f}  "
              f"sd_p/sd_o {p.std() / o.std():.3f}  r {np.corrcoef(p, o)[0, 1]:.3f}")

    # ---- split ----
    inits = pd.DatetimeIndex(ds["init"].values)
    split = assign_split(inits, args.train_months, args.val_months, args.test_months,
                         args.gap_hours)
    for name in ("train", "val", "test", "unused"):
        m = split == name
        if m.any():
            sel = inits[m]
            usable = np.isfinite(cf_obs[m]).any(axis=(1, 2)).sum()
            print(f"  {name:7s} {m.sum():5d} inits  {sel.min():%Y-%m-%d} .. {sel.max():%Y-%m-%d}"
                  f"   ({usable} with any finite target)")
        else:
            print(f"  {name:7s}     0 inits")

    # ---- write ----
    out = ds.assign(
        cf_obs=(("init", "lead_time", "cell"), cf_obs),
        power_obs=(("init", "lead_time", "farm"), P_farm),
        split=("init", split.astype(str)),
    )
    out["cf_obs"].attrs = {
        "long_name": "observed capacity factor per cell (the training target)",
        "note": "NaN if any farm contributing to the cell is NaN at that valid time",
    }
    out["power_obs"].attrs = {"units": "MW", "long_name": "observed farm power at the valid time"}
    out["split"].attrs = {"long_name": "train/val/test by init month, boundary inits dropped",
                          "gap_hours": args.gap_hours}
    out.attrs["obs_shift_hours"] = args.obs_shift_hours
    out.attrs["target"] = ("cf_obs(c,t) = sum_f P_obs(f,t) * G[f,c]/sum_c G[f,c] / cap_cell(c); "
                          "a cell is NaN if any contributing farm is NaN")
    out.attrs["splits"] = (f"train {args.train_months} | val {args.val_months} | "
                           f"test {args.test_months} | gap {args.gap_hours} h")

    enc = {v: {"zlib": True, "complevel": 4}
           for v in list(WIND_VARS) + ["cf_lam", "cf_obs", "power_obs", "G", "cap_cell"]}
    out_path = args.out / cells_path.name.replace("cells_", "dataset_")
    tmp = out_path.with_suffix(".nc.tmp")
    out.to_netcdf(tmp, format="NETCDF4", engine="netcdf4", encoding=enc)
    tmp.replace(out_path)
    ds.close()
    print(f"  wrote {out_path}  ({N} inits x {L} leads x {C} cells)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", type=Path, nargs="+", default=None,
                    help="cells_*.nc from extract_cells.py (default: all in --cells-dir)")
    ap.add_argument("--cells-dir", type=Path, default=CELLS_DIR)
    ap.add_argument("--wpower-dir", type=Path, default=WPOWER_DIR, help="where power_obs.csv lives")
    ap.add_argument("--out", type=Path, default=None, help="default: alongside the cells files")
    ap.add_argument("--obs-shift-hours", type=float, default=0.0,
                    help="shift the obs clock before pairing; 0 matches the scorer (see docstring)")
    ap.add_argument("--gap-hours", type=int, default=GAP_HOURS,
                    help="drop inits within this many hours of a split boundary (0 disables)")
    ap.add_argument("--train-months", type=int, nargs=4, default=None, metavar=("Y1", "M1", "Y2", "M2"))
    ap.add_argument("--val-months", type=int, nargs=4, default=None, metavar=("Y1", "M1", "Y2", "M2"))
    ap.add_argument("--test-months", type=int, nargs=4, default=None, metavar=("Y1", "M1", "Y2", "M2"))
    args = ap.parse_args()

    pair = lambda v, d: ((v[0], v[1]), (v[2], v[3])) if v else d          # noqa: E731
    args.train_months = pair(args.train_months, TRAIN_MONTHS)
    args.val_months   = pair(args.val_months, VAL_MONTHS)
    args.test_months  = pair(args.test_months, TEST_MONTHS)

    files = args.cells or sorted(args.cells_dir.glob("cells_*.nc"))
    if not files:
        raise SystemExit(f"no cells_*.nc in {args.cells_dir} -- run extract_cells.py first")
    args.out = args.out or files[0].parent
    args.out.mkdir(parents=True, exist_ok=True)

    obs = pd.read_csv(args.wpower_dir / "power_obs.csv", index_col=0, parse_dates=True)
    if obs.index.tz is not None:
        obs.index = obs.index.tz_convert(None)
    print(f"power_obs.csv: {len(obs)} rows x {obs.shape[1]} farms "
          f"({obs.index.min():%Y-%m-%d} .. {obs.index.max():%Y-%m-%d})")

    for f in files:
        build_one(f, obs, args)


if __name__ == "__main__":
    main()
