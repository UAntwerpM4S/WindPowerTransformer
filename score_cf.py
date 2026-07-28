#!/usr/bin/env python3
"""Score power forecasts against observations — DIRECT, POWER CURVE, TRANSFORMER.

An extension of WindAI's score_power_configs.py: same reconstruction, power curves, metrics,
decomposition and plots, with a THIRD forecast method added — the cell-level model's
post-processed capacity factor produced by infer_cf.py:

  DIRECT       the LAM's own `capacityfactor`, reconstructed to farms via G           (solid)
  POWERCURVE   the LAM's ws100 pushed through each farm's specs power curve            (dashed)
  TRANSFORMER  infer_cf.py's predicted CF (model applied to forecast wind), via G      (dash-dot)

TRANSFORMER reconstructs identically to DIRECT: P(farm,t) = sum_cell G[farm,cell]*cf(cell,t).
The only difference is whose CF field it is — the LAM's native one (DIRECT) or the post-processor's
(TRANSFORMER) — so the comparison isolates what the learned power curve adds on top of the LAM.

Every method is scored on the init times common to all forecast runs AND all transformer files, so
the comparison uses one identical sample. Metrics per farm and for the regional total, by lead:
MAE (MW and % of capacity), RMSE, BIAS, and the MSE bias/amplitude/phase decomposition of the
total.

Usage:
  python score_cf.py                       # SETTINGS below
  python score_cf.py --region BE
  python score_cf.py --runs A=/path/one B=/path/two
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
# where infer_cf.py wrote cf_<REGION>_<RUN>.nc
TRANSFORMER_DIR = Path("/mnt/weatherloss/WindPowerTransformer/data/cf_forecasts")

VAR        = "capacityfactor"        # absent in a run -> POWERCURVE (+ TRANSFORMER) only
REGION     = "BE"
LEAD_HOURS = list(range(3, 37, 3))
OUT_DIR    = Path("cf_scores")
# --------------------------------------------------

FORECAST_RE = re.compile(r"forecast_(\d{14})")
FLEET_RE = re.compile(r"\s*(\d+)\s*x\s*(.+?)\s*$")

METHOD_LABEL = {"direct": "direct", "curve": "power curve", "transformer": "transformer"}
STYLE = {"direct": "-", "curve": "--", "transformer": "-."}


# =============================================================================
def to_180(lon):
    lon = np.asarray(lon, dtype=float)
    return ((lon + 180.0) % 360.0) - 180.0


def parse_init(path: Path) -> pd.Timestamp:
    return pd.to_datetime(FORECAST_RE.search(path.name).group(1),
                          format="%Y%m%d%H%M%S", utc=True)


def build_reconstruction(fc_lat, fc_lon, turbines, farms):
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


def farm_wind(ws_cells, G):
    w = G / G.sum(1, keepdims=True)
    return ws_cells @ w.T


def turbine_power(ws, cut_in, rated_ws, cut_out, rated_mw):
    ws = np.asarray(ws, dtype=float)
    out = np.zeros_like(ws)
    ramp = (ws >= cut_in) & (ws < rated_ws)
    out[ramp] = rated_mw * (ws[ramp] ** 3 - cut_in ** 3) / (rated_ws ** 3 - cut_in ** 3)
    out[(ws >= rated_ws) & (ws < cut_out)] = rated_mw
    return out


def build_farm_curves(farms_df, specs, farms):
    curves = {}
    meta = farms_df.set_index("farm")
    for farm in farms:
        fleet, cap = meta.loc[farm, "fleet"], float(meta.loc[farm, "capacity_mw"])
        parts = []
        for chunk in str(fleet).split(";"):
            m = FLEET_RE.match(chunk)
            if not m:
                raise SystemExit(f"{farm}: cannot parse fleet entry {chunk!r}")
            count, ttype = int(m.group(1)), m.group(2)
            if ttype not in specs.index:
                raise SystemExit(f"{farm}: turbine type {ttype!r} not in turbine_specs.csv")
            parts.append((count, specs.loc[ttype]))
        scale = cap / sum(c * float(s["rated_power_mw"]) for c, s in parts)

        def curve(ws, parts=parts, scale=scale):
            tot = np.zeros_like(np.asarray(ws, dtype=float))
            for count, s in parts:
                tot += count * turbine_power(ws, float(s["cut_in_ms"]), float(s["rated_ws_ms"]),
                                             float(s["cut_out_ms"]), float(s["rated_power_mw"]))
            return tot * scale
        curves[farm] = curve
    return curves


def load_transformer(tf_dir, region, labels):
    """Read cf_<region>_<label>.nc per run -> raw CF field + reconstruction op + init/lead index.

    Returns {label: dict(inits=tz-aware DatetimeIndex, cf=(N,L,C), G=(F,C), farms=[...],
    leads=(L,))}. Per-farm power is formed later (cf @ G.T) once the farm set is final.
    """
    tf = {}
    for label in labels:
        p = tf_dir / f"cf_{region}_{label}.nc"
        if not p.exists():
            continue
        with xr.open_dataset(p) as ds:
            inits = pd.DatetimeIndex(ds["init"].values)
            if inits.tz is None:
                inits = inits.tz_localize("UTC")
            tf[label] = dict(inits=inits, cf=ds["cf"].values.astype(np.float64),
                             G=ds["G"].values.astype(np.float64),
                             farms=[str(f) for f in ds["farm"].values],
                             leads=np.asarray(ds["lead_time"].values, dtype=int))
    return tf


# =============================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=None, metavar="LABEL=DIR")
    ap.add_argument("--region", default=REGION, choices=["BE", "UK", "all"])
    ap.add_argument("--var", default=VAR)
    ap.add_argument("--leads", type=int, nargs="+", default=LEAD_HOURS)
    ap.add_argument("--transformer-dir", type=Path, default=TRANSFORMER_DIR)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    if args.runs:
        runs = {}
        for r in args.runs:
            label, d = r.split("=", 1)
            runs[label] = Path(d)
    else:
        runs = dict(FORECAST_DIRS)
    args.out.mkdir(parents=True, exist_ok=True)

    farms_df = pd.read_csv(WPOWER_DIR / "farms.csv")
    turbines = pd.read_csv(WPOWER_DIR / "turbines.csv")
    obs = pd.read_csv(WPOWER_DIR / "power_obs.csv", index_col=0, parse_dates=True)
    specs = pd.read_csv(WPOWER_DIR / "turbine_specs.csv")
    specs = specs.rename(columns={specs.columns[0]: "turbine_type"}).set_index("turbine_type")
    if obs.index.tz is None:
        obs.index = obs.index.tz_localize("UTC")

    farms = (farms_df.farm.tolist() if args.region == "all"
             else farms_df[farms_df.region == args.region].farm.tolist())
    cap = farms_df.set_index("farm").loc[farms, "capacity_mw"]
    turbines = turbines[turbines.farm.isin(farms)]
    L = len(args.leads)
    lead_pos = {lh: k for k, lh in enumerate(args.leads)}
    print(f"{args.region}: {len(farms)} farms, {cap.sum():.0f} MW\n")

    # ---- file maps, restricted to init times present in EVERY run ----
    fmaps, has_power = {}, {}
    for label, d in runs.items():
        files = sorted(d.glob("forecast_*.nc"))
        if not files:
            print(f"{label:16s} NO forecast_*.nc in {d} -- skipping")
            continue
        with xr.open_dataset(files[0]) as ds0:
            hp, hw = args.var in ds0, "ws100" in ds0
        if not hw:
            print(f"{label:16s} no ws100 -- skipping")
            continue
        has_power[label] = hp
        fmaps[label] = {parse_init(f): f for f in files}
        print(f"{label:16s} {len(files):4d} files   "
              f"{'DIRECT + POWERCURVE' if hp else 'POWERCURVE only'}")

    if not fmaps:
        raise SystemExit("no usable runs")
    common = sorted(set.intersection(*(set(m) for m in fmaps.values())))
    print(f"\ncommon init times across {len(fmaps)} run(s): {len(common)}")

    # verify the power var is really present in every candidate init (heterogeneous dirs)
    direct_runs = [r for r in fmaps if has_power[r]]
    if direct_runs:
        good = []
        for init in common:
            ok = True
            for r in direct_runs:
                with xr.open_dataset(fmaps[r][init]) as ds:
                    if args.var not in ds:
                        ok = False
                        break
            if ok:
                good.append(init)
        if len(good) < len(common):
            print(f"  dropped {len(common)-len(good)} init(s) missing '{args.var}'")
        common = good

    # ---- transformer CF files -> restrict common to inits they cover (identical sample) ----
    tf = load_transformer(args.transformer_dir, args.region, list(fmaps))
    if tf:
        for label, t in tf.items():
            common = sorted(set(common) & set(t["inits"]))
        print(f"transformer runs: {list(tf)}  -> {len(common)} inits common to all methods")
    else:
        print("no transformer cf_*.nc found -- scoring DIRECT/POWERCURVE only")
    if not common:
        raise SystemExit("no init times common to all methods")

    # ---- drop farms with NO observations in the scored window ----
    valid_times = pd.DatetimeIndex(sorted({i + pd.Timedelta(hours=lh)
                                           for i in common for lh in args.leads}))
    valid_times = valid_times.intersection(obs.index)
    if len(valid_times) == 0:
        raise SystemExit("no forecast valid time overlaps power_obs.csv")
    counts = obs.loc[valid_times, farms].notna().sum()
    dead = [f for f in farms if counts[f] == 0]
    if dead:
        print(f"\n{len(dead)}/{len(farms)} farm(s) have NO obs in the window: {', '.join(dead)}")
        farms = [f for f in farms if counts[f] > 0]
        if not farms:
            raise SystemExit("no farm has observations in the window -- nothing to score")
        cap = farms_df.set_index("farm").loc[farms, "capacity_mw"]
        turbines = turbines[turbines.farm.isin(farms)]

    cap_np = cap.to_numpy()
    total_cap = float(cap.sum())
    curves = build_farm_curves(farms_df, specs, farms)
    F = len(farms)

    # ---- transformer: precompute per-init per-farm power aligned to the final farm set ----
    tf_power = {}   # label -> {init(tz-aware): (L, F) MW}, rows = args.leads, cols = farms
    for label, t in tf.items():
        try:
            fcol = [t["farms"].index(f) for f in farms]
        except ValueError as e:
            raise SystemExit(f"{label}: transformer file is missing farm {e}")
        lmap = [int(np.where(t["leads"] == lh)[0][0]) if lh in t["leads"] else None
                for lh in args.leads]
        i2row = {init: r for r, init in enumerate(t["inits"])}
        Pall = np.einsum("nlc,fc->nlf", t["cf"], t["G"])          # (N, L_t, F_t) MW
        d = {}
        for init in common:
            if init not in i2row:
                continue
            src = Pall[i2row[init]]                               # (L_t, F_t)
            arr = np.full((L, F), np.nan)
            for k, lt in enumerate(lmap):
                if lt is not None:
                    arr[k] = src[lt, fcol]
            d[init] = arr
        tf_power[label] = d

    # ---- accumulators ----
    series = ([(r, "direct") for r in fmaps if has_power[r]]
              + [(r, "curve") for r in fmaps]
              + [(r, "transformer") for r in tf_power])
    keys = list(series)
    z2 = lambda: np.zeros((F, L))
    sae = {s: z2() for s in keys}; sse = {s: z2() for s in keys}
    sbe = {s: z2() for s in keys}; n = {s: z2() for s in keys}
    sae_t = {s: np.zeros(L) for s in keys}; sse_t = {s: np.zeros(L) for s in keys}
    sbe_t = {s: np.zeros(L) for s in keys}; n_t = {s: np.zeros(L) for s in keys}
    sp_t = {s: np.zeros(L) for s in keys}; so_t = {s: np.zeros(L) for s in keys}
    spp_t = {s: np.zeros(L) for s in keys}; soo_t = {s: np.zeros(L) for s in keys}
    spo_t = {s: np.zeros(L) for s in keys}

    def accumulate(s, ppred, ptrue, k):
        ok = np.isfinite(ptrue) & np.isfinite(ppred)
        if ok.any():
            e = ppred[ok] - ptrue[ok]
            sae[s][ok, k] += np.abs(e); sse[s][ok, k] += e * e
            sbe[s][ok, k] += e;         n[s][ok, k] += 1
        if ok.all():
            pt, ot = ppred.sum(), ptrue.sum(); et = pt - ot
            sae_t[s][k] += abs(et); sse_t[s][k] += et * et
            sbe_t[s][k] += et;      n_t[s][k] += 1
            sp_t[s][k] += pt;   so_t[s][k] += ot
            spp_t[s][k] += pt * pt; soo_t[s][k] += ot * ot; spo_t[s][k] += pt * ot

    recon_cache = {}

    def get_recon(lat, lon):
        key = (lat.size, round(float(lat[0]), 4), round(float(lon[-1]), 4))
        if key not in recon_cache:
            recon_cache[key] = build_reconstruction(lat, lon, turbines, farms)
        return recon_cache[key]

    # ---- the runs: direct + curve + transformer ----
    for label, fmap in fmaps.items():
        hp = has_power[label]
        print(f"scoring {label} ({len(common)} inits)...")
        tfp = tf_power.get(label, {})
        for init in common:
            with xr.open_dataset(fmap[init]) as ds:
                cell_idx, G, cap_cell = get_recon(ds["latitude"].values, ds["longitude"].values)
                fc_times = pd.DatetimeIndex(ds["time"].values).tz_localize("UTC")
                ws_farm = farm_wind(ds["ws100"].values[:, cell_idx], G)
                have_var = hp and args.var in ds
                if have_var:
                    field = ds[args.var].values[:, cell_idx]
                    cf = field if args.var == "capacityfactor" else np.divide(
                        field, cap_cell[None, :], out=np.full_like(field, np.nan),
                        where=cap_cell[None, :] > 0)
                    p_direct_all = cf @ G.T
            p_curve_all = np.column_stack([curves[f](ws_farm[:, i]) for i, f in enumerate(farms)])
            p_tf = tfp.get(init)                                  # (L, F) or None

            t2i = {t: j for j, t in enumerate(fc_times)}
            for lh in args.leads:
                vt = init + pd.Timedelta(hours=lh)
                if vt not in t2i or vt not in obs.index:
                    continue
                j, k = t2i[vt], lead_pos[lh]
                ptrue = obs.loc[vt, farms].to_numpy(float)
                accumulate((label, "curve"), p_curve_all[j], ptrue, k)
                if have_var:
                    accumulate((label, "direct"), p_direct_all[j], ptrue, k)
                if p_tf is not None:
                    accumulate((label, "transformer"), p_tf[k], ptrue, k)

    # =========================================================================
    with np.errstate(invalid="ignore", divide="ignore"):
        mae = {s: sae[s] / n[s] for s in keys}
        rmse = {s: np.sqrt(sse[s] / n[s]) for s in keys}
        bias = {s: sbe[s] / n[s] for s in keys}
        nmae = {s: 100.0 * mae[s] / cap_np[:, None] for s in keys}
        mae_t = {s: sae_t[s] / n_t[s] for s in keys}
        rmse_t = {s: np.sqrt(sse_t[s] / n_t[s]) for s in keys}
        bias_t = {s: sbe_t[s] / n_t[s] for s in keys}
        nmae_t = {s: 100.0 * mae_t[s] / total_cap for s in keys}

    lbl = lambda s: f"{s[0]} · {METHOD_LABEL[s[1]]}"

    def decompose(s, k=None):
        sl = slice(None) if k is None else slice(k, k + 1)
        N = float(np.nansum(n_t[s][sl]))
        if N == 0:
            return dict(mse=np.nan, bias2=np.nan, amp=np.nan, phase=np.nan,
                        sd_p=np.nan, sd_o=np.nan, r=np.nan, var_ratio=np.nan)
        mp = np.nansum(sp_t[s][sl]) / N; mo = np.nansum(so_t[s][sl]) / N
        vp = max(np.nansum(spp_t[s][sl]) / N - mp * mp, 0.0)
        vo = max(np.nansum(soo_t[s][sl]) / N - mo * mo, 0.0)
        cov = np.nansum(spo_t[s][sl]) / N - mp * mo
        sd_p, sd_o = np.sqrt(vp), np.sqrt(vo)
        r = cov / (sd_p * sd_o) if sd_p > 0 and sd_o > 0 else np.nan
        return dict(mse=np.nansum(sse_t[s][sl]) / N, bias2=(mp - mo) ** 2,
                    amp=(sd_p - sd_o) ** 2,
                    phase=2 * sd_p * sd_o * (1 - r) if np.isfinite(r) else np.nan,
                    sd_p=sd_p, sd_o=sd_o, r=r, var_ratio=sd_p / sd_o if sd_o > 0 else np.nan)

    # ---- 1. total MAE by lead ----
    print(f"\nTOTAL {args.region} power — MAE as % of {total_cap:.0f} MW")
    hdr = "lead  " + "".join(f"{lbl(s):>26s}" for s in series)
    print(hdr); print("-" * len(hdr))
    for lh in args.leads:
        k = lead_pos[lh]
        print(f"+{lh:2d}h  " + "".join(f"{nmae_t[s][k]:25.2f}%" for s in series))

    # ---- 2. summary pooled over leads ----
    print(f"\nSUMMARY — {args.region} total, pooled over all leads")
    print(f"{'series':34s} {'MAE%':>7s} {'RMSE MW':>9s} {'bias MW':>9s}")
    print("-" * 63)
    summary_rows = []
    for s in series:
        N = np.nansum(n_t[s])
        if N == 0:
            continue
        m = np.nansum(sae_t[s]) / N; r = np.sqrt(np.nansum(sse_t[s]) / N)
        b = np.nansum(sbe_t[s]) / N
        print(f"{lbl(s):34s} {100*m/total_cap:6.2f}% {r:9.1f} {b:+9.1f}")
        d = decompose(s)
        summary_rows.append(dict(series=lbl(s), run=s[0], method=s[1], mae_mw=m,
                                 nmae_pct=100 * m / total_cap, rmse_mw=r, bias_mw=b,
                                 mse=d["mse"], bias2=d["bias2"], amplitude=d["amp"],
                                 phase=d["phase"], sd_pred=d["sd_p"], sd_obs=d["sd_o"],
                                 corr=d["r"], var_ratio=d["var_ratio"], n=int(N)))

    # =========================================================================
    # plots
    # =========================================================================
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {r: c for r, c in zip(fmaps, plt.cm.tab10.colors)}

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for s in series:
        ax.plot(args.leads, nmae_t[s], marker="o", ms=4, lw=1.8,
                color=colors[s[0]], ls=STYLE[s[1]], label=lbl(s))
    ax.set(xlabel="lead time [h]", ylabel="MAE [% of total capacity]",
           title=f"Total {args.region} power — MAE vs observations ({F} farms, {total_cap:.0f} MW)\n"
                 f"solid = direct, dashed = power curve, dash-dot = transformer")
    ax.set_xticks(args.leads); ax.grid(ls="--", alpha=0.5); ax.legend(fontsize=8)
    fig.tight_layout()
    p = args.out / f"mae_total_{args.region}.png"
    fig.savefig(p, dpi=140); plt.close(fig); print(f"\nsaved {p}")

    ncol = 5 if F > 6 else 3
    nrow = int(np.ceil(F / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.8 * nrow),
                            sharex=True, sharey=True, squeeze=False)
    for i, farm in enumerate(farms):
        ax = axs[i // ncol][i % ncol]
        for s in series:
            ax.plot(args.leads, nmae[s][i], marker="o", ms=3, lw=1.2,
                    color=colors[s[0]], ls=STYLE[s[1]], label=lbl(s))
        ax.set_title(f"{farm}\n{cap[farm]:.0f} MW", fontsize=8)
        ax.grid(ls="--", alpha=0.4)
    for j in range(F, nrow * ncol):
        axs[j // ncol][j % ncol].axis("off")
    axs[0][0].legend(fontsize=6)
    fig.supxlabel("lead time [h]"); fig.supylabel("MAE [% of farm capacity]")
    fig.suptitle(f"Per-farm power MAE — {args.region}  "
                 f"(solid = direct, dashed = power curve, dash-dot = transformer)")
    fig.tight_layout()
    p = args.out / f"mae_per_farm_{args.region}.png"
    fig.savefig(p, dpi=140); plt.close(fig); print(f"saved {p}")

    # ---- per-farm BIAS ----
    with np.errstate(invalid="ignore", divide="ignore"):
        nbias = {s: 100.0 * bias[s] / cap_np[:, None] for s in series}
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.8 * nrow),
                            sharex=True, sharey=True, squeeze=False)
    for i, farm in enumerate(farms):
        ax = axs[i // ncol][i % ncol]
        for s in series:
            ax.plot(args.leads, nbias[s][i], marker="o", ms=3, lw=1.2,
                    color=colors[s[0]], ls=STYLE[s[1]], label=lbl(s))
        ax.axhline(0.0, color="k", lw=1.0)
        ax.set_title(f"{farm}\n{cap[farm]:.0f} MW", fontsize=8)
        ax.grid(ls="--", alpha=0.4)
    for j in range(F, nrow * ncol):
        axs[j // ncol][j % ncol].axis("off")
    axs[0][0].legend(fontsize=6)
    fig.supxlabel("lead time [h]"); fig.supylabel("bias [% of farm capacity]")
    fig.suptitle(f"Per-farm power BIAS (forecast − observed) — {args.region}")
    fig.tight_layout()
    p = args.out / f"bias_per_farm_{args.region}.png"
    fig.savefig(p, dpi=140); plt.close(fig); print(f"saved {p}")

    # ---- MSE decomposition + variance ratio ----
    dec_all = {s: decompose(s) for s in series}
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.8))
    labs = [lbl(s) for s in series]
    b2 = np.array([dec_all[s]["bias2"] for s in series])
    am = np.array([dec_all[s]["amp"] for s in series])
    ph = np.array([dec_all[s]["phase"] for s in series])
    x = np.arange(len(labs))
    axL.bar(x, b2, label="bias²  (systematic)", color="tab:red")
    axL.bar(x, am, bottom=b2, label="amplitude  (σ mismatch)", color="tab:orange")
    axL.bar(x, ph, bottom=b2 + am, label="phase  (timing)", color="tab:blue")
    axL.set_xticks(x); axL.set_xticklabels(labs, rotation=30, ha="right", fontsize=7)
    axL.set_ylabel("MSE of the regional total [MW²]")
    axL.set_title("MSE decomposition, pooled over leads")
    axL.grid(ls="--", alpha=0.4, axis="y"); axL.legend(fontsize=8)
    for s in series:
        vr = [decompose(s, k)["var_ratio"] for k in range(L)]
        axR.plot(args.leads, vr, marker="o", ms=4, lw=1.8,
                 color=colors[s[0]], ls=STYLE[s[1]], label=lbl(s))
    axR.axhline(1.0, color="grey", ls="--", lw=1.2)
    axR.set(xlabel="lead time [h]", ylabel=r"$\sigma_{pred}\,/\,\sigma_{obs}$",
            title="Variance ratio of the regional total\nbelow 1 = under-dispersive")
    axR.set_xticks(args.leads); axR.grid(ls="--", alpha=0.5); axR.legend(fontsize=7)
    fig.tight_layout()
    p = args.out / f"decomposition_{args.region}.png"
    fig.savefig(p, dpi=140); plt.close(fig); print(f"saved {p}")

    print(f"\nMSE DECOMPOSITION — {args.region} total, pooled over leads")
    print(f"{'series':34s} {'RMSE':>8s} {'bias²':>9s} {'ampl':>9s} {'phase':>9s} "
          f"{'σp/σo':>7s} {'r':>6s}")
    print("-" * 88)
    for s in series:
        d = dec_all[s]
        if not np.isfinite(d["mse"]):
            continue
        print(f"{lbl(s):34s} {np.sqrt(d['mse']):8.1f} {d['bias2']:9.0f} {d['amp']:9.0f} "
              f"{d['phase']:9.0f} {d['var_ratio']:7.3f} {d['r']:6.3f}")

    # =========================================================================
    # CSVs
    # =========================================================================
    rows = []
    for s in keys:
        for lh in args.leads:
            k = lead_pos[lh]
            rows.append(dict(run=s[0], method=s[1], lead_hours=lh, scope="TOTAL",
                             mae_mw=mae_t[s][k], nmae_pct=nmae_t[s][k], rmse_mw=rmse_t[s][k],
                             bias_mw=bias_t[s][k], n=int(n_t[s][k])))
            for i, farm in enumerate(farms):
                rows.append(dict(run=s[0], method=s[1], lead_hours=lh, scope=farm,
                                 mae_mw=mae[s][i, k], nmae_pct=nmae[s][i, k],
                                 rmse_mw=rmse[s][i, k], bias_mw=bias[s][i, k],
                                 n=int(n[s][i, k])))
    pd.DataFrame(rows).to_csv(args.out / f"scores_{args.region}.csv", index=False)
    print(f"saved {args.out / f'scores_{args.region}.csv'}")
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(args.out / f"summary_{args.region}.csv", index=False)
        print(f"saved {args.out / f'summary_{args.region}.csv'}")


if __name__ == "__main__":
    main()
