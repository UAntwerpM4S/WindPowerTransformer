#!/usr/bin/env python3
"""Score the forecast-tier post-processors on the HELD-OUT TEST split — one model per run.

For each forecasting run, three power forecasts are reconstructed to farms and scored against the
observations, on the run's own `test` inits only (the chronological hold-out from build_targets):

  DIRECT       the LAM's own `cf_lam`, reconstructed via G                              (solid)
  POWERCURVE   the LAM's ws100 pushed through each farm's specs power curve             (dashed)
  CERRA        one shared converter fitted on CERRA ANALYSIS wind -> observed cf over the LAM's
               own training years, applied unchanged to every run's forecast wind. Never saw a
               forecast, so it cannot favour a run: use THIS to compare runs' wind quality.
  TRANSFORMER  infer_cf.py's post-processed CF (model on wind + cf_lam + static), via G (dash-dot).
               One model PER RUN, so it absorbs each run's own bias -- good for "how well can this
               run be post-processed", wrong for "whose wind is better".

All three come from the SAME dataset_BE_<RUN>.nc, so they share the exact valid times, obs pairing
and cell geometry — the comparison isolates whether post-processing the run's forecast beats using
its raw power head or the physical power curve. Metrics per farm and for the regional total, by
lead: MAE (MW and % of capacity), RMSE, BIAS, and the MSE bias/amplitude/phase decomposition.

    P(farm,t) = sum_cell G[farm,cell] * cf(cell,t)          (direct: cf=cf_lam; transformer: cf=model)

Usage:  python score_cf.py            # SETTINGS below
        python score_cf.py --runs HighCapacityGT VanillaPowerGT
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# -------------------- SETTINGS --------------------
WPOWER_DIR = Path("/mnt/weatherloss/WindPower/data/WPDistr")
CELLS_DIR  = Path("/mnt/weatherloss/WindPowerTransformer/data/cells")            # dataset_BE_<RUN>.nc
TRANSFORMER_DIR = Path("/mnt/weatherloss/WindPowerTransformer/data/cf_forecasts")  # cf_BE_<RUN>.nc

# LAM runs whose dataset_BE_<run>.nc supplies obs + direct (cf_lam) + power curve (ws100).
# RegularWeather has no cf_lam -> it contributes a power curve + a wind-only transformer only.
SOURCE_RUNS = ["HighCapacityGT", "VanillaPowerGT", "VeryHighCapacityGT", "RegularWeather"]

# transformer forecasts to score: (cf file tag, source LAM run, method name in the plot).
# 'transformer' = wind + cf_lam ; 'transformer_wo' = wind-only (dash-dot vs dotted).
TRANSFORMERS = [
    # CERRA tier: ONE converter (trained on CERRA analysis wind, 2020..2024-01) applied to each
    # run's forecast wind. Same instrument for every run, so differences are purely the WIND --
    # the calibrated replacement for the shared specs power curve. Files are cf_BE_<tag>@<run>.nc.
    ("CERRAcell@HighCapacityGT",     "HighCapacityGT",     "cerra"),
    ("CERRAcell@VanillaPowerGT",     "VanillaPowerGT",     "cerra"),
    ("CERRAcell@VeryHighCapacityGT", "VeryHighCapacityGT", "cerra"),
    ("CERRAcell@RegularWeather",     "RegularWeather",     "cerra"),
    # forecast tier: one post-processor per run, trained on that run's own forecasts
    ("HighCapacityGT",        "HighCapacityGT",     "transformer"),
    ("VanillaPowerGT",        "VanillaPowerGT",     "transformer"),
    ("VeryHighCapacityGT",    "VeryHighCapacityGT", "transformer"),
    ("HighCapacityGT_wo",     "HighCapacityGT",     "transformer_wo"),
    ("VanillaPowerGT_wo",     "VanillaPowerGT",     "transformer_wo"),
    ("VeryHighCapacityGT_wo", "VeryHighCapacityGT", "transformer_wo"),
    ("RegularWeather",        "RegularWeather",     "transformer_wo"),
]
REGION = "BE"
SPLIT  = "test"                      # which split to report on

# Wind-regime edges in m/s, converted to capacity-factor thresholds through each farm's own power
# curve, then applied to the OBSERVED power -- truth-conditioned, labelled in m/s so the bins match
# score_power_configs.py and score_wpx_power.py exactly.
REGIME_WS_EDGES = [4.5, 8.0, 12.0]
OUT_DIR = Path("cf_scores")

# which method families to score/plot (override on the CLI with --methods)
# 'cerra' is the shared CERRA-fitted converter -- the one to use when the question is
# "whose WIND is better"; the transformer_* families are per-run models and therefore absorb each
# run's own biases, which compresses exactly the differences the cerra tier is there to measure.
METHODS = ["direct", "curve", "cerra", "transformer", "transformer_wo"]

# which season to restrict the scored inits to (by init month). "all" = no restriction.
SEASON  = "all"
SEASONS = {"all": None, "DJF": {12, 1, 2}, "MAM": {3, 4, 5}, "JJA": {6, 7, 8}, "SON": {9, 10, 11}}
# --------------------------------------------------

FLEET_RE = re.compile(r"\s*(\d+)\s*x\s*(.+?)\s*$")


def cf_edges_from_ws(curve, capacity_mw, ws_edges):
    """CF thresholds equivalent to wind-speed edges, for ONE power curve.

    Identical to the helper in score_power_configs.py and score_wpx_power.py, so all three
    scripts bin on the same boundaries and their regime tables are directly comparable.
    """
    cf = np.asarray([float(curve(np.asarray([e], dtype=float))[0]) / capacity_mw
                     for e in ws_edges], dtype=float)
    for i in range(1, len(cf)):
        if cf[i] <= cf[i - 1]:
            cf[i] = cf[i - 1] + 1e-9
    return cf
METHOD_LABEL = {"direct": "direct", "curve": "power curve (specs)",
                "cerra": "CERRA converter (shared)",
                "transformer": "transformer (wind+cf_lam)", "transformer_wo": "transformer (wind-only)"}
STYLE = {"direct": "-", "curve": "--", "cerra": (0, (3, 1, 1, 1, 1, 1)),
         "transformer": "-.", "transformer_wo": ":"}


# =============================================================================
# per-farm aggregate power curve (specs -> P(v))
# =============================================================================
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


def farm_power_curve(ws100, G, curves, farms):
    """ws100 (N,L,C) -> per-farm power (N,L,F) via capacity-weighted farm wind + specs curve."""
    w = G / G.sum(1, keepdims=True)                                    # (F,C) shares sum to 1
    ws_farm = np.einsum("nlc,fc->nlf", ws100, w)                       # (N,L,F)
    return np.stack([curves[f](ws_farm[:, :, i]) for i, f in enumerate(farms)], axis=-1)


# =============================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-runs", nargs="+", default=SOURCE_RUNS)
    ap.add_argument("--region", default=REGION)
    ap.add_argument("--split", default=SPLIT, help="test | all (all = every init, e.g. k-fold OOF)")
    ap.add_argument("--per-run-inits", action="store_true",
                    help="score each run on its OWN inits (default: intersect to common inits so "
                         "the cross-run comparison is on one identical sample)")
    ap.add_argument("--methods", nargs="+", default=METHODS,
                    choices=list(METHOD_LABEL),          # derived, so a new method needs one edit
                    help=f"which method families to include (default: {' '.join(METHODS)})")
    ap.add_argument("--season", default=SEASON, choices=list(SEASONS),
                    help="restrict scored inits to a season by init month (default: all)")
    ap.add_argument("--cells-dir", type=Path, default=CELLS_DIR)
    ap.add_argument("--transformer-dir", type=Path, default=TRANSFORMER_DIR)
    ap.add_argument("--wpower-dir", type=Path, default=WPOWER_DIR)
    ap.add_argument("--regime-ws-edges", type=float, nargs="+", default=REGIME_WS_EDGES,
                    help="interior wind-speed edges in m/s for the regime split (default 4.5 8 12)")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    enabled = set(args.methods)
    season_months = SEASONS[args.season]                      # None = all months
    scope = args.split + ("" if season_months is None else f", {args.season}")
    sfx = "" if season_months is None else f"_{args.season}"  # output-filename suffix
    print(f"scoring methods: {', '.join(args.methods)} | season: {args.season}")

    farms_df = pd.read_csv(args.wpower_dir / "farms.csv")
    specs = pd.read_csv(args.wpower_dir / "turbine_specs.csv")
    specs = specs.rename(columns={specs.columns[0]: "turbine_type"}).set_index("turbine_type")

    # ---- per source run: dataset gives obs + direct(cf_lam) + power curve; store for transformers ----
    farms = leads = cap_np = None
    total_cap = 0.0
    curves = None
    # method_pred[(source_run, method)] = (P_pred (N,L,F), obs (N,L,F), test-mask (N,))
    method_pred = {}
    runs_data = {}          # source_run -> dict(G, obs, test, init)
    present_runs = []

    for run in args.source_runs:
        dpath = args.cells_dir / f"dataset_{args.region}_{run}.nc"
        if not dpath.exists():
            print(f"{run:22s} {dpath} missing -- skipping")
            continue
        ds = xr.open_dataset(dpath)
        f_run = [str(f) for f in ds["farm"].values]
        l_run = np.asarray(ds["lead_time"].values, dtype=int)
        if farms is None:
            farms, leads = f_run, l_run
            cap = farms_df.set_index("farm").loc[farms, "capacity_mw"]
            cap_np, total_cap = cap.to_numpy(), float(cap.sum())
            curves = build_farm_curves(farms_df, specs, farms)
        elif f_run != farms:
            raise SystemExit(f"{run}: farm set differs from the first run -- cannot compare")

        G = ds["G"].values
        obs = ds["power_obs"].values.astype(np.float64)                # (N,L,F)
        test = (np.ones(ds.sizes["init"], bool) if args.split == "all"
                else ds["split"].values.astype(str) == args.split)
        init = pd.DatetimeIndex(ds["init"].values)
        runs_data[run] = dict(G=G, obs=obs, test=test, init=init)
        present_runs.append(run)

        if "cf_lam" in ds and "direct" in enabled:
            method_pred[(run, "direct")] = (np.einsum("nlc,fc->nlf", ds["cf_lam"].values, G),
                                            obs, test)
        if "curve" in enabled:
            method_pred[(run, "curve")] = (farm_power_curve(ds["ws100"].values, G, curves, farms),
                                           obs, test)
        n_te = int(test.sum())
        print(f"{run:22s} {len(init):5d} inits, {n_te} in '{args.split}'  "
              f"({init[test].min():%Y-%m-%d}..{init[test].max():%Y-%m-%d})" if n_te else
              f"{run:22s} {len(init):5d} inits, 0 in '{args.split}'")
        ds.close()

    # ---- transformers: reconstruct each cf_<tag>.nc and attach to its source run's obs/test ----
    for tag, run, method in TRANSFORMERS:
        if method not in enabled or run not in runs_data:
            continue
        tfp = args.transformer_dir / f"cf_{args.region}_{tag}.nc"
        if not tfp.exists():
            print(f"{tag:22s} no cf_{args.region}_{tag}.nc -- run infer_cf.py; skipped")
            continue
        d = runs_data[run]
        with xr.open_dataset(tfp) as tf:
            tf_init = pd.DatetimeIndex(tf["init"].values)
            cf_tf = tf["cf"].values.astype(np.float64)
        row = {t: i for i, t in enumerate(tf_init)}
        rows = np.array([row.get(t, -1) for t in d["init"]])
        cf_al = np.full((len(d["init"]),) + cf_tf.shape[1:], np.nan, dtype=np.float64)
        ok = rows >= 0
        cf_al[ok] = cf_tf[rows[ok]]
        method_pred[(run, method)] = (np.einsum("nlc,fc->nlf", cf_al, d["G"]), d["obs"], d["test"])

    if not method_pred:
        raise SystemExit("nothing to score -- no dataset_*.nc found")

    # ---- one identical sample across runs: intersect the scored inits (default) ----
    def in_season(run):
        if season_months is None:
            return np.ones(len(runs_data[run]["init"]), bool)
        return np.isin(runs_data[run]["init"].month, list(season_months))

    if args.per_run_inits:
        eff_mask = {r: runs_data[r]["test"] & in_season(r) for r in present_runs}
    else:
        sets = [set(runs_data[r]["init"][runs_data[r]["test"]]) for r in present_runs]
        common = pd.DatetimeIndex(sorted(set.intersection(*sets))) if sets else pd.DatetimeIndex([])
        eff_mask = {r: runs_data[r]["init"].isin(common) & in_season(r) for r in present_runs}
        print(f"\ncommon inits across {len(present_runs)} runs: {len(common)} "
              f"({common.min():%Y-%m-%d}..{common.max():%Y-%m-%d})" if len(common) else
              f"\ncommon inits across {len(present_runs)} runs: 0  <-- runs share no inits!")
    if season_months is not None:
        scored = set()
        for r in present_runs:
            scored |= set(runs_data[r]["init"][eff_mask[r]])
        print(f"season {args.season}: {len(scored)} distinct inits scored")

    F, L = len(farms), len(leads)
    series = [k for k in method_pred]

    # ---- wind regimes: truth-conditioned, expressed in m/s ----
    ws_edges = list(args.regime_ws_edges)
    reg_labels = ([f"0-{ws_edges[0]:g}"]
                  + [f"{ws_edges[i]:g}-{ws_edges[i+1]:g}" for i in range(len(ws_edges) - 1)]
                  + [f"{ws_edges[-1]:g}+"])
    R = len(reg_labels)
    cf_thr_tot = cf_edges_from_ws(lambda ws: sum(curves[f](ws) for f in farms),
                                  total_cap, ws_edges)
    print(f"\nwind regimes: " + " | ".join(reg_labels) + "  m/s   (binned on OBSERVED power "
          f"through the fleet curve)")
    print("  fleet thresholds: "
          + ", ".join(f"{e:g} m/s -> CF {c:.3f}" for e, c in zip(ws_edges, cf_thr_tot)))

    # ---- accumulators ----
    z2 = lambda: np.zeros((F, L))
    sae = {s: z2() for s in series}; sse = {s: z2() for s in series}
    sbe = {s: z2() for s in series}; n = {s: z2() for s in series}
    sae_t = {s: np.zeros(L) for s in series}; sse_t = {s: np.zeros(L) for s in series}
    sbe_t = {s: np.zeros(L) for s in series}; n_t = {s: np.zeros(L) for s in series}
    sp_t = {s: np.zeros(L) for s in series}; so_t = {s: np.zeros(L) for s in series}
    spp_t = {s: np.zeros(L) for s in series}; soo_t = {s: np.zeros(L) for s in series}
    spo_t = {s: np.zeros(L) for s in series}
    # regime split of the regional total: (L, R)
    sae_tr = {s: np.zeros((L, R)) for s in series}
    sse_tr = {s: np.zeros((L, R)) for s in series}
    sbe_tr = {s: np.zeros((L, R)) for s in series}
    n_tr = {s: np.zeros((L, R)) for s in series}

    def accumulate(s, ppred, ptrue, k, rt=None):
        ok = np.isfinite(ptrue) & np.isfinite(ppred)
        if ok.any():
            e = ppred[ok] - ptrue[ok]
            sae[s][ok, k] += np.abs(e); sse[s][ok, k] += e * e
            sbe[s][ok, k] += e;         n[s][ok, k] += 1
        if ok.all():                                  # regional total: every farm reporting
            pt, ot = ppred.sum(), ptrue.sum(); et = pt - ot
            sae_t[s][k] += abs(et); sse_t[s][k] += et * et
            sbe_t[s][k] += et;      n_t[s][k] += 1
            sp_t[s][k] += pt;   so_t[s][k] += ot
            spp_t[s][k] += pt * pt; soo_t[s][k] += ot * ot; spo_t[s][k] += pt * ot
            if rt is not None:
                sae_tr[s][k, rt] += abs(et); sse_tr[s][k, rt] += et * et
                sbe_tr[s][k, rt] += et;      n_tr[s][k, rt] += 1

    for s, (P, obs, _test) in method_pred.items():
        for nidx in np.where(eff_mask[s[0]])[0]:
            for k in range(L):
                o = obs[nidx, k]
                # regime from the OBSERVED fleet total, so every method shares the same bins
                rt = (int(np.digitize(o.sum() / total_cap, cf_thr_tot))
                      if np.isfinite(o).all() else None)
                accumulate(s, P[nidx, k], o, k, rt)

    # ---- metrics ----
    with np.errstate(invalid="ignore", divide="ignore"):
        mae = {s: sae[s] / n[s] for s in series}
        rmse = {s: np.sqrt(sse[s] / n[s]) for s in series}
        bias = {s: sbe[s] / n[s] for s in series}
        nmae = {s: 100.0 * mae[s] / cap_np[:, None] for s in series}
        mae_t = {s: sae_t[s] / n_t[s] for s in series}
        rmse_t = {s: np.sqrt(sse_t[s] / n_t[s]) for s in series}
        bias_t = {s: sbe_t[s] / n_t[s] for s in series}
        nmae_t = {s: 100.0 * mae_t[s] / total_cap for s in series}

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
        return dict(mse=np.nansum(sse_t[s][sl]) / N, bias2=(mp - mo) ** 2, amp=(sd_p - sd_o) ** 2,
                    phase=2 * sd_p * sd_o * (1 - r) if np.isfinite(r) else np.nan,
                    sd_p=sd_p, sd_o=sd_o, r=r, var_ratio=sd_p / sd_o if sd_o > 0 else np.nan)

    # ---- 1. total MAE by lead ----
    # runs x methods gives well over ten columns; full labels ran off the line and the header
    # became unreadable, so number the columns and print the legend above.
    print(f"\nTOTAL {args.region} power ({scope}) — MAE as % of {total_cap:.0f} MW")
    for i, s in enumerate(series, 1):
        print(f"  [{i:2d}] {lbl(s)}")
    hdr = "lead  " + "".join(f"{f'[{i}]':>9s}" for i in range(1, len(series) + 1))
    print(hdr); print("-" * len(hdr))
    for li, lh in enumerate(leads):
        print(f"+{lh:2d}h  " + "".join(f"{nmae_t[s][li]:8.2f}%" for s in series))

    # ---- 2. summary pooled over leads ----
    print(f"\nSUMMARY — {args.region} total ({scope}), pooled over leads")
    print(f"{'series':36s} {'MAE%':>7s} {'RMSE MW':>9s} {'bias MW':>9s}")
    print("-" * 65)
    summary_rows = []
    for s in series:
        N = np.nansum(n_t[s])
        if N == 0:
            continue
        m = np.nansum(sae_t[s]) / N; r = np.sqrt(np.nansum(sse_t[s]) / N); b = np.nansum(sbe_t[s]) / N
        print(f"{lbl(s):36s} {100*m/total_cap:6.2f}% {r:9.1f} {b:+9.1f}")
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
    colors = {r: c for r, c in zip(present_runs, plt.cm.tab10.colors)}

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for s in series:
        ax.plot(leads, nmae_t[s], marker="o", ms=4, lw=1.8,
                color=colors[s[0]], ls=STYLE[s[1]], label=lbl(s))
    ax.set(xlabel="lead time [h]", ylabel="MAE [% of total capacity]",
           title=f"Total {args.region} power ({scope}) — {F} farms, {total_cap:.0f} MW\n"
                 f"solid = direct, dashed = power curve, dash-dot = transformer")
    ax.set_xticks(leads); ax.grid(ls="--", alpha=0.5); ax.legend(fontsize=8)
    fig.tight_layout()
    p = args.out / f"mae_total_{args.region}{sfx}.png"
    fig.savefig(p, dpi=140); plt.close(fig); print(f"\nsaved {p}")

    ncol = 5 if F > 6 else 3
    nrow = int(np.ceil(F / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.8 * nrow),
                            sharex=True, sharey=True, squeeze=False)
    for i, farm in enumerate(farms):
        ax = axs[i // ncol][i % ncol]
        for s in series:
            ax.plot(leads, nmae[s][i], marker="o", ms=3, lw=1.2,
                    color=colors[s[0]], ls=STYLE[s[1]], label=lbl(s))
        ax.set_title(f"{farm}\n{cap_np[i]:.0f} MW", fontsize=8)
        ax.grid(ls="--", alpha=0.4)
    for j in range(F, nrow * ncol):
        axs[j // ncol][j % ncol].axis("off")
    axs[0][0].legend(fontsize=6)
    fig.supxlabel("lead time [h]"); fig.supylabel("MAE [% of farm capacity]")
    fig.suptitle(f"Per-farm power MAE ({scope}) — {args.region}")
    fig.tight_layout()
    p = args.out / f"mae_per_farm_{args.region}{sfx}.png"
    fig.savefig(p, dpi=140); plt.close(fig); print(f"saved {p}")

    with np.errstate(invalid="ignore", divide="ignore"):
        nbias = {s: 100.0 * bias[s] / cap_np[:, None] for s in series}
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.8 * nrow),
                            sharex=True, sharey=True, squeeze=False)
    for i, farm in enumerate(farms):
        ax = axs[i // ncol][i % ncol]
        for s in series:
            ax.plot(leads, nbias[s][i], marker="o", ms=3, lw=1.2,
                    color=colors[s[0]], ls=STYLE[s[1]], label=lbl(s))
        ax.axhline(0.0, color="k", lw=1.0)
        ax.set_title(f"{farm}\n{cap_np[i]:.0f} MW", fontsize=8)
        ax.grid(ls="--", alpha=0.4)
    for j in range(F, nrow * ncol):
        axs[j // ncol][j % ncol].axis("off")
    axs[0][0].legend(fontsize=6)
    fig.supxlabel("lead time [h]"); fig.supylabel("bias [% of farm capacity]")
    fig.suptitle(f"Per-farm power BIAS (forecast − observed) — {args.region} ({scope})")
    fig.tight_layout()
    p = args.out / f"bias_per_farm_{args.region}{sfx}.png"
    fig.savefig(p, dpi=140); plt.close(fig); print(f"saved {p}")

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
        axR.plot(leads, vr, marker="o", ms=4, lw=1.8,
                 color=colors[s[0]], ls=STYLE[s[1]], label=lbl(s))
    axR.axhline(1.0, color="grey", ls="--", lw=1.2)
    axR.set(xlabel="lead time [h]", ylabel=r"$\sigma_{pred}\,/\,\sigma_{obs}$",
            title="Variance ratio of the regional total\nbelow 1 = under-dispersive")
    axR.set_xticks(leads); axR.grid(ls="--", alpha=0.5); axR.legend(fontsize=7)
    fig.tight_layout()
    p = args.out / f"decomposition_{args.region}{sfx}.png"
    fig.savefig(p, dpi=140); plt.close(fig); print(f"saved {p}")

    print(f"\nMSE DECOMPOSITION — {args.region} total ({scope}), pooled over leads")
    print(f"{'series':36s} {'RMSE':>8s} {'bias²':>9s} {'ampl':>9s} {'phase':>9s} "
          f"{'σp/σo':>7s} {'r':>6s}")
    print("-" * 90)
    for s in series:
        d = dec_all[s]
        if not np.isfinite(d["mse"]):
            continue
        print(f"{lbl(s):36s} {np.sqrt(d['mse']):8.1f} {d['bias2']:9.0f} {d['amp']:9.0f} "
              f"{d['phase']:9.0f} {d['var_ratio']:7.3f} {d['r']:6.3f}")

    # ---- wind-regime split, with the one-sidedness diagnostic ----
    with np.errstate(invalid="ignore", divide="ignore"):
        nmae_tr = {s_: 100.0 * (sae_tr[s_] / n_tr[s_]) / total_cap for s_ in series}

    counts = n_tr[series[0]].sum(axis=0)
    print(f"\nTOTAL {args.region} POWER BY WIND REGIME ({scope}) - MAE as % of {total_cap:.0f} MW")
    for i, s_ in enumerate(series, 1):
        print(f"  [{i:2d}] {lbl(s_)}")
    hdr = (f"{'regime':>10s} {'n':>7s}"
           + "".join(f"{f'[{i}]':>9s}" for i in range(1, len(series) + 1)))
    print(hdr); print("-" * len(hdr))
    for ri, rl in enumerate(reg_labels):
        row = f"{rl:>10s} {int(counts[ri]):7d}"
        for s_ in series:
            N = n_tr[s_][:, ri].sum()
            row += f"{(100.0 * sae_tr[s_][:, ri].sum() / N / total_cap) if N else np.nan:8.2f}%"
        print(row)

    print("\nBIAS BY WIND REGIME [MW;  + = over-predict]")
    print(hdr); print("-" * len(hdr))
    for ri, rl in enumerate(reg_labels):
        row = f"{rl:>10s} {int(counts[ri]):7d}"
        for s_ in series:
            N = n_tr[s_][:, ri].sum()
            row += f"{(sbe_tr[s_][:, ri].sum() / N) if N else np.nan:8.0f} "
        print(row)

    # |bias| / MAE == 1 means EVERY case errs the same way: not inaccuracy, but a forecast that
    # cannot reach the state at all. Well below 1 the errors straddle the truth as they should.
    print("\nONE-SIDEDNESS  |bias| / MAE   (1.00 = every case errs the same way)")
    print(hdr); print("-" * len(hdr))
    for ri, rl in enumerate(reg_labels):
        row = f"{rl:>10s} {int(counts[ri]):7d}"
        for s_ in series:
            N = n_tr[s_][:, ri].sum()
            m = sae_tr[s_][:, ri].sum() / N if N else np.nan
            b = abs(sbe_tr[s_][:, ri].sum() / N) if N else np.nan
            row += f"{(b / m if m else np.nan):9.3f}"
        print(row)
    print("  a value at 1.000 is a REPRESENTATIONAL limit, not a skill one: the forecast never")
    print("  once lands on the other side of the observation.")

    ncw_r = int(np.ceil(R / 2))
    fig, axs = plt.subplots(2, ncw_r, figsize=(5.2 * ncw_r, 8.4), squeeze=False, sharex=True)
    for ri in range(R):
        axk = axs[ri // ncw_r][ri % ncw_r]
        for s_ in series:
            axk.plot(leads, nmae_tr[s_][:, ri], marker="o", ms=4, lw=1.7,
                     color=colors[s_[0]], ls=STYLE[s_[1]], label=lbl(s_))
        axk.set_title(f"{reg_labels[ri]} m/s   ({int(counts[ri])} valid times)", fontsize=10)
        axk.set_xlabel("lead time [h]"); axk.set_ylabel(f"MAE [% of {total_cap:.0f} MW]")
        axk.set_xticks(leads); axk.grid(ls="--", alpha=0.5)
    for j in range(R, 2 * ncw_r):
        axs[j // ncw_r][j % ncw_r].axis("off")
    axs[0][0].legend(fontsize=6, ncol=2)
    fig.suptitle(f"{args.region} total power MAE by wind regime ({scope})\n"
                 f"regime set by OBSERVED power through the fleet power curve", fontsize=11)
    fig.tight_layout()
    pth = args.out / f"mae_regimes_{args.region}{sfx}.png"
    fig.savefig(pth, dpi=140); plt.close(fig); print(f"\nsaved {pth}")

    reg_rows = []
    for s_ in series:
        for ri, rl in enumerate(reg_labels):
            for li, lh in enumerate(leads):
                N = n_tr[s_][li, ri]
                reg_rows.append(dict(run=s_[0], method=s_[1], regime=rl, lead_hours=lh,
                                     scope="TOTAL",
                                     mae_mw=sae_tr[s_][li, ri] / N if N else np.nan,
                                     nmae_pct=nmae_tr[s_][li, ri],
                                     rmse_mw=np.sqrt(sse_tr[s_][li, ri] / N) if N else np.nan,
                                     bias_mw=sbe_tr[s_][li, ri] / N if N else np.nan, n=int(N)))
    pth = args.out / f"scores_regimes_{args.region}{sfx}.csv"
    pd.DataFrame(reg_rows).to_csv(pth, index=False)
    print(f"saved {pth}")

    # ---- CSVs ----
    rows = []
    for s in series:
        for li, lh in enumerate(leads):
            rows.append(dict(run=s[0], method=s[1], lead_hours=int(lh), scope="TOTAL",
                             mae_mw=mae_t[s][li], nmae_pct=nmae_t[s][li], rmse_mw=rmse_t[s][li],
                             bias_mw=bias_t[s][li], n=int(n_t[s][li])))
            for i, farm in enumerate(farms):
                rows.append(dict(run=s[0], method=s[1], lead_hours=int(lh), scope=farm,
                                 mae_mw=mae[s][i, li], nmae_pct=nmae[s][i, li],
                                 rmse_mw=rmse[s][i, li], bias_mw=bias[s][i, li], n=int(n[s][i, li])))
    scores_csv = args.out / f"scores_{args.region}{sfx}.csv"
    pd.DataFrame(rows).to_csv(scores_csv, index=False)
    print(f"saved {scores_csv}")
    if summary_rows:
        summary_csv = args.out / f"summary_{args.region}{sfx}.csv"
        pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
        print(f"saved {summary_csv}")


if __name__ == "__main__":
    main()
