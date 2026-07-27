#!/usr/bin/env python3
import os
import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd

# ========= CONFIG =========
MODEL_NAMES = [
    "RegularWeather",
    "VanillaPowerGT",
    "VanillaPowerTF",
    "WindHeavyTinyPower",
    "WindHeavyVanillaPower",
    "WindWeather",
]

DIM = 128
HEADS = 4
LAYERS = 4

TOTAL_CAPACITY_MW = 2177.2

TEST_FCS_DIR   = "TEST_FCS"
POWERCURVE_DIR = "PC_FCS"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
# ==========================


def _sum_over_windparks(files):
    """Fleet total = straight sum over per-farm files (targets are per-farm)."""
    f_sum = t_sum = None
    for fp in sorted(files):
        ds = xr.open_dataset(fp)
        f = ds["forecast"].astype(float)
        t = ds["truth"].astype(float)
        if f_sum is None:
            f_sum, t_sum = f, t
        else:
            f, f_sum = xr.align(f, f_sum, join="inner")
            t, t_sum = xr.align(t, t_sum, join="inner")
            f_sum = f_sum + f
            t_sum = t_sum + t
        ds.close()
    return xr.align(f_sum, t_sum, join="inner")


def _metrics(f, t):
    """Per-lead MAE and RMSE (% of capacity), aggregated over inits ('date')."""
    diff = f - t
    mae_mw = np.abs(diff).mean(dim="date", skipna=True).values
    rmse_mw = np.sqrt((diff ** 2).mean(dim="date", skipna=True).values)
    return {
        "mae_pct": mae_mw / TOTAL_CAPACITY_MW * 100.0,
        "rmse_pct": rmse_mw / TOTAL_CAPACITY_MW * 100.0,
    }


def main():
    plt.figure(figsize=(12, 5))

    for model in MODEL_NAMES:
        tf_files = sorted(glob.glob(
            f"{TEST_FCS_DIR}/{model}_dim{DIM}_heads{HEADS}_layers{LAYERS}_*.nc"
        ))
        pc_files = sorted(glob.glob(
            f"{POWERCURVE_DIR}/{model}/{model}_PowerCurve_*.nc"
        ))

        if not tf_files:
            print(f"⚠️ No TEST_FCS files for {model}, skipping.")
            continue

        # transformer: per-farm forecast + per-farm obs truth, summed over farms
        tf_f, tf_t = _sum_over_windparks(tf_files)

        lt_hours = tf_f["lead_time"].values * 3
        m_tf = _metrics(tf_f, tf_t)
        plt.plot(lt_hours, m_tf["mae_pct"], linewidth=2, label=f"{model} (TEST_FCS)")

        cols = {
            "lead_time_hours": lt_hours,
            "mae_test_fcs_pct": m_tf["mae_pct"],
            "rmse_test_fcs_pct": m_tf["rmse_pct"],
        }

        # PowerCurve baseline is optional: per-farm power curve, summed over farms
        if pc_files:
            pc_f, pc_t = _sum_over_windparks(pc_files)
            m_pc = _metrics(pc_f, pc_t)
            plt.plot(lt_hours, m_pc["mae_pct"], linewidth=2, linestyle="--", label=f"{model} (PowerCurve)")
            cols["mae_powercurve_pct"] = m_pc["mae_pct"]
            cols["rmse_powercurve_pct"] = m_pc["rmse_pct"]
        else:
            print(f"ℹ️ No PowerCurve baseline for {model}; plotting transformer only.")

        # ---- CSV ----
        df = pd.DataFrame(cols)
        csv_path = os.path.join(
            RESULTS_DIR,
            f"MAE_{model}_dim{DIM}_heads{HEADS}_layers{LAYERS}.csv",
        )
        df.to_csv(csv_path, index=False)

    plt.title("Aggregated Offshore MAE vs Lead Time", fontsize=18)
    plt.xlabel("Lead Time [h]", fontsize=16)
    plt.ylabel("MAE [% of capacity]", fontsize=16)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    out_png = os.path.join(
        RESULTS_DIR,
        f"Compare_MAE_PCT_6lines_dim{DIM}_heads{HEADS}_layers{LAYERS}.png",
    )
    plt.savefig(out_png)
    plt.close()

    print(f"✅ Saved plot: {out_png}")
    print(f"✅ Saved CSVs in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
