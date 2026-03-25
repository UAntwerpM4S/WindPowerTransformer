#!/usr/bin/env python3
import os
import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd

# ========= CONFIG =========
MODEL_NAMES = ["GraphTransformer", "GNN", "Transformer"]

DIM = 128
HEADS = 4
LAYERS = 4

TOTAL_CAPACITY_MW = 2177.2

TEST_FCS_DIR   = "TEST_FCS2"
POWERCURVE_DIR = "PC_FCS"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
# ==========================


def _sum_over_windparks(files):
    f_sum = t_sum = None
    for fp in files:
        ds = xr.open_dataset(fp)
        f = ds["forecast"].astype(float)
        t = ds["truth"].astype(float)
        if f_sum is None:
            f_sum, t_sum = f, t
        else:
            f, f_sum = xr.align(f, f_sum, join="inner")
            t, t_sum = xr.align(t, t_sum, join="inner")
            f_sum += f
            t_sum += t
        ds.close()
    return xr.align(f_sum, t_sum, join="inner")


def _rmse_pct(f, t):
    err = f - t
    rmse = np.sqrt((err ** 2).mean(dim="date", skipna=True)).values
    return (rmse / TOTAL_CAPACITY_MW) * 100.0


def main():
    plt.figure(figsize=(12, 5))

    for model in MODEL_NAMES:
        tf_files = sorted(glob.glob(
            f"{TEST_FCS_DIR}/{model}_dim{DIM}_heads{HEADS}_layers{LAYERS}_*.nc"
        ))
        pc_files = sorted(glob.glob(
            f"{POWERCURVE_DIR}/{model}/{model}_PowerCurve_*.nc"
        ))

        tf_f, tf_t = _sum_over_windparks(tf_files)
        pc_f, pc_t = _sum_over_windparks(pc_files)

        tf_f, tf_t, pc_f, pc_t = xr.align(tf_f, tf_t, pc_f, pc_t, join="inner")

        lt_steps = tf_f["lead_time"].values
        lt_hours = lt_steps * 3

        rmse_tf = _rmse_pct(tf_f, tf_t)
        rmse_pc = _rmse_pct(pc_f, pc_t)

        # ---- plot ----
        plt.plot(lt_hours, rmse_tf, linewidth=2, label=f"{model} (TEST_FCS)")
        plt.plot(lt_hours, rmse_pc, linewidth=2, linestyle="--", label=f"{model} (PowerCurve)")

        # ---- CSV ----
        df = pd.DataFrame({
            "lead_time_hours": lt_hours,
            "rmse_test_fcs_pct": rmse_tf,
            "rmse_powercurve_pct": rmse_pc,
        })
        csv_path = os.path.join(
            RESULTS_DIR,
            f"RMSE_{model}_dim{DIM}_heads{HEADS}_layers{LAYERS}.csv",
        )
        df.to_csv(csv_path, index=False)

    plt.title("Aggregated Offshore RMSE vs Lead Time", fontsize=18)
    plt.xlabel("Lead Time [h]", fontsize=16)
    plt.ylabel("RMSE [% of capacity]", fontsize=16)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    out_png = os.path.join(
        RESULTS_DIR,
        f"Compare_RMSE_PCT_6lines_dim{DIM}_heads{HEADS}_layers{LAYERS}.png",
    )
    plt.savefig(out_png)
    plt.close()

    print(f"✅ Saved plot: {out_png}")
    print(f"✅ Saved CSVs in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
