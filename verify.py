#!/usr/bin/env python3
import os
import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# ========= CONFIG =========
MODEL_NAME = "GraphTransformer"
DIM = 128
HEADS = 4
LAYERS = 4

TOTAL_CAPACITY_MW = 2262.0  # used to convert MW -> % of capacity

TRANSFORMER_DIR = "forecast_netcdfs"
POWERCURVE_DIR  = "forecast_netcdfs_powercurve_obs"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Transformer file pattern (matches your existing naming)
TRANSFORMER_PATTERN = os.path.join(
    TRANSFORMER_DIR,
    f"{MODEL_NAME}_dim{DIM}_heads{HEADS}_layers{LAYERS}_*.nc"
)

# Power-curve file pattern (matches OUT_PREFIX = f"{MODEL_NAME}_PowerCurve_")
POWERCURVE_PATTERN = os.path.join(
    POWERCURVE_DIR,
    f"{MODEL_NAME}_PowerCurve_*.nc"
)
# ==========================


def _sum_over_windparks(nc_files):
    """Return summed forecast/truth DataArrays (date, lead_time) across all provided files."""
    if not nc_files:
        return None, None, None

    summed_f = None
    summed_t = None
    lead_time = None

    for fp in nc_files:
        ds = xr.open_dataset(fp)

        if "forecast" not in ds or "truth" not in ds:
            print(f"⚠️ Skipping (missing forecast/truth): {fp}")
            ds.close()
            continue

        f = ds["forecast"].astype(np.float64)
        t = ds["truth"].astype(np.float64)

        # First file initializes
        if summed_f is None:
            summed_f = f
            summed_t = t
            lead_time = ds["lead_time"].values if "lead_time" in ds.coords else np.arange(f.shape[1])
        else:
            # Align by common coords (date, lead_time) before summing
            f, summed_f = xr.align(f, summed_f, join="inner")
            t, summed_t = xr.align(t, summed_t, join="inner")
            summed_f = summed_f + f
            summed_t = summed_t + t

        ds.close()

    if summed_f is None:
        return None, None, None

    # Final alignment safety
    summed_f, summed_t = xr.align(summed_f, summed_t, join="inner")
    lead_time = summed_f["lead_time"].values if "lead_time" in summed_f.coords else lead_time
    return summed_f, summed_t, lead_time


def _metrics_pct(summed_f, summed_t):
    """Compute MAE% and RMSE% vs lead_time."""
    err = (summed_f - summed_t)

    # If any NaNs exist, these will propagate; use skipna=True if needed.
    mae_mw  = np.abs(err).mean(dim="date", skipna=True).values
    rmse_mw = np.sqrt((err ** 2).mean(dim="date", skipna=True)).values

    mae_pct  = (mae_mw  / TOTAL_CAPACITY_MW) * 100.0
    rmse_pct = (rmse_mw / TOTAL_CAPACITY_MW) * 100.0
    return mae_pct, rmse_pct


def _save_two_line_plot(x, y1, y2, label1, label2, title, ylabel, out_path):
    plt.figure(figsize=(12, 5))
    plt.plot(x, y1, marker="o", markersize=4, linewidth=2, label=label1)
    plt.plot(x, y2, marker="o", markersize=4, linewidth=2, label=label2)
    plt.title(title, fontsize=18)
    plt.xlabel("Lead Time [h]", fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    transformer_files = sorted(glob.glob(TRANSFORMER_PATTERN))
    powercurve_files  = sorted(glob.glob(POWERCURVE_PATTERN))

    if not transformer_files:
        raise FileNotFoundError(f"No transformer NetCDFs found:\n  {TRANSFORMER_PATTERN}")
    if not powercurve_files:
        raise FileNotFoundError(f"No power-curve NetCDFs found:\n  {POWERCURVE_PATTERN}")

    print(f"Found {len(transformer_files)} transformer files")
    print(f"Found {len(powercurve_files)} power-curve files")

    # --- Aggregate totals ---
    tf_f, tf_t, tf_lt = _sum_over_windparks(transformer_files)
    pc_f, pc_t, pc_lt = _sum_over_windparks(powercurve_files)

    if tf_f is None:
        raise RuntimeError("No valid transformer NetCDFs were loaded.")
    if pc_f is None:
        raise RuntimeError("No valid power-curve NetCDFs were loaded.")

    # --- Ensure both approaches share the same lead_times and dates for fair comparison ---
    tf_f, tf_t, pc_f, pc_t = xr.align(tf_f, tf_t, pc_f, pc_t, join="inner")

    # lead_time in your plotting code is step index; convert to hours (3h increments)
    lt_steps = tf_f["lead_time"].values
    lt_hours = lt_steps * 3

    # --- Metrics (% of capacity) ---
    tf_mae_pct, tf_rmse_pct = _metrics_pct(tf_f, tf_t)
    pc_mae_pct, pc_rmse_pct = _metrics_pct(pc_f, pc_t)

    base = f"{MODEL_NAME}_dim{DIM}_heads{HEADS}_layers{LAYERS}"
    out_mae = os.path.join(RESULTS_DIR, f"Compare_MAE_PCT_{base}.png")
    out_rmse = os.path.join(RESULTS_DIR, f"Compare_RMSE_PCT_{base}.png")

    _save_two_line_plot(
        lt_hours,
        tf_mae_pct, pc_mae_pct,
        label1="Transformer", label2="Power curve",
        title="Aggregated Offshore MAE vs Lead Time",
        ylabel="MAE [% of capacity]",
        out_path=out_mae,
    )

    _save_two_line_plot(
        lt_hours,
        tf_rmse_pct, pc_rmse_pct,
        label1="Transformer", label2="Power curve",
        title="Aggregated Offshore RMSE vs Lead Time",
        ylabel="RMSE [% of capacity]",
        out_path=out_rmse,
    )

    print("✅ Done.")
    print(f"Saved: {out_mae}")
    print(f"Saved: {out_rmse}")
    print(f"Mean MAE%  - Transformer: {np.mean(tf_mae_pct):.3f}%, PowerCurve: {np.mean(pc_mae_pct):.3f}%")
    print(f"Mean RMSE% - Transformer: {np.mean(tf_rmse_pct):.3f}%, PowerCurve: {np.mean(pc_rmse_pct):.3f}%")


if __name__ == "__main__":
    main()
