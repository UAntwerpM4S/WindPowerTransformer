#!/usr/bin/env python3
"""
Evaluate CERRA-trained transformers on the CERRA test split (2024-08..2025-07).

For each windpark it loads the matching checkpoint, runs inference over the test
windows, and saves a NetCDF with `forecast` and `truth` indexed by (init date,
lead_time hours). This is the in-distribution CERRA test; the separate forecast
test year (the six weather models) is evaluated by feeding their forecasts
through the same checkpoints with the CERRA train stats.
"""
import os
import re
import glob

import numpy as np
import torch
import xarray as xr
from tqdm import tqdm

from Transformer import TemporalTransformer
from Loader import loader_prepare

# ======== CONFIG ========
RUN_TAG = "CERRA"
INPUT_DIM = 6
MODEL_DIM = 128
N_HEADS = 4
NUM_LAYERS = 4
MLP_MULT = 4
LEAD_HOURS = 36
FREQ_HOURS = 3
BATCH_SIZE = 8

ZARR_PATH = "/mnt/weatherloss/WindPower/data/EGU26/Anemoidatasets/New_Cerra_A_large.zarr"
METADATA_PATH = "/mnt/weatherloss/WindPower/data/NorthSea/Power/windfarm_metadata.csv"

TRAIN_RANGE = ("2020-01-01", "2024-01-31 23:00")
VAL_RANGE = ("2024-02-01", "2024-07-31 23:00")
TEST_RANGE = ("2024-08-01", "2025-07-31 23:00")

OUT_DIR = "TEST_CERRA"
# ========================

windparks = [
    "Belwind Phase 1",
    "Thorntonbank - C-Power - Area NE",
    "Thorntonbank - C-Power - Area SW",
    "Mermaid Offshore WP",
    "Nobelwind Offshore Windpark",
    "Norther Offshore WP",
    "Northwester 2",
    "Northwind",
    "Rentel Offshore WP",
    "Seastar Offshore WP",
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_")


def find_ckpt(windpark: str) -> str | None:
    pat = os.path.join(
        "checkpoints", RUN_TAG,
        f"{RUN_TAG}_dim{MODEL_DIM}_park{windpark}_in{INPUT_DIM}_"
        f"heads{N_HEADS}_layers{NUM_LAYERS}_*.pt",
    )
    matches = sorted(glob.glob(pat))
    return matches[0] if matches else None


@torch.no_grad()
def infer(model, loader):
    model.eval()
    preds, truths = [], []
    for X, y, mask in tqdm(loader, desc="Inference", leave=False):
        out = model(X.to(device)).cpu().numpy()
        y = y.numpy().copy()
        y[~mask.numpy()] = np.nan          # keep gaps as NaN for honest scoring
        preds.append(out)
        truths.append(y)
    if not preds:
        return None, None
    return np.concatenate(preds, 0), np.concatenate(truths, 0)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    for windpark in windparks:
        ckpt = find_ckpt(windpark)
        if ckpt is None:
            print(f"⚠️ No checkpoint for {windpark}, skipping.")
            continue
        print(f"\n🔍 {windpark} | {os.path.basename(ckpt)}")

        _, _, test_loader, test_set = loader_prepare(
            windpark=windpark,
            zarr_path=ZARR_PATH,
            metadata_path=METADATA_PATH,
            run_tag=RUN_TAG,
            train_range=TRAIN_RANGE,
            val_range=VAL_RANGE,
            test_range=TEST_RANGE,
            batch_size=BATCH_SIZE,
            lead_hours=LEAD_HOURS,
            freq_hours=FREQ_HOURS,
            stride=1,
        )

        model = TemporalTransformer(
            input_dim=INPUT_DIM, model_dim=MODEL_DIM,
            n_heads=N_HEADS, num_layers=NUM_LAYERS, mlp_mult=MLP_MULT,
        ).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))

        preds, truths = infer(model, test_loader)
        if preds is None:
            print(f"⚠️ No test windows for {windpark}.")
            continue

        n = preds.shape[0]
        init_dates = test_set.init_dates[:n].tz_localize(None).values.astype("datetime64[ns]")
        lead_time = np.arange(0, LEAD_HOURS + 1, FREQ_HOURS, dtype=np.int32)  # hours: 0,3,...,36

        ds = xr.Dataset(
            {
                "forecast": (("date", "lead_time"), preds.astype(np.float32)),
                "truth": (("date", "lead_time"), truths.astype(np.float32)),
            },
            coords={"date": init_dates, "lead_time": lead_time},
            attrs={"windpark": windpark, "run_tag": RUN_TAG, "lead_hours": LEAD_HOURS},
        )
        out_nc = os.path.join(OUT_DIR, f"{RUN_TAG}_{safe_name(windpark)}.nc")
        ds.to_netcdf(out_nc)
        print(f"💾 {out_nc}  ({n} windows)")


if __name__ == "__main__":
    main()
