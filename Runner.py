"""CERRA tier, end to end: build the shared wind->power converter, apply it, score it.

WHAT THIS IS FOR.  The question the paper asks is "does training the LAM on power improve its
WIND for power conversion". To answer it you need ONE converter that is identical for every run,
so any difference it reports is a difference in the wind and not in the converter. The specs power
curve is such a converter but it is uncalibrated: it ignores wake losses and over-predicts the
Belgian fleet by ~180-200 MW, which lets a run whose wind happens to sit low look better for the
wrong reason. This chain builds the calibrated replacement.

    fitted on   CERRA ANALYSIS wind  ->  observed cf,  2020-01-01 .. 2024-01-31
    applied to  every LAM run's FORECAST wind,          2024-08-01 .. 2025-07-31

The fit window is the LAM's own training period, so (a) the converter has never seen an
observation from the scored year, and (b) it had access to exactly the history the LAMs had --
which is what makes the comparison fair. It never sees a forecast at all, so it cannot prefer
one run over another.

WHY NO K-FOLD.  The forecast-tier post-processor needs k-fold because LAM forecasts only exist
from 2024-02, so the only data available to train it is the same year it must be scored on. CERRA
runs continuously from 2020, so a plain chronological separation does the job here and one
checkpoint serves every run.

CAVEAT to carry into the paper: the converter is fitted on analysis wind and applied to forecast
wind, which is smoother and biased. That mis-application is identical for every run -- the same
argument that justifies a shared specs curve -- but state it rather than let a referee find it.

Steps
  1. extract_cells_cerra.py   CERRA zarr        -> cells_BE_CERRA.nc     (6 wind at the 15 cells)
  2. build_targets.py         + cf_obs + split  -> dataset_BE_CERRA.nc
  3. Train.py                 dataset_BE_CERRA  -> checkpoints/CERRAcell + artifacts/CERRAcell
  4. infer_cf.py --only CERRAcell                -> cf_BE_CERRAcell@<run>.nc, one per LAM run
  5. score_cf.py --methods direct curve cerra    -> the comparison table + figures

Steps 4-5 need dataset_BE_<run>.nc for each LAM run. Build them first with

    python3 run_forecast_tier.py --datasets-only

which runs extract_cells.py + build_targets.py per run and trains nothing. Then run this file;
--from N resumes at a step, --only N runs one, --dry-run just prints.
"""
import argparse
import os

CELLS_DIR = "/mnt/weatherloss/WindPowerTransformer/data/cells"
RUN_TAG   = "CERRAcell"
REGION    = "BE"

# fit window = the LAM's training period; val/test mirror the LAM's own boundaries so the split
# labels in dataset_BE_CERRA.nc mean the same thing they do in the forecast-tier datasets.
TRAIN_MONTHS = "2020 1 2024 1"
VAL_MONTHS   = "2024 2 2024 7"
TEST_MONTHS  = "2024 8 2025 7"

STEPS = [
    # 1) the 6 wind vars at the 15 BE cells, from CERRA truth (no cf_lam: there is no forecast here)
    f"python3 data/extract_cells_cerra.py --start 2020-01-01 --end '2025-07-31 21:00' "
    f"--out {CELLS_DIR}",

    # 2) attach the cf_obs target + the leakage-safe split
    f"python3 data/build_targets.py --cells {CELLS_DIR}/cells_{REGION}_CERRA.nc --out {CELLS_DIR} "
    f"--train-months {TRAIN_MONTHS} --val-months {VAL_MONTHS} --test-months {TEST_MONTHS}",

    # 3) train the shared cell-level converter. input_dim is inferred (9 = 6 wind + 3 static).
    #    Small + regularised on purpose: the wind->CF map is simple and a dim128/4-layer model
    #    overfit within 2 epochs.
    f"python3 Train.py --run_tag {RUN_TAG} --model_dim 64 --n_heads 4 --num_layers 2 "
    f"--mlp_mult 4 --dropout 0.1 --batch_size 64 --epochs 40 --lr 3e-4 --weight_decay 1e-4 "
    f"--patience 6 --dataset_nc {CELLS_DIR}/dataset_{REGION}_CERRA.nc",

    # 4) apply that ONE checkpoint to every LAM run's forecast wind (use_cf_lam=False in
    #    infer_cf.EXPERIMENTS; the loader would otherwise add a 7th channel the model never saw)
    f"python3 infer_cf.py --only {RUN_TAG} --region {REGION}",

    # 5) score it against direct + the specs power curve.
    #    --split all, NOT the datasets' own 'test' split: those splits exist for the forecast-tier
    #    models, which trained on part of this same year. The CERRA converter trained only on
    #    2020..2024-01, so EVERY forecast init here is out-of-sample for it and restricting to the
    #    last 20% would throw away three quarters of the sample (and land in summer, which biases
    #    the comparison -- that is exactly what inflated the earlier forecast-tier headline).
    f"python3 score_cf.py --region {REGION} --methods direct curve cerra --split all",
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="start", type=int, default=1,
                    help=f"resume at this step (1..{len(STEPS)})")
    ap.add_argument("--only", type=int, default=None, help="run just this one step")
    ap.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    a = ap.parse_args()

    todo = [a.only] if a.only else list(range(a.start, len(STEPS) + 1))
    for i in todo:
        if not 1 <= i <= len(STEPS):
            raise SystemExit(f"step {i} out of range 1..{len(STEPS)}")
        cmd = STEPS[i - 1]
        print(f"\n[{i}/{len(STEPS)}] {cmd}\n")
        if a.dry_run:
            continue
        if os.system(cmd) != 0:
            raise SystemExit(f"step {i} failed: {cmd}")
    print("\ndone" if not a.dry_run else "\ndry run only")
