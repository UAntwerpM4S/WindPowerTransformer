"""Build the dataset and train the wind-power post-processor. One run, one model.

  1. extract_cells.py  : the 6 wind variables at the Belgian cells, per init and lead, from the
                         MLWP run's forecast_*.nc          -> cells_<REGION>_<run>.nc
  2. build_targets.py  : attach observed power + the chronological split
                                                           -> dataset_<REGION>_<run>.nc
  3. Train.py          : the transformer, 6 wind + static  -> checkpoints/<tag>/

Both steps skip if their output already exists, so re-running is cheap. Then:

    python3 infer_cf.py   &&   python3 score_cf.py

SPLITS come from build_targets.py (TRAIN/VAL/TEST_MONTHS), NOT from --split-frac, because they
have to clear the MLWP model's own training cutoff: the LAM trained to 2024-01-31, so every
forecast the post-processor sees must be initialised on or after 2024-02-01 or it is learning
from forecasts the LAM had already fitted. Check that inference actually covers the configured
TRAIN_MONTHS before running this.

--datasets-only stops after step 2 (useful for inspecting the dataset before committing GPU time).
"""
import argparse
import glob
import os

_ap = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("--datasets-only", action="store_true",
                 help="build dataset_<REGION>_<run>.nc and stop; train nothing")
ARGS = _ap.parse_args()

CELLS_DIR = "/mnt/weatherloss/WindPowerTransformer/data/cells"
REGION    = "BE"

# The MLWP run being post-processed. RegularWeather is weather-only: no power head, so nothing
# here depends on in-model power. "none" = extract wind only, no cf_lam channel.
RUN     = "RegularWeather"
RUN_DIR = "/mnt/weatherloss/WindPower/inference/WindAI/RegularWeather"

TRAIN_ARGS = ("--model_dim 64 --n_heads 4 --num_layers 2 --mlp_mult 4 --dropout 0.1 "
              "--batch_size 64 --epochs 40 --lr 3e-4 --weight_decay 1e-4 --patience 6 "
              "--no-cf-lam")


def run(cmd):
    print(f"\n🚀 {cmd}\n")
    if os.system(cmd) != 0:
        raise SystemExit(f"step failed: {cmd}")


dataset = f"{CELLS_DIR}/dataset_{REGION}_{RUN}.nc"
if os.path.exists(dataset):
    print(f"✓ dataset exists, skipping build: {dataset}")
else:
    run(f"python3 data/extract_cells.py --runs {RUN}={RUN_DIR} --region {REGION} "
        f"--power-var none --out {CELLS_DIR}")
    run(f"python3 data/build_targets.py --cells {CELLS_DIR}/cells_{REGION}_{RUN}.nc "
        f"--out {CELLS_DIR}")

if ARGS.datasets_only:
    print(f"\n✅ dataset built: {dataset}")
    raise SystemExit(0)

if glob.glob(f"checkpoints/{RUN}/*.pt"):
    print(f"✓ checkpoint exists, skipping train: {RUN}")
else:
    run(f"python3 Train.py --run_tag {RUN} --dataset_nc {dataset} {TRAIN_ARGS}")

print("\n✅ built + trained. Next:  python3 infer_cf.py  &&  python3 score_cf.py")
