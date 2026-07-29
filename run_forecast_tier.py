"""Forecast-tier post-processors: build datasets + train one model per experiment.

Two tiers, so we can answer both questions:
  - wind + cf_lam  vs  wind-only   (same run, same inits): does cf_lam add anything?
  - wind-only on RegularWeather vs on the power runs: did power co-training even help the wind?

  1. build ONE dataset per source LAM run (extract_cells -> build_targets), skipped if it exists.
     RegularWeather is extracted wind-only (--power-var none: it has no power channel).
  2. train ONE model per experiment (tag, source run, use_cf_lam), skipped if a checkpoint exists.

Split: chronological 0.6 / 0.2 / 0.2 (train / early-stop / reported test). Then:
     python3 infer_cf.py   &&   python3 score_cf.py
"""
import glob
import os

CELLS_DIR  = "/mnt/weatherloss/WindPowerTransformer/data/cells"
REGION     = "BE"
SPLIT_FRAC = "0.6 0.2 0.2"

# source LAM run -> (inference dir, power var).  'none' = wind-only extraction.
SOURCE_RUNS = {
    "HighCapacityGT":     ("/mnt/weatherloss/WindPower/inference/WPDistr/HighCapacityGT",     "capacityfactor"),
    "VanillaPowerGT":     ("/mnt/weatherloss/WindPower/inference/WPDistr/VanillaPowerGT",     "capacityfactor"),
    "VeryHighCapacityGT": ("/mnt/weatherloss/WindPower/inference/WPDistr/VeryHighCapacityGT", "capacityfactor"),
    "RegularWeather":     ("/mnt/weatherloss/WindPower/inference/WindAI/RegularWeather",      "none"),
}

# (tag = run_tag/output name, source run whose dataset feeds it, use cf_lam?)
EXPERIMENTS = [
    ("HighCapacityGT",        "HighCapacityGT",     True),
    ("VanillaPowerGT",        "VanillaPowerGT",     True),
    ("VeryHighCapacityGT",    "VeryHighCapacityGT", True),
    ("HighCapacityGT_wo",     "HighCapacityGT",     False),
    ("VanillaPowerGT_wo",     "VanillaPowerGT",     False),
    ("VeryHighCapacityGT_wo", "VeryHighCapacityGT", False),
    ("RegularWeather",        "RegularWeather",     False),
]

TRAIN_ARGS = ("--model_dim 64 --n_heads 4 --num_layers 2 --mlp_mult 4 --dropout 0.1 "
              "--batch_size 64 --epochs 40 --lr 3e-4 --weight_decay 1e-4 --patience 6")


def run(cmd):
    print(f"\n🚀 {cmd}\n")
    if os.system(cmd) != 0:
        raise SystemExit(f"step failed: {cmd}")


# 1) datasets ---------------------------------------------------------------
for src, (d, pv) in SOURCE_RUNS.items():
    dataset = f"{CELLS_DIR}/dataset_{REGION}_{src}.nc"
    if os.path.exists(dataset):
        print(f"✓ dataset exists, skipping build: {dataset}")
        continue
    cells = f"{CELLS_DIR}/cells_{REGION}_{src}.nc"
    run(f"python3 data/extract_cells.py --runs {src}={d} --region {REGION} "
        f"--power-var {pv} --out {CELLS_DIR}")
    run(f"python3 data/build_targets.py --cells {cells} --out {CELLS_DIR} --split-frac {SPLIT_FRAC}")

# 2) one model per experiment ----------------------------------------------
for tag, src, use_cf_lam in EXPERIMENTS:
    if glob.glob(f"checkpoints/{tag}/*.pt"):
        print(f"✓ checkpoint exists, skipping train: {tag}")
        continue
    dataset = f"{CELLS_DIR}/dataset_{REGION}_{src}.nc"
    flag = "" if use_cf_lam else "--no-cf-lam"
    run(f"python3 Train.py --run_tag {tag} --dataset_nc {dataset} {flag} {TRAIN_ARGS}")

print("\n✅ built + trained. Next:  python3 infer_cf.py  &&  python3 score_cf.py")
