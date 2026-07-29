"""Forecast-tier post-processor: build + train ONE model per forecasting run, in order.

For each run WITH a power channel (cf_lam):

  1. extract_cells.py   forecast dir -> cells_BE_<RUN>.nc   (6 wind + cf_lam + turbinecount)
  2. build_targets.py   + cf_obs + power_obs + chronological split -> dataset_BE_<RUN>.nc
  3. Train.py           dataset_BE_<RUN>.nc -> one post-processor (input_dim 10: 6 wind + cf_lam
                        + 3 static), stats + checkpoint under artifacts/<RUN>/ and checkpoints/<RUN>/

Split: chronological fractions 0.6 / 0.2 / 0.2 (train / early-stop / reported test) of whatever
period the inference dir covers -- no calendar assumptions. score_cf.py reports on 'test'.

Then, separately:  python infer_cf.py   &&   python score_cf.py
"""
import os

CELLS_DIR = "/mnt/weatherloss/WindPowerTransformer/data/cells"
REGION    = "BE"
SPLIT_FRAC = "0.6 0.2 0.2"                      # train / early-stop val / reported test

# run label -> LAM inference directory (only runs that have cf_lam)
RUNS = {
    "HighCapacityGT":     "/mnt/weatherloss/WindPower/inference/WPDistr/HighCapacityGT",
    "VanillaPowerGT":     "/mnt/weatherloss/WindPower/inference/WPDistr/VanillaPowerGT",
    "VeryHighCapacityGT": "/mnt/weatherloss/WindPower/inference/WPDistr/VeryHighCapacityGT",
}

for run, d in RUNS.items():
    cells   = f"{CELLS_DIR}/cells_{REGION}_{run}.nc"
    dataset = f"{CELLS_DIR}/dataset_{REGION}_{run}.nc"
    steps = [
        f"python3 data/extract_cells.py --runs {run}={d} --region {REGION} --out {CELLS_DIR}",
        f"python3 data/build_targets.py --cells {cells} --out {CELLS_DIR} --split-frac {SPLIT_FRAC}",
        f"python3 Train.py --run_tag {run} --dataset_nc {dataset} "
        f"--model_dim 64 --n_heads 4 --num_layers 2 --mlp_mult 4 --dropout 0.1 "
        f"--batch_size 64 --epochs 40 --lr 3e-4 --weight_decay 1e-4 --patience 6",
    ]
    for cmd in steps:
        print(f"\n🚀 {cmd}\n")
        if os.system(cmd) != 0:
            raise SystemExit(f"step failed: {cmd}")

print("\n✅ all runs built + trained. Next:  python3 infer_cf.py  &&  python3 score_cf.py")
