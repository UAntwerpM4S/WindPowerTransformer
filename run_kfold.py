"""K-fold CV: train K models per experiment so every init gets an out-of-fold prediction.

This removes the summer-only test caveat of run_forecast_tier.py: instead of one 60/20/20 split
(test = last 20% = summer), the year is cut into K chronological folds and each is held out once,
so the reported metric covers the WHOLE period.

Requires the datasets from run_forecast_tier.py (dataset_BE_<RUN>.nc). For each experiment it trains
K models tagged <tag>_f0 .. <tag>_f{K-1} (split derived from init timestamps by Train --n-folds /
--test-fold), skipping any whose checkpoint already exists.

Then:  python3 infer_cf.py --kfold K   (assembles OOF cf_BE_<tag>.nc)
        python3 score_cf.py --split all (scores every init, year-round)

Cost: len(EXPERIMENTS) * K trainings (default 7 * 5 = 35). Lower K if that's too heavy.
"""
import glob
import os

CELLS_DIR = "/mnt/weatherloss/WindPowerTransformer/data/cells"
REGION    = "BE"
K         = 5

# same experiments as infer_cf.py / run_forecast_tier.py: (tag, source run, use_cf_lam)
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

for tag, src, use_cf_lam in EXPERIMENTS:
    dataset = f"{CELLS_DIR}/dataset_{REGION}_{src}.nc"
    if not os.path.exists(dataset):
        print(f"⚠️  {dataset} missing -- run run_forecast_tier.py first; skipping {tag}")
        continue
    flag = "" if use_cf_lam else "--no-cf-lam"
    for f in range(K):
        fold_tag = f"{tag}_f{f}"
        if glob.glob(f"checkpoints/{fold_tag}/*.pt"):
            print(f"✓ checkpoint exists, skipping: {fold_tag}")
            continue
        cmd = (f"python3 Train.py --run_tag {fold_tag} --dataset_nc {dataset} {flag} "
               f"--n-folds {K} --test-fold {f} {TRAIN_ARGS}")
        print(f"\n🚀 {cmd}\n")
        if os.system(cmd) != 0:
            raise SystemExit(f"step failed: {cmd}")

print(f"\n✅ trained {len(EXPERIMENTS)}x{K} fold models. Next:")
print(f"   python3 infer_cf.py --kfold {K}   &&   python3 score_cf.py --split all")
