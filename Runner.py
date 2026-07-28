"""
CERRA-truth baseline: the full 3-step chain, in order.

  1. extract_cells_cerra.py  CERRA zarr  -> cells_BE_CERRA.nc     (6 wind at the 15 cells)
  2. build_targets.py        + cf_obs + split -> dataset_BE_CERRA.nc
  3. Train.py                dataset_BE_CERRA.nc -> a shared cell-level model

Splits are the LAM's (train 2020-2024, val 2024-02..07, test 2024-08..2025-07) so the CERRA
curve trains on the full record; it is model-agnostic, so training over the LAM's own period is
fine. The real evaluation applies this model to forecast wind later (scoring step, not here).
"""
import os

CELLS_DIR = "/mnt/weatherloss/WindPowerTransformer/data/cells"
run_tag = "CERRAcell"

steps = [
    # 1) extract the 6 wind vars at the 15 BE cells from CERRA truth
    f"python3 data/extract_cells_cerra.py --start 2020-01-01 --end '2025-07-31 21:00' "
    f"--out {CELLS_DIR}",

    # 2) attach the cf_obs target + leakage-safe split (LAM boundaries)
    f"python3 data/build_targets.py --cells {CELLS_DIR}/cells_BE_CERRA.nc --out {CELLS_DIR} "
    f"--train-months 2020 1 2024 1 --val-months 2024 2 2024 7 --test-months 2024 8 2025 7",

    # 3) train the shared cell-level model (input_dim is inferred from the dataset)
    f"python3 Train.py --run_tag {run_tag} --model_dim 128 --n_heads 4 --num_layers 4 "
    f"--mlp_mult 4 --batch_size 64 --epochs 40 --lr 0.001 --patience 6 "
    f"--dataset_nc {CELLS_DIR}/dataset_BE_CERRA.nc",
]

for cmd in steps:
    print(f"\n🚀 {cmd}\n")
    if os.system(cmd) != 0:
        raise SystemExit(f"step failed: {cmd}")
