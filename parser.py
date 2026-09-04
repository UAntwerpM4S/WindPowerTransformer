import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Train the cell-level CERRA-truth PowerTransformer")

    # input_dim is NOT a flag: Train.py infers it from the dataset (6 wind [+cf_lam] + 3 static)
    # and names the checkpoint `...in<input_dim>...` so inference can only load a matching model.

    # Transformer architecture — smaller defaults: the wind->CF map is simple and a big model
    # overfit within 2 epochs (train/val gap ~3.4x). dim64/2 layers + dropout generalises better.
    parser.add_argument("--no-cf-lam", dest="no_cf_lam", action="store_true",
                        help="drop the cf_lam channel (wind-only variant); input_dim 10 -> 9. "
                             "No effect on a CERRA dataset, which has no cf_lam to begin with.")
    # k-fold is for the FORECAST tier only, where the only data available to train the
    # post-processor is the same year it must be scored on. The CERRA tier does not need it:
    # CERRA runs from 2020, so it trains on 2020..2024-01 and is scored on 2024-08..2025-07 with
    # a plain chronological separation. NOTE: the run_kfold.py driver was removed; the plumbing
    # below (and infer_cf.py --kfold) still works if you drive the folds yourself.
    parser.add_argument("--n-folds", dest="n_folds", type=int, default=None,
                        help="k-fold CV: split from init timestamps instead of the file's split")
    parser.add_argument("--test-fold", dest="test_fold", type=int, default=None,
                        help="which fold 0..n_folds-1 is held out as this model's test")
    parser.add_argument("--model_dim", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--mlp_mult", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Training
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)

    # Checkpointing
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--run_tag", type=str, default="RegularWeather")

    # ---- data: the self-contained dataset_*.nc from build_targets.py ----
    # (splits and the cf_obs target live IN this file; the loader just reads them)
    parser.add_argument("--dataset_nc", type=str,
                        default="/mnt/weatherloss/WindPowerTransformer/data/cells/dataset_BE_CERRA.nc",
                        help="dataset_*.nc from build_targets.py (inputs + cf_obs + split + geometry)")
    parser.add_argument("--farms_csv", type=str,
                        default="/mnt/weatherloss/WindPower/data/WPDistr/farms.csv")
    parser.add_argument("--specs_csv", type=str,
                        default="/mnt/weatherloss/WindPower/data/WPDistr/turbine_specs.csv")

    return parser.parse_args()
