import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Train the cell-level CERRA-truth PowerTransformer")

    # Model — input_dim = 6 wind + 3 static cell features (capacity, turbinecount, rated_ws)
    parser.add_argument("--input_dim", type=int, default=9, help="6 wind + 3 static cell features")

    # Transformer architecture — smaller defaults: the wind->CF map is simple and a big model
    # overfit within 2 epochs (train/val gap ~3.4x). dim64/2 layers + dropout generalises better.
    parser.add_argument("--no-cf-lam", dest="no_cf_lam", action="store_true",
                        help="drop the cf_lam channel (wind-only variant); input_dim 10 -> 9")
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
    parser.add_argument("--run_tag", type=str, default="CERRAcell")

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
