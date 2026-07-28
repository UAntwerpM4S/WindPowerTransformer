import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Train the cell-level CERRA-truth PowerTransformer")

    # Model — input_dim = 6 wind + 3 static cell features (capacity, turbinecount, rated_ws)
    parser.add_argument("--input_dim", type=int, default=9, help="6 wind + 3 static cell features")

    # Transformer architecture
    parser.add_argument("--model_dim", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--mlp_mult", type=int, default=4)

    # Training
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
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
