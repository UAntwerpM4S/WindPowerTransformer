import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Train Transformer on Wind data")

    # Model & data
    parser.add_argument("--input_dim", type=int, default=6, help="Input feature dimension")
    parser.add_argument('--windpark', type=str, default="Belwind Phase 1",
                    help="Windpark name to train the model on")

    # Transformer architecture
    parser.add_argument("--model_dim", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--mlp_mult", type=int, default=2)

    # Training
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)

    # Checkpointing
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--run_tag", type=str, default="CERRA")

    # ---- CERRA pseudo-forecast training ----
    parser.add_argument(
        "--zarr_path", type=str,
        default="/mnt/weatherloss/WindPower/data/EGU26/Anemoidatasets/New_Cerra_A_large.zarr",
        help="Anemoi-format CERRA zarr (data/latitudes/longitudes/dates, attrs.variables)",
    )
    parser.add_argument(
        "--metadata_path", type=str,
        default="/mnt/weatherloss/WindPower/data/NorthSea/Power/windfarm_metadata.csv",
        help="windfarm_metadata.csv with cerra_grid_lat/cerra_grid_lon per farm",
    )
    parser.add_argument("--target", type=str, default="power",
                        help="CERRA variable used as target (injected observed power)")
    parser.add_argument("--lead_hours", type=int, default=36,
                        help="Pseudo-forecast horizon; window length is lead_hours+1 steps")
    parser.add_argument("--stride", type=int, default=1,
                        help="Hours between consecutive pseudo-forecast inits")
    parser.add_argument("--ensemble", type=int, default=0,
                        help="Ensemble member index to read from the zarr")

    # Date-range splits (init timestamp falls in the range)
    parser.add_argument("--train_start", type=str, default="2020-01-01")
    parser.add_argument("--train_end",   type=str, default="2024-01-31 23:00")
    parser.add_argument("--val_start",   type=str, default="2024-02-01")
    parser.add_argument("--val_end",     type=str, default="2024-07-31 23:00")
    parser.add_argument("--test_start",  type=str, default="2024-08-01")
    parser.add_argument("--test_end",    type=str, default="2025-07-31 23:00")

    return parser.parse_args()
