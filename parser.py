import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Train Transformer on Wind data")

    # Model & data
    parser.add_argument("--model_name", type=str, default="GraphTransformer", help="Forecast model to train on")
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

    return parser.parse_args()
