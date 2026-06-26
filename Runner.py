import os

windparks = [
    "Belwind Phase 1",
    "Thorntonbank - C-Power - Area NE",
    "Thorntonbank - C-Power - Area SW",
    "Mermaid Offshore WP",
    "Nobelwind Offshore Windpark",
    "Norther Offshore WP",
    "Northwester 2",
    "Northwind",
    "Rentel Offshore WP",
    "Seastar Offshore WP",
]

run_tag = "CERRA"
input_dim = 6
model_dim = 128
n_heads = 4
num_layers = 4

ZARR_PATH = "/mnt/weatherloss/WindPower/data/WindAI/Anemoidatasets/New_Cerra_A_large.zarr"
METADATA_PATH = "/mnt/weatherloss/WindPower/data/NorthSea/Power/windfarm_metadata.csv"

base_command = (
    "python3 Train.py "
    "--run_tag {run_tag} "
    "--input_dim {input_dim} "
    "--model_dim {model_dim} "
    "--n_heads {n_heads} "
    "--num_layers {num_layers} "
    "--mlp_mult 4 "
    "--batch_size 8 "
    "--epochs 25 "
    "--lr 0.001 "
    "--patience 5 "
    "--lead_hours 36 "
    "--freq_hours 3 "
    "--stride 1 "
    "--zarr_path {zarr_path} "
    "--metadata_path {metadata_path} "
    '--windpark "{windpark}"'
)

for windpark in windparks:
    cmd = base_command.format(
        run_tag=run_tag,
        input_dim=input_dim,
        model_dim=model_dim,
        n_heads=n_heads,
        num_layers=num_layers,
        windpark=windpark,
        zarr_path=ZARR_PATH,
        metadata_path=METADATA_PATH,
    )
    print(f"\n🚀 Running: {cmd}")
    os.system(cmd)
