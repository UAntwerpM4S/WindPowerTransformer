import os

windparks = ['Belwind Phase 1', 'Thorntonbank - C-Power - Area NE',
       'Thorntonbank - C-Power - Area SW', 'Mermaid Offshore WP',
       'Nobelwind Offshore Windpark', 'Norther Offshore WP', 'Northwester 2',
       'Northwind', 'Rentel Offshore WP', 'Seastar Offshore WP']

model_name = "GraphTransformer"
model_dim = 64
n_heads = 4
num_layers = 4 

base_command = (
    "python3 Train.py "
    "--model_name {model_name} "
    "--input_dim 6 "
    "--model_dim {model_dim} "
    "--n_heads {n_heads} "
    "--num_layers {num_layers} "
    "--mlp_mult 4 "
    "--batch_size 8 "
    "--epochs 25 "
    "--lr 0.001 "
    "--patience 5 "
    "--windpark \"{windpark}\""
)


for windpark in windparks:
    cmd = base_command.format(
            model_name=model_name,
            model_dim=model_dim,
            n_heads=n_heads,
            num_layers=num_layers,
            windpark=windpark)
    print(f"\n🚀 Running: {cmd}")
    os.system(cmd)
