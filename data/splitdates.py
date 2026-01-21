import os
import glob
import pickle
from datetime import datetime

# Base FCS directory that contains the 3 subfolders
fcs_base = "FCS"
subdirs = ["GNN", "GraphTransformer", "Transformer"]

def list_timestamps(folder):
    """Return set of timestamps (YYYYMMDDHHMMSS) found in fcs.*.nc in a folder."""
    files = glob.glob(os.path.join(folder, "fcs.*.nc"))
    ts = set()
    for f in files:
        name = os.path.basename(f)
        parts = name.split(".")
        if len(parts) >= 3:
            ts.add(parts[1])  # the YYYYMMDDHHMMSS part
    return ts

# Collect timestamps per model folder
ts_sets = []
for sd in subdirs:
    folder = os.path.join(fcs_base, sd)
    s = list_timestamps(folder)
    print(f"{sd}: {len(s)} files")
    ts_sets.append(s)

# Intersection: only timestamps present in ALL 3 folders
common_ts = set.intersection(*ts_sets)
print(f"Common timestamps in all 3: {len(common_ts)}")

# Convert to datetime for filtering and sort
date_objs = sorted(datetime.strptime(ts, "%Y%m%d%H%M%S") for ts in common_ts)

def in_range_month(d, start_ym, end_ym):
    """Inclusive check for (year, month) range."""
    ym = (d.year, d.month)
    return start_ym <= ym <= end_ym

# Define splits (inclusive)
train_start, train_end = (2024, 8), (2025, 3)  # 2020-08 .. 2025-03
val_start,   val_end   = (2025, 4), (2025, 5)  # 2025-04 .. 2025-05
test_start,  test_end  = (2025, 6), (2025, 7)  # 2025-06 .. 2025-07

train_dates = [d.strftime("%Y%m%d%H%M%S") for d in date_objs if in_range_month(d, train_start, train_end)]
val_dates   = [d.strftime("%Y%m%d%H%M%S") for d in date_objs if in_range_month(d, val_start, val_end)]
test_dates  = [d.strftime("%Y%m%d%H%M%S") for d in date_objs if in_range_month(d, test_start, test_end)]

print(f"Train: {len(train_dates)} timestamps ({train_start}..{train_end})")
print(f"Val:   {len(val_dates)} timestamps ({val_start}..{val_end})")
print(f"Test:  {len(test_dates)} timestamps ({test_start}..{test_end})")

def save_pickle(obj, filename):
    with open(filename, "wb") as f:
        pickle.dump(obj, f)

save_pickle(train_dates, "train_dates.pkl")
save_pickle(val_dates,   "val_dates.pkl")
save_pickle(test_dates,  "test_dates.pkl")
print("Saved: train_dates.pkl, val_dates.pkl, test_dates.pkl")

