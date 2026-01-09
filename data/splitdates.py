import os
import glob
import pickle
from datetime import datetime

fcs_dir = "FCS/GraphTransformer"
obs_dir = "OBS"  # not used below, but kept for completeness

# Get list of forecast files
fcs_files = sorted(glob.glob(os.path.join(fcs_dir, "fcs.*.nc")))

# Extract timestamps from filenames: fcs.YYYYMMDDHHMMSS.nc
ts_strs = [os.path.basename(f).split(".")[1] for f in fcs_files]
date_objs = [datetime.strptime(ts, "%Y%m%d%H%M%S") for ts in ts_strs]

def in_range_month(d, start_ym, end_ym):
    """Inclusive check for (year, month) range."""
    ym = (d.year, d.month)
    return start_ym <= ym <= end_ym

# Define splits (inclusive)
train_start, train_end = (2024, 8), (2025, 3)  # 2024-08 .. 2025-03
val_start,   val_end   = (2025, 4), (2025, 5)  # 2025-04 .. 2025-05
test_start,  test_end  = (2025, 6), (2025, 7)  # 2025-06 .. 2025-07

# Keep original timestamp strings (YYYYMMDDHHMMSS) since that matches your files
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

