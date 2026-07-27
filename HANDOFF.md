# Handoff — PowerTransformer as a cell-level post-processor

## The new goal (this is what the next chat is for)

Previously the **PowerTransformer** (`Transformer.py`: temporal transformer, `SinusoidalPE` +
`TemporalAttention`) mapped **per-farm** wind → power: one time series per farm, wind in, farm
power out.

**Now: apply it at the CELL level, not the farm level.** Take the anemoi LAM forecast at the
**15 Belgian CERRA cells**, and for each cell feed the PowerTransformer **7 variables**:

- 6 wind: `ws100, ws10, wdir100_cos, wdir100_sin, wdir10_cos, wdir10_sin`
- 1 power: `capacityfactor` (or `power`) — the LAM's own direct power prediction

It runs as a **post-processor on the LAM output** (refine the cell-level power using all 7
forecast variables at that cell), then **aggregate the 15 cells back to per-farm / Belgian-total
power** and score against observations.

Why this is expected to help: the LAM's direct `capacityfactor` channel is **under-dispersive**
(σ_pred/σ_obs ≈ 0.67, optimal ≈ 0.90) and carries a systematic bias; a learned post-processor
that sees wind speed AND direction AND the raw power prediction per cell can correct amplitude,
bias, and wake/direction effects the single-channel LAM output misses. Cell-level (not farm-level)
keeps the spatial structure the distributed target was built to exploit.

---

## How we got from farm level to cell level (the essential background)

This is the "distributed power target" project (successor to VanillaPowerGT). Full detail in the
memory files; the load-bearing facts:

**The distribution (farm → cells).** Each farm's observed power is spread across the CERRA cells
its turbines occupy, weighted by **capacity**, and cells shared by multiple farms **SUM** (power
is extensive):

```
power(cell,t)          = Σ_farms  P_obs(farm,t) · capacity(farm's turbines in cell) / capacity_total(farm)
capacity(cell)         = Σ_farms  capacity of that farm's turbines in the cell        (static)
capacityfactor(cell,t) = power(cell,t) / capacity(cell)                               ∈ [0,1]
```

NaN rule: a cell is NaN at t if **any** contributing farm is NaN (can't sum a known + unknown).
The target trained is **capacityfactor** (homogeneous, bounded [0,1]); `capacity`, `turbinecount`,
`turbmask` are static per-cell forcings.

**The reconstruction (cells → farm), which is what you'll use to aggregate.** The adjoint of the
distribution:

```
P_pred(farm,t) = Σ_cell  capacity(farm's turbines in cell) · CF_pred(cell,t)
```

Exact for a farm's un-shared cells (a perfect CF field reproduces P_obs). This is implemented in
`WindAI/data/Wpower/score_power_configs.py` → `build_reconstruction()` (KD-tree snaps turbines to
forecast cells; `G[farm,cell]` = capacity of that farm in that cell). Reuse it.

**Geometry.** CERRA grid = 72,668 cells, 3-hourly. Belgium's 10 farms occupy **15 cells (12
shared)**; the full BE+UK set is 172 cells. **This work is Belgium-only** → the 15 cells. Belgium
is the evaluation target; UK is auxiliary training signal whose observations end in 2023.

---

## Where everything lives

**Metadata dir** — `/mnt/weatherloss/WindPower/data/WPDistr/` on the server (= `WindAI/data/Wpower/`
in the laptop repo). Regenerate any CSV by running its script; all fail loudly.

| file | what |
|---|---|
| `farms.csv` | 32 farms: obs columns, turbine count, capacity, fleet, cut_in/cut_out. BE = 10 farms, 2261.2 MW |
| `turbines.csv` | one row per turbine: farm, lon, lat, **capacity_mw** ← the distribution/reconstruction weights |
| `power_obs.csv` | **the observations**: 16,312 rows × 31 farm columns, 3-hourly mean MW, UTC, 2020-01→2025-07 |
| `turbine_specs.csv` | per-model cut_in, rated_ws, cut_out, rated_power_mw (hand-curated) |
| `coordinates/` | per-farm turbine CSVs (cleaned) |
| `build_power.py` | builds the target zarr `power_cerra_src.zarr` (+ ERA5 companion): power, capacityfactor, capacity, turbinecount, turbmask |
| `farm_metadata.py`, `extract_uk_turbines.py`, `build_power_obs.py`, `validate_against_nost.py` | the metadata pipeline |
| `recipes/` | `power_{cerra,era5}_{build,join}.yaml` → anemoi-datasets create → `Anemoidatasets/power_cerra_A.zarr` (inner) + `power_era5_A.zarr` (outer) |
| `score_power_configs.py` | **the scorer** — reconstruction, power-curve baseline, MAE/RMSE/bias, MSE decomposition, persistence. Reuse `build_reconstruction`, `farm_wind`, `build_farm_curves` |
| `check_forecast_vars.py` | which forecast files are missing a variable (dirs can be heterogeneous) |

**Training config** — `WindAI/training/WPDistr/*.yaml` (graph transformer LAM, `capacityfactor`
diagnostic, min-max norm, `HardtanhBounding[0,1]`, the loss-weight sweep).

**Inference** — `/mnt/weatherloss/WindPower/inference/WPDistr/<run>/forecast_YYYYMMDDHHMMSS.nc`
(one per init, rollout of lead times, on the inner cutout grid, carrying lat/lon). Runs:
`HighCapacityGT` (best), `VeryHighCapacity`, `ExtremelyHighCapacity`, `VanillaPowerGT`, and
`RegularWeather` (weather-only baseline). **These forecasts are the PowerTransformer's INPUT.**

---

## Results so far (why the direct LAM output needs post-processing)

Belgium total, 2024-08→2025-07, pooled over 3–36 h leads:

| series | MAE % | RMSE MW | bias MW | r | σp/σo |
|---|---|---|---|---|---|
| HighCapacityGT · direct | **11.39** | 374 | −151 | 0.904 | **0.67** |
| VanillaPowerGT · direct | 12.32 | 396 | −138 | 0.887 | 0.63 |
| HighCapacityGT · power curve | 12.03 | 393 | +198 | 0.900 | 1.11 |
| RegularWeather · power curve | 11.86 | 393 | +181 | 0.897 | 1.13 |
| persistence | 21.97 | 721 | −1 | 0.471 | 1.00 |

Key takeaways feeding the new work:
- **The loss weight is the mechanism**: with the tiny VanillaPower weight the direct channel ties
  its own power curve (12.32 vs its curve); the high weight makes direct win (11.39). Correlation
  rose 0.887 → 0.904.
- **The direct forecast is badly under-dispersive** (σp/σo 0.67 vs optimal r=0.90). Just inflating
  amplitude is ~10% RMSE for free — a post-processor should fix this automatically.
- **Sign separation across all 10 farms**: power curve over-predicts (+2..+19%, no wake losses),
  direct under-predicts (−1..−14%). Much of direct's MAE win is a *calibration* advantage — a
  reviewer will ask about bias correction. A learned post-processor is the principled fix.
- Persistence wins only at +3 h; from +6 h the LAM dominates.

So the PowerTransformer's job: take the 7-variable cell forecast and produce a better-calibrated,
better-dispersed cell power than the LAM's raw single channel — then aggregate.

---

## Gotchas to carry over

- **Time convention**: `power_obs` value at t = mean MW over `[t, t+3h)`, UTC, no DST. It **leads**
  the instantaneous CERRA/forecast field by ~1.5 h (uniform across all farms). BE obs = ENTSO-E/Elia;
  UK obs = Elexon BMRS (they differ a few % — different feeds, not a bug).
- **ALLOWED_YEARS** (in `aggregate_uk_wind.py`) = Nøst (2025) full-capacity periods, NOT arbitrary.
  Do not "improve" it. UK obs end 2023 → **evaluate on Belgium only**, 2024–25.
- **Inference dirs can be heterogeneous** (an early pass without the power var + a later pass with
  it). Verify with `check_forecast_vars.py`; the scorer already drops inits missing the variable.
- **The forecast NetCDF grid is a subset** of the full CERRA grid (~24k–27k inner cells, and can
  differ between files). Match cells to turbines by **lat/lon**, never by index (the reconstruction
  in `score_power_configs.py` already caches per distinct grid).
- **anemoi LR is auto-scaled** by `num_gpus_per_node` internally — the config `lr.rate` is per-GPU.
  (Not relevant to a standalone PowerTransformer, but relevant if you fine-tune the LAM.)

## Immediate first steps for the new chat

1. Extract the 7 variables at the 15 Belgian cells from the inference NetCDFs (use the BE-cell
   subset of the reconstruction's `cell_idx`).
2. Build train/val/test from `power_obs.csv` aggregated to cells via `build_power.py`'s target
   (`capacityfactor(cell,t)`), or score directly against per-farm obs after aggregating.
3. Adapt `Transformer.py` from a per-farm series model to ingest a (cells × 7-var × time) input;
   decide whether cells are batched independently or attended jointly.
4. Aggregate PowerTransformer cell output → farm/total via `P_pred(farm)=Σ_cell cap(farm,cell)·CF`,
   score with `score_power_configs.py`'s machinery, compare to the LAM's raw direct forecast.
