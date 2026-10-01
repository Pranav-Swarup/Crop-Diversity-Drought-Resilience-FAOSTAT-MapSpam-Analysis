# Metrics: methods note

Factual description of what `src/metrics/` computes. Fixed choices come from `CLAUDE.md`; choices marked **default** are ones `CLAUDE.md` leaves open and can be changed.

## Run

```
python -m src.metrics.all          # run -> stats -> robustness -> figures
USE_DUMMY=1 python -m src.metrics.all
python -m pytest src/metrics       # unit tests
```

## Functions to import

| Function | Returns |
|---|---|
| `detrend.detrend(y, x, frac=0.5)` | DataFrame: `y, trend, resid, anom, index` |
| `diversity.hill1(mean_area)` | effective number of crops |
| `faces.stability(y, resid)` | 1 / CV |
| `faces.resistance(index, event_years)`, `faces.recovery(...)` | median over events |
| `faces.vulnerability(index, years=None)` | share of years with `I_t < 0.9` |
| `events.find_events(climate, thr=-1.0)` | event catalogue (iso3, year, spei12_w, severity) |
| `response.resp_beta(yield_anom, spei)` | OLS slope |
| `response.resp_div(betas, weights)`, `response.resp_divergence(betas)` | response diversity |
| `decompose.decompose(values, resid)` | `phi_sync, mean_crop_cv, cv_total` |
| `run.compute(crops, climate, events, included, frac, thr)` | all output tables |
| `data.analysis_table()` | metrics + region + control means, one row per country |

## Definitions

- **Trend**: LOWESS (`statsmodels`, `frac=0.5`, default 3 robustness iterations) of the series on year, 1993–2023. `anom = (y − trend) / trend`, `index I = y / trend`. Series with fewer than 10 values get no trend.
- **National value**: sum of `value_const` over all crops with a value, per country-year. Imputed rows are kept in this sum (**default**).
- **hill1**: `exp(−Σ p ln p)`, `p` = share of each crop in harvested area summed over the window.
- **stability**: `mean(y) / sd(y − trend)` of national value (sample SD).
- **resistance**: median over events of `I_t / mean(I_{t−3..t−1})`. **recovery**: median over events of `mean(I_{t+1..t+3}) / I_t`. A country needs at least 1 event (**default**); `n_events` is the number used.
- **vulnerability**: share of years with `I_t < 0.9`. **vuln_drought**: the same over years with `spei12_w ≤ −1.0`.
- **Drought event**: first year of a run of years with `spei12_w ≤ −1.0`, with 3 years before and after inside 1993–2023 and no drought year in the 3 years before.
- **resp_beta**: per crop, yield is detrended on non-imputed years only, then `yield_anom` is regressed on `spei12_w` (OLS slope). Crops need ≥ 20 non-imputed years and ≥ 1% of the country's total value over the window.
- **resp_div**: `sqrt(Σ w_i (β_i − β̄)²)`, `w_i` = value share renormalised over the eligible crops, `β̄` the weighted mean.
- **resp_divergence** (Ross et al. 2023, section 3.1): `(max β − min β − | |max β| − |min β| |) / (max β − min β)`. Unweighted. 0 when all slopes share a sign, 1 when `min β = −max β`.
- **Portfolio decomposition**: on crops with a value in all 31 years (**default**). Per-crop residual `r_i = y_i − LOWESS_i`.
  `phi_sync = Var(Σ r_i) / (Σ sd_i)²`, `mean_crop_cv = Σ w_i · sd_i / mean_i`, `w_i = mean_i / Σ mean_j`.
  `cv_total = sd(Σ r_i) / mean(Σ y_i) = sqrt(phi_sync) · mean_crop_cv`, asserted to 1e-9.
  `cv_total` here is built from summed per-crop residuals on the balanced crop set, so `1 / cv_total` is close to but not identical to `stability`. Both are in `outputs/metrics_diagnostics.csv`.

## Outputs

| File | Content |
|---|---|
| `outputs/metrics.parquet` | contract file, one row per included country |
| `outputs/crop_anomalies.parquet` | contract file; eligible crops, non-imputed years only |
| `outputs/metrics_national.parquet` | iso3, year, value_const, trend, resid, anom, index, spei12_w, is_drought, is_event |
| `outputs/metrics_diagnostics.csv` | per country: crops used in each metric, value share covered, event counts |
| `outputs/stats/spearman.csv` | rho, bootstrap 95% CI (2000 resamples, percentile), p, n |
| `outputs/stats/ols.csv` | standardised betas, HC3 SE and CI, n, R² |
| `outputs/stats/r2_comparison.csv` | hill1-only vs resp_div-only vs both, same countries |
| `outputs/stats/robustness.csv` | Spearman table under each variant |
| `outputs/stats/summary.csv` | sample sizes and medians |
| `figures/results/01..04_*.png` | the four result figures |

## Statistics

- Spearman CIs resample countries with replacement (fixed seed).
- OLS: outcome and all terms z-scored on the complete-case sample; controls are window means (`log gdp_pc`, `fert_kg_ha`, `irrig_share`). The merged entities BLX, SCG and SDX have no WDI series and drop out of the OLS.
- Robustness variants: LOWESS `frac` 0.3 and 0.7; severe droughts only (`spei12_w ≤ −1.5`); countries with ≤ 10% of value imputed; each region left out in turn.
