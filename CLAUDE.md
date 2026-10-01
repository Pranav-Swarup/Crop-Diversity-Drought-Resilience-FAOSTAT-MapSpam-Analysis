# EVST Project: Crop Diversity & Drought Resilience (POC)

## Research question
Does greater crop diversity make national agricultural production more resilient to climatic variability?
POC angle (three upgrades over the literature):
1. **Four faces of resilience** measured separately (Urruty et al. 2016): stability, resistance, recovery, vulnerability.
2. **Response diversity** (Ross et al. 2023): how *differently* a country's crops react to drought, not just how many crops it grows.
3. **Cropland-weighted drought exposure**: SPEI averaged over where crops grow (MapSPAM), not over whole countries.

## Hard rules
- **Never fabricate, interpolate or substitute data.** If a download fails or a source looks different from what is described here, STOP and tell the human. Do not switch to another dataset silently.
- Every download is a script in `src/<module>/download.py`, with the source URL/DOI printed and written to `data/raw/SOURCES.md`.
- Only write to your own module's folders and your own contract files (see Ownership). Read anyone's outputs; never edit them.
- Scripts must run top to bottom from repo root: `python -m src.<module>.<script>`.
- Every pipeline ends with **assertions** on its output (schema, dtypes, no duplicate keys, value ranges) and prints a short summary (rows, countries, years, % missing).
- Python 3.11. Packages: pandas, pyarrow, numpy, scipy, statsmodels, xarray, netCDF4, rioxarray, rasterio, geopandas, exactextract, plotly, matplotlib, requests. Add new ones to `requirements.txt`.
- `data/raw/` is gitignored. `data/clean/` and `outputs/` are committed if < 50 MB each file.

## Fixed analysis choices (do not change without team agreement)
- **Window: 1993–2023.** Starts after the USSR / Czechoslovakia / Yugoslavia / Ethiopia PDR splits; SPEIbase v2.10 ends Dec 2023.
- **Join keys:** `iso3` (str, 3 letters), `year` (int).
- **Merged entities** (to keep series continuous across the window):
  - `BLX` = Belgium + Luxembourg (FAOSTAT "Belgium-Luxembourg" to 1999, then BEL + LUX summed)
  - `SCG` = Serbia + Montenegro (+ Kosovo in geometry) (combined to 2005, then summed)
  - `SDX` = Sudan (former) to 2011, then Sudan + South Sudan summed
  - China = FAOSTAT "China, mainland" → `CHN`. Drop the "China" aggregate. Taiwan → `TWN`.
- **Country inclusion:** mean harvested area ≥ 100,000 ha, ≥ 5 crops, ≥ 25 non-imputed years of national production.
- **Production weighting:** FAOSTAT Gross Production Value, constant 2014–2016 international $ (`value_const`). Never sum tonnes across crops.
- **Yield:** always compute `prod_t / area_ha` ourselves (FAOSTAT yield units changed over versions).
- **Imputed data:** FAOSTAT flags `E` (estimated) and `I` (imputed) → `is_imputed = True`. Confirm flag meanings from the flags file inside the FAOSTAT zip.
- **Detrending:** per series, LOWESS (`statsmodels`, `frac=0.5`) on the 1993–2023 series. Relative anomaly `a_t = (y_t − trend_t) / trend_t`. Index `I_t = y_t / trend_t`.
- **Drought event:** `spei12_w ≤ −1.0` (SPEI-12 at December, cropland-weighted). Event year = first year of a run of drought years. Needs 3 years before and 3 after inside the window, and the 3 years before must be drought-free.

## Metric definitions (implemented once, in `src/metrics/`; everyone imports from there)
All on the national value index `I_t` (sum of `value_const` across crops, detrended) unless stated.
- `hill1` = exp(Shannon entropy) of mean harvested-area shares over the window (effective number of crops).
- `stability` = 1 / CV of detrended residuals of national value.
- `resistance` = median over events of `I_t / mean(I_{t-3..t-1})` (Lloret et al. 2011).
- `recovery` = median over events of `mean(I_{t+1..t+3}) / I_t`.
- `vulnerability` = share of years with `I_t < 0.9` (loss > 10% below trend). Also `vuln_drought` = same, restricted to drought years.
- `resp_beta` (per crop) = OLS slope of crop yield anomaly on `spei12_w`, crops with ≥ 20 non-imputed years and ≥ 1% mean value share.
- `resp_div` = value-weighted SD of `resp_beta` across a country's crops. `resp_divergence` = Ross et al. 2023 divergence on the same slopes (check the paper for the exact formula before implementing).
- Portfolio decomposition (Loreau & de Mazancourt 2008; Thibaut & Connolly 2013), on detrended per-crop value residuals:
  `phi_sync = Var(total) / (Σ sd_i)^2`, `mean_crop_cv = Σ w_i · CV_i` (w_i = mean value share),
  so `CV_total = sqrt(phi_sync) · mean_crop_cv`. Assert this identity holds numerically.

## Data contract
| File | Owner | Columns |
|---|---|---|
| `data/clean/countries.csv` | Gursahib | iso3, name, region, member_iso3 (semicolon list, for merged entities) |
| `data/clean/crops.parquet` | Gursahib | iso3, year, item_code, item, area_ha, prod_t, yield_t_ha, value_const, is_imputed |
| `data/clean/controls.parquet` | Gursahib | iso3, year, gdp_pc, fert_kg_ha, irrig_share |
| `data/clean/quality.csv` | Gursahib | iso3, n_crops, share_imputed, included (bool), reason |
| `data/clean/climate.parquet` | Prathmesh | iso3, year, spei12_w, spei12_unw, spei6min_w |
| `data/clean/events.csv` | Prathmesh | iso3, year, spei12_w, severity (`moderate` ≤ −1.0, `severe` ≤ −1.5, `extreme` ≤ −2.0) |
| `data/clean/countries.geojson` | Prathmesh | Natural Earth admin-0 dissolved to our iso3 codes; property `iso3` |
| `outputs/metrics.parquet` | Pranav | iso3, hill1, stability, resistance, recovery, vulnerability, vuln_drought, n_events, resp_div, resp_divergence, phi_sync, mean_crop_cv |
| `outputs/crop_anomalies.parquet` | Pranav | iso3, year, item, yield_anom, value_share, resp_beta |
| `outputs/stats/*.csv` | Pranav | correlation and regression tables |
| `outputs/validation/*.csv` | Arushi | drought hit-rate table, case-study tables |
| `explorer/index.html` | Pragya | single self-contained file |

Until a real file exists, build against dummy data with exactly these columns: `python -m src.common.make_dummy`.

## Ownership (folders)
- Pranav: `src/common/`, `src/metrics/`, `outputs/metrics*`, `outputs/crop_anomalies*`, `outputs/stats/`, `figures/results/`
- Gursahib: `src/crops/`, the four `data/clean/` files above that he owns, `figures/data/`
- Prathmesh: `src/climate/`, `climate.parquet`, `events.csv`, `countries.geojson`, `figures/climate/`
- Arushi: `src/validation/`, `outputs/validation/`, `figures/validation/`
- Pragya: `src/explorer/`, `explorer/`, `figures/deck/`

## Figures
Static: matplotlib, PNG at 300 dpi, 8×5 in, white background, sans-serif, label every axis with units, source line in the caption. Colour-blind-safe palette (`tab10` or viridis). Filenames `figures/<owner>/<nn>_<short_name>.png`.

## Writing
Slide text and interpretation are written by the team, not generated. Claude writes code, figures and short factual README notes only.
