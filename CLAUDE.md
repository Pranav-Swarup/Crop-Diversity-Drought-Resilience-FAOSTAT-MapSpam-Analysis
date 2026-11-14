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

## Phase 2 extensions
Additions on top of the POC. Nothing in "Fixed analysis choices", "Metric definitions" or the data contract above changes; every extension is a new script writing new files. All hard rules apply.

### Extension definitions
- **Panel model** (`src/metrics/panel.py`): country-year regression of the national value anomaly `anom` on `spei12_w` and its interactions with `z(hill1)` and `z(resp_div)`, country and year fixed effects, standard errors clustered by country. `hill1` and `resp_div` are time-invariant, so only their interactions with SPEI are identified. Variants: dry/wet split (`min(spei,0)`, `max(spei,0)`), control interactions (log gdp_pc, fert_kg_ha, irrig_share), time-varying lagged `hill1`, excluding mostly-imputed country-years, and a split sample.
- **Split-sample `resp_div`:** `resp_div` is estimated from the same yields and SPEI as the panel outcome, so its interaction with SPEI is partly mechanical. The `split_resp_div` variant estimates `resp_div` on odd years and fits the model on even years. For that variant only, the per-crop minimum is 10 years instead of 20 (odd years give at most 16). Any claim about `resp_div` in the panel must cite this variant, not the core model.
- **Wet event:** exact mirror of the drought event: `spei12_w ≥ +1.0`, event year = first year of a run of wet years, 3 years before and after inside the window, 3 years before wet-free. Severity `moderate` ≥ 1.0, `severe` ≥ 1.5, `extreme` ≥ 2.0. Implemented by calling `src.metrics.events.find_events` on the sign-flipped series (the rule is not copied).
- `resistance_wet`, `recovery_wet` = same formulas as `resistance`, `recovery`, over wet events. `vuln_wet` = share of wet years (`spei12_w ≥ 1.0`) with `I_t < 0.9`.
- **Out-of-sample test** (`src/metrics/oos.py`): train 1993–2012, test 2013–2023. Yield anomalies use the fixed full-window LOWESS trend; only the response slope is out-of-sample. Train slopes need ≥ 15 non-imputed training years (the 20-year rule cannot be met inside a 20-year window), test needs ≥ 8 years.
- **Crop portfolio** (`src/metrics/portfolio.py`): decision variable `w` = harvested-area shares across crops a country already grows (complete 31-year series, ≥ 20 non-imputed years, ≥ 1% value share). `mu_i` = area-weighted LOWESS trend of `value_const / area_ha` over 2019–2023. `exp_value_ha = Σ w_i·mu_i`; `s_i = w_i·mu_i / exp_value_ha`; `risk_sd = sqrt(s' Σ s)` with Σ the covariance of relative anomalies of value per ha (imputed years removed, correlations shrunk towards a constant-correlation target); `drought_change = −1.5 · Σ s_i·resp_beta_i`. Each share stays within 0.5–1.5× and ±10 percentage points of its current value; no new crops. The model assumes a crop's value per ha and its variability do not change when its area changes; outputs are scenarios, not recommendations. `evaluate()` is the single implementation; the explorer's JavaScript mirrors it and `src/explorer/build.py` asserts they agree.
- **Cereal import dependency** (`src/crops/trade.py`): `import_dep = import_t / (prod_t + import_t − export_t)` from FAOSTAT Food Balances, item "Cereals - Excluding Beer". FBSH (old methodology) for 1993–2009, FBS for 2010–2023; the two are never blended, `fbs_source` records which, and series are detrended per segment. Published negative values (IRQ 2010 export, KNA 2020 supply) are kept as published.
- **Producer prices** (`src/crops/prices.py`): FAOSTAT annual producer price, USD/tonne, nominal (no deflator in the project; analyses use LOWESS relative anomalies). Merged entities: single reporting member, or production-weighted mean when several report.
- **Calories** (`src/crops/calories.py`): `kcal = prod_t × kcal_per_t`. One time-invariant factor per crop item = median over country-years of FAOSTAT SUA `Calories/Year ÷ Food supply quantity` (2010–2023). Items need ≥ 10 country-years from ≥ 3 areas; fibre crops, tobacco, rubber, oil palm fruit and others have no factor and are left out (`data/clean/calorie_factors.csv`). It is the potential food energy of primary production, not consumption; no conversion chains to processed products. `*_kcal` faces = the same formulas as the value faces on the national kcal sum; a country needs ≥ 50% of crop value covered.

- **Heat** (`src/climate/download_cru.py`, `src/climate/heat.py`, `src/metrics/heat.py`): CRU TS v4.10 monthly maximum temperature (0.5°, same grid as SPEI). Per cell, mean Tmax of the cell's 3 climatologically warmest calendar months (ranked on 1993–2023 monthly means, taken inside the calendar year); `tmx_warm_w` = cropland-weighted (MapSPAM) country mean, °C. `heat_z_w` = (`tmx_warm_w` − its LOWESS trend, `frac=0.5`) / SD of that residual, per country: hot relative to the temperature expected for that year. Monthly data measures hot warm seasons, not daily extreme degree days. `stn_share_w` = cropland-weighted share of warm-month cells with ≥ 1 contributing station; where it is 0 CRU returns climatology.
- **Hot year:** `heat_z_w ≥ +1.0`. **Heat event:** mirror of the drought event on `heat_z_w` (`find_events` on the sign-flipped series; severity `moderate` ≥ 1.0, `severe` ≥ 1.5, `extreme` ≥ 2.0). **Compound year:** `spei12_w ≤ −1.0` and `heat_z_w ≥ +1.0`.
- `resistance_heat`, `recovery_heat` = same formulas as `resistance`, `recovery`, over heat events. `vuln_heat`, `vuln_compound` = share of hot / compound years with `I_t < 0.9`.
- **Heat panel** (`outputs/stats/heat_panel.csv`, estimator `src.metrics.panel.fe_fit`): `heat_only`, `additive` (SPEI and heat, each interacted with `z(hill1)`), `additive_stn` (drops countries with mean `stn_share_w` < 0.5), `year_types` (0/1 dummies `dry_only`, `hot_only`, `compound`; `compound_minus_sum` = compound − (dry_only + hot_only)), `year_types_hill1`.

### Phase 2 data contract
| File | Module | Columns |
|---|---|---|
| `data/clean/wet_events.csv` | `src/climate/wet_events.py` | iso3, year, spei12_w, severity |
| `data/clean/heat.parquet` | `src/climate/heat.py` | iso3, year, tmx_warm_w, tmx_warm_trend, heat_z_w, stn_share_w |
| `data/clean/heat_events.csv` | `src/climate/heat.py` | iso3, year, heat_z_w, severity |
| `data/clean/trade.parquet` | `src/crops/trade.py` | iso3, year, prod_t, import_t, export_t, supply_t, import_dep, fbs_source |
| `data/clean/prices.parquet` | `src/crops/prices.py` | iso3, year, item_code, item, price_usd_t, is_imputed |
| `data/clean/calories.parquet` | `src/crops/calories.py` | iso3, year, item_code, item, kcal, is_imputed |
| `data/clean/calorie_factors.csv` | `src/crops/calories.py` | item_code, item, kcal_per_t, kcal_per_t_ratio, n_country_years, n_areas, p05, p25, p75, p95, iqr_rel, share_within_1pct, in_scope, reason |
| `outputs/metrics_wet.parquet` | `src/metrics/wet.py` | iso3, resistance_wet, recovery_wet, vuln_wet, n_wet_events |
| `outputs/metrics_heat.parquet` | `src/metrics/heat.py` | iso3, resistance_heat, recovery_heat, vuln_heat, vuln_compound, n_heat_events, n_hot_years, n_compound_years |
| `outputs/metrics_kcal.parquet` | `src/metrics/calorie_faces.py` | iso3, stability_kcal, resistance_kcal, recovery_kcal, vulnerability_kcal, vuln_drought_kcal, n_events_kcal, kcal_value_coverage |
| `outputs/portfolio_crops.parquet` | `src/metrics/portfolio.py` | key iso3, item_code: item, in_portfolio, exclude_reason, value_share, area_share_all, area_share, share_lo, share_hi, value_ha, sd_anom, n_years, resp_beta, has_beta |
| `outputs/portfolio_frontier.parquet` | `src/metrics/portfolio.py` | key iso3, point, item_code: kind (`frontier` 0–19, `current` 20, `min_risk_at_current` 21), target_value_ha, exp_value_ha, risk_sd, drought_change, drought_coverage, item, weight |
| `outputs/portfolio_cov.json` | `src/metrics/portfolio.py` | per country: items, item_codes, w0, lo, hi, mu, cov, beta, shrinkage, mean_corr, frontier |
| `outputs/stats/panel_fe.csv`, `panel_summary.csv` | `src/metrics/panel.py` | model, term, beta, se, ci_lo, ci_hi, p, n_obs, n_countries, r2_within |
| `outputs/stats/wet_spearman.csv`, `wet_vs_dry.csv` | `src/metrics/wet.py` | face, predictor, rho, ci_lo, ci_hi, p, n |
| `outputs/stats/heat_spearman.csv` | `src/metrics/heat.py` | face, predictor, rho, ci_lo, ci_hi, p, n |
| `outputs/stats/heat_panel.csv` | `src/metrics/heat.py` | model, term, beta, se, ci_lo, ci_hi, p, n_obs, n_countries, r2_within |
| `outputs/stats/kcal_spearman.csv`, `kcal_vs_value.csv` | `src/metrics/calorie_faces.py` | face, predictor, rho, ci_lo, ci_hi, p, n |
| `outputs/stats/oos_crop.csv`, `oos_national.csv`, `oos_summary.csv` | `src/metrics/oos.py` | out-of-sample tables |
| `outputs/stats/portfolio_summary.csv` | `src/metrics/portfolio.py` | per-country current vs minimum-risk portfolio |
| `outputs/stats/trade_buffer.csv`, `price_response.csv`, `trade_fbs_overlap.csv` | `src/crops/buffers.py`, `src/crops/trade.py` | trade and price response tables |

The contract tables above (not the stats CSVs or the portfolio files) are in `src/common/contract.py` and checked by `python -m src.common.contract`. Dummy versions exist only for `wet_events`, `metrics_wet` and the portfolio, panel and out-of-sample outputs; `calories`, `trade`, `prices`, `metrics_kcal`, `heat`, `heat_events` and `metrics_heat` have no dummy data: do not run `src.crops.trade`, `prices`, `calories`, `buffers`, `src.climate.heat` or `src.metrics.heat` with `USE_DUMMY=1` (they read the real raw files).

Phase 2 figures: `figures/results/05_panel_interaction.png`, `06_portfolio_frontier.png`, `07_wet_vs_dry.png`, `08_oos_prediction.png`, `09_kcal_vs_value.png`, `10_heat_compound.png`; `figures/data/03_import_dependency.png`, `04_price_response.png`.

Phase 2 run order (after the phase 1 pipeline): `src.crops.download` → `src.crops.trade`, `src.crops.prices`, `src.crops.calories` → `src.climate.wet_events`, `src.climate.download_cru`, `src.climate.heat` → `src.metrics.wet`, `src.metrics.heat`, `src.metrics.panel`, `src.metrics.oos`, `src.metrics.portfolio`, `src.metrics.calorie_faces` → `src.crops.buffers` → `src.explorer.build`, `src.explorer.collect`. `src.crops.download` re-fetches the WDI indicators from the live API each run, so the raw WDI files can change between runs.

Explorer (`explorer/index.html`): adds a portfolio simulator tab per country, wet-event shading, and map layers `vuln_wet`, `resistance_wet`, `import_dep`.

### Phase 2 ownership
Phase 2 files were added in one pass by Gursahib (with Claude). They sit in the module they belong to thematically; the module owner listed under "Ownership" takes them over and should review them. Existing phase 1 files of other owners were not edited, except additive entries in `src/common/paths.py` and `src/common/contract.py`.

### Not done yet (needs data that must be fetched by a person)
These need the gridded raw files (`data/raw/spei/`, `data/raw/mapspam/`, `data/raw/derived/`, `data/raw/cru/`), which are gitignored and exist only on the machines that ran `src.climate` (`src.climate.heat` needs `src.climate.cropland` to have run there). Do not start them without the files; do not substitute another dataset.
- Daily extreme degree days: the heat extension uses monthly CRU TS Tmax; a daily measure needs ERA5 (Copernicus CDS account).
- Observed crop stress: FAO ASIS vegetation health or ESA CCI soil moisture, cropland-weighted.
- Growing-season SPEI windows from crop calendars instead of December SPEI-12 (this would change a fixed analysis choice: team agreement first).
- Subnational case study (India, ICRISAT district data; registration required).

## Phase 3: what makes a portfolio diversified
Three analyses on top of phases 1 and 2. Nothing in "Fixed analysis choices", "Metric definitions" or the earlier contracts changes; the window stays 1993–2023 and the drought variable stays `spei12_w`. All hard rules apply. Added in one pass by Gursahib (with Claude); files sit in the module they belong to and the module owner should review them. None has dummy data: do not run them with `USE_DUMMY=1`.

### Definitions
- **Similarity-sensitive diversity** (`src/metrics/similarity.py`, the single implementation): Leinster & Cobbold 2012, `D_q^Z(p) = (Σ p_i (Zp)_i^(q−1))^(1/(1−q))`, `q = 1`: `exp(−Σ p_i log (Zp)_i)`. `Z` = identity gives the Hill number (`q = 1`: `hill1`). Here `Z_ij = max(0, r_ij)`, `r` = Pearson correlation of two drought-exposure series over 1993–2023.
- **Exposure class** (`src/metrics/crop_classes.py`): every FAOSTAT item mapped to one MapSPAM 2020 class, or a merged class where FAOSTAT cannot be split (coffee = COFF+RCOF, millet = PMIL+MILL; MapSPAM-area-weighted). Mapped from class names because the official SPAM crop list sits behind the Dataverse form; uncertain items are listed in the module. This mapping is separate from `FAOSTAT_TO_MAPSPAM` in `src/metrics/run.py`, which is unchanged (see "Known issues").
- **Exposure-effective diversity** (`src/metrics/exposure_diversity.py`): `p` = class shares of mean harvested area; exposure of a class = its `crop_spei12_w` (`crop_climate.parquet`), national `spei12_w` where MapSPAM has no area for it. `hill1_ms` = Hill-1 at class resolution; `hill1_exp` = the same with `Z`; `exp_ratio = hill1_exp / hill1_ms`; `exp_corr` = share-weighted mean off-diagonal correlation (unclipped). Variants: `_pre` (correlations from 1961–1992, `crop_climate_pre.parquet`), `_season` (SPEI-6 at the harvest month for the 19 classes with a GGCMI calendar, December SPEI-12 for the rest: two index types in one matrix), `_main` (classes mostly in detached territories dropped).
- **Seasonal exposure** (`src/climate/exposure_season.py`): GGCMI Phase 3 crop calendar (Zenodo 10.5281/zenodo.5062513). Harvest month = month of the maturity day; exposure = SPEI-6 of that month. Every calendar file is filled on all land cells, so the rainfed calendar is always used and wheat uses the winter-wheat calendar everywhere; `crop_climate_season_swh.parquet` holds the spring-wheat alternative. It is used only as a similarity layer, never as the drought variable.
- **Severity bins** (`src/metrics/severity.py`): exclusive dummies `moderate` (−1.5 < spei ≤ −1.0), `severe` (−2.0 < spei ≤ −1.5), `extreme` (spei ≤ −2.0), each interacted with `z(hill1)`; estimator `src.metrics.panel.fe_fit`. For the extreme bin cite the clustered, wild-bootstrap or studentised-permutation p-value (`p_perm_t`), not the coefficient permutation (`p_perm`), which is not pivotal. `breadth` = value-weighted share of crops with `yield_anom < 0`; `comove` = weighted mean pairwise product of standardised crop anomalies in a country-year.
- **Bilateral cereal trade** (`src/crops/trade_matrix.py`): FAOSTAT Detailed Trade Matrix, 18 unprocessed cereal grain items, tonnes summed as in the Food Balance cereals aggregate. `trade_partners.parquet` = importer-reported, `trade_partners_mirror.parquet` = the same flows as reported by the exporter. Never blended; no gap filling. Trade inside a merged entity is dropped.
- **Supply diversity** (`src/crops/supply_diversity.py`): sources of a country's cereal supply = own production (`s_dom = Σ prod / Σ (prod + import)`, Food Balances) plus each partner (`(1 − s_dom) × w_p`). Exposure of a source = that country's `spei12_w`. `supply_div` = Hill-1 of source shares, `supply_div_exp` = with `Z`, `partner_corr = Σ w_p r(own, p)`, `partner_dry_share_t = Σ w_p 1[spei_p,t ≤ −1]`. Computed separately for `share_source` = `reporter` (≥ 10 reporting years) and `mirror`. The recorded partner can be a re-exporter, not the grower.

### Phase 3 data contract
| File | Module | Columns |
|---|---|---|
| `data/clean/mapspam_area.parquet` | `src/climate/mapspam_area.py` | iso3, mapspam_crop, area_ha, irr_ha, rf_ha |
| `data/clean/crop_climate_pre.parquet` | `src/climate/exposure_pre.py` | iso3, year (1961–1992), mapspam_crop, crop_spei12_w |
| `data/clean/crop_climate_season.parquet`, `crop_climate_season_swh.parquet` | `src/climate/exposure_season.py` | iso3, year, mapspam_crop, crop_spei6_harv_w |
| `data/clean/trade_partners.parquet`, `trade_partners_mirror.parquet` | `src/crops/trade_matrix.py` | iso3, partner_iso3, year, import_t, import_usd |
| `outputs/exposure_diversity.parquet` | `src/metrics/exposure_diversity.py` | iso3, n_classes, hill1_ms, hill1_exp, exp_ratio, exp_corr, hill2_ms, hill2_exp, fallback_share, rest_share, sd_spei, n_drought_years, log_area, hill1_exp_pre, exp_corr_pre, hill1_exp_season, exp_corr_season, season_share, rf_share, hill1_exp_main, exp_corr_main, overseas_share |
| `outputs/supply_diversity.parquet` | `src/crops/supply_diversity.py` | key iso3, share_source: s_dom, n_partners, n_years, partner_cover, supply_div, supply_div_exp, supply_ratio, import_div, import_div_exp, partner_corr, partner_dry_own_dry, partner_dry_other |
| `outputs/supply_exposure.parquet` | `src/crops/supply_diversity.py` | iso3, share_source, year, partner_spei, partner_dry_share |
| `outputs/stats/expdiv_spearman.csv`, `expdiv_ols.csv`, `expdiv_panel.csv` | `src/metrics/exposure_diversity.py` | correlation, OLS and panel tables |
| `outputs/stats/severity_panel.csv`, `severity_counts.csv`, `severity_robustness.csv`, `severity_breadth.csv` | `src/metrics/severity.py` | panel, bin counts, permutation / leave-one-out / bootstrap, breadth and co-movement |
| `outputs/stats/supply_spearman.csv`, `supply_panel.csv`, `supply_source_compare.csv`, `trade_matrix_vs_fbs.csv` | `src/crops/supply_diversity.py`, `trade_matrix.py` | supply tables |

`crop_climate_pre` (years before the window) and `supply_exposure` are not in `src/common/contract.py`; the other parquet files above are.

Figures: `figures/results/11_exposure_diversity.png`, `12_severity.png`; `figures/data/05_supply_diversity.png`.

Run order (after phases 1 and 2; the gridded scripts need `data/raw/spei/`, `mapspam/`, `derived/`): `src.climate.mapspam_area`, `src.climate.exposure_pre`, `src.climate.download_calendar`, `src.climate.exposure_season` → `src.metrics.exposure_diversity`; `src.metrics.severity` (about 12 minutes, permutations); `src.crops.download_trade_matrix` (420 MB) → `src.crops.trade_matrix` → `src.crops.supply_diversity`.

### Known issues (not fixed; owner to decide)
- `FAOSTAT_TO_MAPSPAM` in `src/metrics/run.py` uses six codes that are not MapSPAM 2020 classes (ACOF, CANO, OCRS, OROO, POTS, SUGS), so 13 items (8.9% of harvested area in included countries) fall back to national SPEI when `resp_beta` is fitted, and 58 items are not mapped. `resp_beta` is also fitted on crop-specific SPEI, not on `spei12_w` as "Metric definitions" states.
- Extending the severity test to 1961–2023 for power would change a fixed analysis choice: team agreement first.

## Figures
Static: matplotlib, PNG at 300 dpi, 8×5 in, white background, sans-serif, label every axis with units, source line in the caption. Colour-blind-safe palette (`tab10` or viridis). Filenames `figures/<owner>/<nn>_<short_name>.png`.

## Writing
Slide text and interpretation are written by the team, not generated. Claude writes code, figures and short factual README notes only.
