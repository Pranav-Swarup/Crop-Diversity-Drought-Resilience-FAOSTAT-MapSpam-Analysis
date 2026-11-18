# Project Metadata: Repository Structure, Ownership & Change Log

The ownership sections record the original contributions. Their line counts are historical. Current results are reported in [the research report](report.md) and the underlying [statistical tables](outputs/stats).

---

## November 2026 verification

Pmjarvis added the district loader, both India map scopes, district time series and the results tab on 18 November (`9edefc4`, `80fae80`, `056f8d4`). The next three commits added district figures and the pipeline diagram, rebuilt the explorer and its tests, and updated the project report (`da6bcb3`, `51646e5`, `09e1486`).

The subsequent verification corrected the national crop exposure crosswalk, restricted trend fitting and value weights to training years in the holdout, and pinned the two WDI release files. The district rebuild corrected the Bengaluru matches, preserved missing crop areas, converted the price applied to milled rice, and excluded 66 mirror irrigation shares outside the valid range. Both district models and their seasonal and spatial robustness checks were rerun. The archived five-crop workbook matched the publisher’s SHA-256. [National methods](src/metrics/METHODS.md) · [District sources and methods](src/india/README.md)

The explorer now bundles its map and plotting assets, preserves gaps and numeric precision, aligns kharif coefficients and intervals with their labels, and exposes the saved results as tables. Browser checks covered all 36 map layers across the three scopes. With HTTP and HTTPS blocked, the real explorer displayed 53 tables with 9,532 rows and 76,560 data cells. The synthetic build displayed 15 tables with 613 rows and 5,761 data cells. The verification independently reproduced the primary national and district fixed-effects estimates and the national severity estimates. All 2,310 saved portfolio evaluations agreed with the JavaScript evaluator to rounding precision. The full regression suite passed all 142 tests, including offline Chromium. The table inventory covered 133 CSV and Parquet files with 2,051,701 rows and 14,504,445 cells across the cleaned data, release snapshot, outputs and synthetic fixtures. [Explorer regression checks](tests/test_explorer.py) · [Offline browser checks](tests/test_explorer_browser.py) · [National models](outputs/stats/panel_fe.csv) · [District models](outputs/stats/india_panel.csv)

## 1. Repository Structure Overview

```
Crop-Diversity-Drought-Resilience-FAOSTAT-MapSpam-Analysis/
├── CLAUDE.md                  # Project rules, data contracts, metric definitions (shared)
├── PROMPTS.md                 # Task assignments per team member (shared)
├── README.md                  # Run order and project overview (Pranav)
├── report.md                  # Comprehensive research report (Prathmesh — NEW)
├── metadata.md                # This file (Prathmesh — NEW)
├── requirements.txt           # Python dependencies (Pranav)
├── references/                # Academic papers referenced by the project
│   ├── EVST Literature Review.pdf
│   └── Methods Ecol Evol - 2023 - Ross - How to measure response diversity.pdf
│
├── src/
│   ├── common/                # Shared infrastructure (Pranav)
│   ├── crops/                 # FAOSTAT data pipeline (Gursahib)
│   ├── climate/               # Climate exposure pipeline (Prathmesh)
│   │   └── analysis/          # Extended analyses (Prathmesh — NEW)
│   ├── metrics/               # Resilience metrics engine (Pranav)
│   ├── india/                 # District sources, cleaning, climate and models
│   ├── validation/            # Drought validation and country case studies (Arushi)
│   └── explorer/              # Offline national and Indian district dashboards (Pragya)
│
├── data/
│   ├── raw/                   # Raw downloaded data (.gitignored)
│   ├── sources/               # Versioned national WDI extracts
│   ├── clean/                 # Processed contract files (committed)
│   └── dummy/                 # Synthetic test data (Pranav)
│
├── outputs/                   # Analytical results
│   ├── stats/                 # Regression and correlation CSVs
│   ├── india/                 # District panel and metrics
│   ├── validation/            # Hit rates, case slopes and district drought checks
│   └── dummy/                 # Dummy analytical results (Pranav)
│
├── figures/
│   ├── climate/               # Climate pipeline figures (Prathmesh)
│   ├── data/                  # Data quality figures (Gursahib)
│   ├── results/               # Metrics figures (Pranav)
│   ├── india/                 # District panel and severity figures
│   ├── validation/            # Hit-rate and country case figures
│   └── deck/                  # Assembled presentation deck (Pragya)
│
└── explorer/                  # Standalone real and synthetic HTML explorers
```

---

## 2. Prathmesh's Work — Climate Exposure Pipeline (`src/climate/`)

### 2.1 Files Already Present Before This Session (Steps 1–5)
These scripts were written as a "first iteration" by the metrics team to bootstrap the pipeline. Prathmesh verified their correctness but did not modify them (except `cropland.py`).

| File | Lines | What It Does |
|---|---|---|
| `download.py` | 50 | Downloads SPEIbase v2.10 netCDFs (SPEI-6, SPEI-12), MapSPAM 2020 zip, and Natural Earth shapefiles. Logs sources to `data/raw/SOURCES.md`. Includes a help message for MapSPAM's manual download (Harvard Dataverse guestbook). |
| `geometry.py` | 46 | Reads Natural Earth 1:10m admin-0 shapefile, maps to ISO3 codes using `ISO_A3_EH` (falling back to `ADM0_A3`), dissolves merged entities (BLX, SCG, SDX). Outputs `data/clean/countries.geojson`. |
| `aggregate.py` | 110 | The core climate processing script. Reads SPEIbase `.nc` files and the cropland `.npy` raster. For each country (via `exactextract`), computes 3 indices: `spei12_w` (cropland-weighted December SPEI-12), `spei12_unw` (area-weighted), and `spei6min_w` (minimum 6-month seasonal SPEI). Outputs `data/clean/climate.parquet`. Also generates per-crop specific SPEI data by iterating over `cropland_05deg.npz` and outputs `data/clean/crop_climate.parquet` when using the `--per-crop` flag. |
| `events.py` | 23 | Reads `climate.parquet` and identifies drought events per CLAUDE.md rules (≤ -1.0 threshold, 3-year pre-event window, 3-year post-event window). Outputs `data/clean/events.csv`. |

### 2.2 Files Created by Prathmesh This Session

| File | Lines | Status | What It Does |
|---|---|---|---|
| **`figures.py`** | 154 | **NEW** (Step 6) | Generates the 3 required climate figures (`01_weighted_vs_unweighted.png`, `02_cropland_weights.png`, `03_event_counts_map.png`) plus a bonus LogNorm variant (`02_cropland_weights_lognorm.png`). Prints validation metrics: 230 countries, 431 events, top-10 weighted vs. unweighted divergences. |
| **`cropland.py`** | 76 | **MODIFIED** | Originally saved a single flat `.npy` (summed across all 46 crops). Now produces **both**: the legacy `cropland_05deg.npy` and a new `cropland_05deg.npz` containing 46 individual crop arrays, which `aggregate.py` now uses to calculate crop-specific climate exposures. |
| **`analysis/__init__.py`** | 1 | **NEW** | Package init for the analysis submodule. |
| **`analysis/response_paradox.py`** | 153 | **NEW** | Fully reproducible script investigating the Response Diversity Paradox. Computes `max_abs_beta` per country, runs Spearman/Pearson correlations, OLS regression controlling for climate variance, generates a 3-panel scatter figure (`04_response_diversity_paradox.png`), and saves results to `outputs/stats/response_paradox.csv`. |
| **`analysis/spatial_and_flash.py`** | 166 | **NEW** | Fully reproducible script comparing spatial averaging and seasonal and annual drought indices. Correlates country area with SPEI variance and crop sensitivity, compares `spei12_w` vs `spei6min_w` predictive power globally and during severe droughts, generates a 3-panel figure (`05_spatial_dilution_flash_droughts.png`), and saves results to `outputs/stats/spatial_flash.csv`. |

### 2.3 Data Artifacts Produced by Prathmesh

| File | Location | Description |
|---|---|---|
| `cropland_05deg.npy` | `data/raw/derived/` | Total cropland physical area (360×720), all 46 crops summed. 2 MB. |
| `cropland_05deg.npz` | `data/raw/derived/` | Per-crop physical area (46 arrays, each 360×720). 3.4 MB compressed. |
| `crop_climate.parquet` | `data/clean/` | Extracted per-crop climate exposures (`crop_spei12_w`). ~170K rows for 196 countries. |
| `response_paradox.csv` | `outputs/stats/` | 8 rows: all correlations and OLS coefficients for the paradox analysis. |
| `spatial_flash.csv` | `outputs/stats/` | 6 rows: spatial dilution and flash drought correlation results. |

### 2.4 Figures Produced by Prathmesh

| Figure | Location | Description |
|---|---|---|
| `01_weighted_vs_unweighted.png` | `figures/climate/` | 2×3 panel: RUS, AUS, BRA, IND, USA, CHN comparing unweighted vs cropland-weighted SPEI-12 time series. Drought years shaded red. |
| `02_cropland_weights.png` | `figures/climate/` | Global heatmap of MapSPAM cropland area at 0.5° resolution (linear color scale). |
| `02_cropland_weights_lognorm.png` | `figures/climate/` | Same, but with logarithmic color scale — reveals sparse agricultural regions that linear scaling washes out. |
| `03_event_counts_map.png` | `figures/climate/` | Choropleth map: number of drought events per country (1993–2023). |
| `04_response_diversity_paradox.png` | `figures/climate/` | 3-panel scatterplot: (1) resp_div vs vulnerability, (2) resp_div vs max_abs_beta, (3) max_abs_beta vs mean_crop_cv. Displays the three country-level associations. |
| `05_spatial_dilution_flash_droughts.png` | `figures/climate/` | 3-panel: (1) log(area) vs SPEI variance, (2) log(area) vs max crop sensitivity, (3) bar chart comparing spei12_w and spei6min_w predictive power globally vs during severe droughts. |

### 2.5 What Prathmesh Added Beyond the Original Prompt

The climate extensions compare country area with SPEI variance, six-month and annual drought indices with output anomalies, and dispersion of crop drought slopes with vulnerability. The code uses MapSPAM crop footprints for the crop slopes. [Spatial comparisons](outputs/stats/spatial_flash.csv) · [Crop sensitivity](outputs/stats/response_paradox.csv)

## 3. Pranav's Work — Infrastructure & Metrics Engine

### 3.1 Repo Setup (`src/common/`)

| File | Lines | What It Does |
|---|---|---|
| `paths.py` | 42 | Central path definitions. `USE_DUMMY=1` environment variable switches all reads/writes between `data/clean/` ↔ `data/dummy/` and `outputs/` ↔ `outputs/dummy/`. |
| `contract.py` | 184 | One dictionary per contract file mapping column names → expected dtypes. The `validate(df, name)` function asserts correct columns, dtypes, no duplicate keys, and value bounds. This is called by every pipeline script before writing output. |
| `make_dummy.py` | 290 | Generates realistic synthetic versions of ALL 7 contract files (30 fake countries, 1993–2023, 8 crops each, plausible magnitudes, a few drought years). Written to `data/dummy/` and `outputs/dummy/`. Allows all team members to develop independently before real data lands. |
| `sources.py` | 46 | Utilities for downloading files with progress bars and logging sources to `data/raw/SOURCES.md`. |

### 3.2 Metrics Engine (`src/metrics/`)

| File | Lines | What It Does |
|---|---|---|
| `detrend.py` | 31 | LOWESS detrending with `frac=0.5`. Given a raw time series, computes trend, relative anomaly $a_t = (y - \text{trend}) / \text{trend}$, and index $I_t = y / \text{trend}$. |
| `diversity.py` | 15 | Computes Hill number of order 1 (`hill1`): the exponent of Shannon entropy of mean harvested-area shares. |
| `faces.py` | 57 | Implements the four faces of resilience: `stability` (1/CV), `resistance` (median value ratio at drought onset), `recovery` (median rebound ratio), `vulnerability` (share of years below 0.9 index). |
| `response.py` | 47 | `resp_beta`: OLS slope of yield anomaly on that crop class’s SPEI-12. `resp_div`: value-weighted SD of betas across crops. `resp_divergence`: the Ross et al. (2023) divergence formula, with the exact equation quoted in the docstring. |
| `decompose.py` | 26 | Portfolio decomposition following Loreau & de Mazancourt (2008). Computes `phi_sync` (crop synchrony), `mean_crop_cv`, and asserts $CV_\text{total} = \sqrt{\phi_\text{sync}} \times \text{mean\_crop\_cv}$ to $10^{-9}$. |
| `events.py` | 31 | Identifies drought events from `climate.parquet` using the rules in `CLAUDE.md` (threshold ≤ -1.0, 3-year pre/post windows). |
| `run.py` | 185 | The main orchestrator. For each included country: detrends national value, computes all metrics, extracts per-crop yield anomalies and response betas (using observed exposures from the FAOSTAT–MapSPAM item crosswalk). Outputs `outputs/metrics.parquet`, `outputs/crop_anomalies.parquet`, `outputs/metrics_national.parquet`, and `outputs/metrics_diagnostics.csv`. |
| `stats.py` | 120 | Runs all statistical regressions. Spearman with bootstrap 95% CI (2000 resamples), OLS with HC3 robust standard errors, R² comparison tables. Outputs `outputs/stats/{spearman,ols,r2_comparison,summary}.csv`. |
| `figures.py` | 163 | Generates 4 result figures: `01_faces_2x2.png` (hill1 vs each face), `02_count_vs_response.png` (standardized beta coefficient plot), `03_decomposition.png` (portfolio decomposition), `04_maps.png` (world maps of hill1 and resp_div). |
| `robustness.py` | 62 | Runs robustness checks: varying the LOWESS fraction, drought threshold, and SPEI timescale. Outputs `outputs/stats/robustness.csv`. |
| `data.py` | 31 | Helper that loads `metrics.parquet` and `controls.parquet`, joins them into a single analysis table for `stats.py`. |
| `all.py` | 15 | Convenience script that runs the full metrics + stats + figures pipeline in sequence. |
| `METHODS.md` | — | Internal documentation of the metric formulas. |
| `tests/test_metrics.py` | — | Unit tests for detrending, diversity, faces, response, and decomposition on synthetic data. |

### 3.3 Pranav's Data Artifacts

| File | Location | Description |
|---|---|---|
| `metrics.parquet` | `outputs/` | 107 rows × 12 columns. One row per included country with hill1, stability, resistance, recovery, vulnerability, vuln_drought, n_events, resp_div, resp_divergence, phi_sync, mean_crop_cv. |
| `crop_anomalies.parquet` | `outputs/` | 42,338 rows. Per-crop yield anomalies with value_share and resp_beta for each country-year-item. |
| `metrics_national.parquet` | `outputs/` | 3,317 rows. National-level detrended value index, SPEI, and drought flags per country-year. |
| `metrics_diagnostics.csv` | `outputs/` | Diagnostic counts per country (n_years, n_drought_years, n_crops_response, etc.). |
| `spearman.csv` | `outputs/stats/` | Spearman correlations with bootstrap CIs for each face × predictor pair. |
| `ols.csv` | `outputs/stats/` | OLS regression coefficients with HC3 standard errors for each face. |
| `r2_comparison.csv` | `outputs/stats/` | R² comparison: hill1-only vs resp_div-only vs both. |
| `robustness.csv` | `outputs/stats/` | Robustness check results across parameter variations. |
| `summary.csv` | `outputs/stats/` | Sample sizes and median metric values. |
| All `data/dummy/*` files | `data/dummy/` | Synthetic test data for all 7 contract files. |
| All `outputs/dummy/*` files | `outputs/dummy/` | Synthetic analytical results. |

---

## 4. Gursahib's Work — FAOSTAT Crop Data Pipeline (`src/crops/`)

| File | Lines | What It Does |
|---|---|---|
| `download.py` | 57 | Downloads FAOSTAT bulk files (QCL, QV, RL) and restores the pinned World Bank WDI indicators (GDP per capita, fertilizer usage). Logs all URLs to `data/raw/SOURCES.md`. |
| `faostat.py` | 39 | Low-level reader for FAOSTAT bulk CSV zips. Extracts flag metadata and filters by element codes. |
| `countries.py` | 89 | Maps FAOSTAT area codes to ISO3 via the M49 column. Drops regional aggregates. Applies merged entities (BLX, SCG, SDX, CHN). Outputs `data/clean/countries.csv`. |
| `crops.py` | 138 | The main crop processing pipeline. Filters to base-level crops (drops 11 aggregates), computes `yield_t_ha = prod_t / area_ha`, joins constant I$ values from QV, computes missing values as `production × item price` (recovering FAOSTAT's constant prices from published data), tracks imputation flags, sums merged entities. Outputs `data/clean/crops.parquet` and `data/clean/value_computed.csv`. |
| `controls.py` | 66 | Builds `data/clean/controls.parquet` from World Bank WDI (gdp_pc, fert_kg_ha) and FAOSTAT RL (irrig_share). |
| `quality.py` | 136 | Applies inclusion rules: ≥100K ha, ≥5 crops, ≥25 non-imputed years. Detects "suspiciously flat" series (detrended CV < 1% or repeated values with high imputation). Outputs `data/clean/quality.csv` and `data/clean/flat_series.csv`. |
| `figures.py` | 101 | Generates `figures/data/01_imputed_share_map.png` (choropleth of imputation share) and `figures/data/02_example_flat_series.png` (examples of flat vs real series). |
| `README.md` | — | Internal documentation for the crops module. |

### 4.1 Gursahib's Data Artifacts

| File | Location | Description |
|---|---|---|
| `countries.csv` | `data/clean/` | 195 rows. ISO3 codes, names, regions, merged entity membership. |
| `crops.parquet` | `data/clean/` | 267,838 rows. iso3 × year × item_code with area_ha, prod_t, yield_t_ha, value_const, is_imputed. |
| `controls.parquet` | `data/clean/` | 6,045 rows. iso3 × year with gdp_pc, fert_kg_ha, irrig_share. |
| `quality.csv` | `data/clean/` | 195 rows. Per-country: n_crops, share_imputed, included (boolean), reason for exclusion. |
| `flat_series.csv` | `data/clean/` | Flagged suspicious crop series with detrended CV, max identical run, imputation share. |
| `value_computed.csv` | `data/clean/` | Country-years where gross production value was computed as production × item price rather than taken from FAOSTAT directly. |

---

## 5. Pragya's Work — Interactive Explorer & Deck Assets (`src/explorer/`)

| File | Lines | What It Does |
|---|---|---|
| `india_data.py` | — | Loads district metrics, time series, statistical tables, and valid display geometry. Numeric observations retain source precision. |
| `diagram.py` | 83 | Generates the `pipeline.svg` and `pipeline.png` flowcharts (covering Phases 1–4) using a Mermaid string and the Kroki API, and retains existing diagram files on a network error. |
| `build.py` | 1,180 | Builds `explorer/index.html`, a single self-contained interactive Plotly dashboard combining the Global 107-country explorer, the portfolio Simulator, the Phase 4 India District choropleth map & district timeseries (`core` 308 districts and `ext5` 127 districts across 690 polygons), and the interactive Phase 4 statistical results & validation tab. |
| `collect.py` | 76 | Assembles the presentation deck. Cleans out old assets from `figures/deck/`, copies all final PNGs from `climate`, `data`, `results`, `validation`, and `india`, prepends sequential numbers, and builds `figures/deck/INDEX.md`. |
| `__main__.py` | 17 | Orchestrator script that builds the explorer and collects the figures using the bundled diagram. |

### 5.1 Pragya's Data Artifacts & Figures

| File | Location | Description |
|---|---|---|
| `index.html` | `explorer/` | Interactive Plotly.js dashboard mapping global countries (Phases 1–3) and Indian districts (Phase 4) with crop time series, portfolio simulations and statistical tables. |
| `pipeline.{png,svg}` | `figures/deck/` | Visual flowchart of the project's global and Phase 4 India district data architecture. |
| `INDEX.md` | `figures/deck/` | Markdown table cataloging all 30 presentation figures across all project modules. |
| `*.png` | `figures/deck/` | 30 assembled and sequentially numbered PNGs from `explorer`, `climate`, `data`, `results`, `validation`, and `india`. |


---

## 6. Arushi's Work — Validation & Case Studies (`src/validation/`)

| File | Lines | What It Does |
|---|---|---|
| `hit_rate.py` | 177 | Evaluates how accurately our climate indices detect known real-world drought events (from `known_droughts.csv`). Computes hit rates for weighted vs unweighted SPEI. Outputs `outputs/validation/drought_hits.csv` and summary files. |
| `case_studies.py` | 190 | Generates detailed case study plots for three contrasting countries (IND, COL, FIN) to visually validate the metric behaviors. Overlays national index over per-crop yield anomalies. |
| `known_droughts.csv` | - | Curated list of 12 highly documented historical drought events (e.g., USA 2012, AUS 2006). |

### 6.1 Arushi's Data Artifacts & Figures

| File | Location | Description |
|---|---|---|
| `drought_hits_summary.csv` | `outputs/validation/` | Summary of hit rates showing cropland-weighted SPEI-12 detected 6 of 12 selected droughts in their listed year and 7 within one year. |
| `metric_map.md` | `outputs/validation/` | Skeleton bridging project metrics to the literature review. |
| `01_hit_rate.png` | `figures/validation/` | Bar chart comparing the hit rates of the different SPEI indices. |
| `02_case_*.png` | `figures/validation/` | Timeseries plots for the three chosen case study countries (IND, COL, FIN). |

---

## 7. Current data coverage

| Dataset | Coverage |
|---|---|
| National crop table | 195 countries or harmonized units, 157 crop items, 1993–2023 |
| National analysis | 107 countries after quality filtering, 232 usable events |
| Drought event catalogue | 431 events across 187 countries |
| MapSPAM | 46 physical-area crop layers |
| India district geometry | 690 matched polygons |
| Core district model | 308 districts, 1998–2022 |
| Five-crop district model | 127 districts, 1993–2015 |

[National sample](outputs/stats/summary.csv) · [District samples](outputs/stats/india_summary.csv) · [District crosswalk](data/clean/india/district_match.csv)
