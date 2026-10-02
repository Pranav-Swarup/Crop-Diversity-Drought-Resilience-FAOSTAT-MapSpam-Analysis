# Project Metadata: Repository Structure, Ownership & Change Log

This file documents every file and folder in the repository, who wrote it, what it does, what was changed from the original plan, and what new contributions were added beyond the baseline assignments.

---

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
│   ├── validation/            # Case studies (Arushi — NOT IMPLEMENTED)
│   └── explorer/              # Interactive dashboard (Pragya — NOT IMPLEMENTED)
│
├── data/
│   ├── raw/                   # Raw downloaded data (.gitignored)
│   ├── clean/                 # Processed contract files (committed)
│   └── dummy/                 # Synthetic test data (Pranav)
│
├── outputs/                   # Analytical results
│   ├── stats/                 # Regression and correlation CSVs
│   ├── validation/            # (empty — Arushi's placeholder)
│   └── dummy/                 # Dummy analytical results (Pranav)
│
├── figures/
│   ├── climate/               # Climate pipeline figures (Prathmesh)
│   ├── data/                  # Data quality figures (Gursahib)
│   ├── results/               # Metrics figures (Pranav)
│   ├── validation/            # (empty — Arushi's placeholder)
│   └── deck/                  # Assembled presentation deck (Pragya)
│
└── explorer/                  # (empty — Pragya's placeholder)
```

---

## 2. Prathmesh's Work — Climate Exposure Pipeline (`src/climate/`)

### 2.1 Files Already Present Before This Session (Steps 1–5)
These scripts were written as a "first iteration" by the metrics team to bootstrap the pipeline. Prathmesh verified their correctness but did not modify them (except `cropland.py`).

| File | Lines | What It Does |
|---|---|---|
| `download.py` | 50 | Downloads SPEIbase v2.10 netCDFs (SPEI-6, SPEI-12), MapSPAM 2020 zip, and Natural Earth shapefiles. Logs sources to `data/raw/SOURCES.md`. Includes a help message for MapSPAM's manual download (Harvard Dataverse guestbook). |
| `geometry.py` | 46 | Reads Natural Earth 1:10m admin-0 shapefile, maps to ISO3 codes using `ISO_A3_EH` (falling back to `ADM0_A3`), dissolves merged entities (BLX, SCG, SDX). Outputs `data/clean/countries.geojson`. |
| `aggregate.py` | 99 | The core climate processing script. Reads SPEIbase `.nc` files and the cropland `.npy` raster. For each country (via `exactextract`), computes 3 indices: `spei12_w` (cropland-weighted December SPEI-12), `spei12_unw` (area-weighted), and `spei6min_w` (minimum 6-month seasonal SPEI). Outputs `data/clean/climate.parquet`. |
| `events.py` | 23 | Reads `climate.parquet` and identifies drought events per CLAUDE.md rules (≤ -1.0 threshold, 3-year pre-event window, 3-year post-event window). Outputs `data/clean/events.csv`. |

### 2.2 Files Created by Prathmesh This Session

| File | Lines | Status | What It Does |
|---|---|---|---|
| **`figures.py`** | 154 | **NEW** (Step 6) | Generates the 3 required climate figures (`01_weighted_vs_unweighted.png`, `02_cropland_weights.png`, `03_event_counts_map.png`) plus a bonus LogNorm variant (`02_cropland_weights_lognorm.png`). Prints validation metrics: 230 countries, 431 events, top-10 weighted vs. unweighted divergences. |
| **`cropland.py`** | 76 | **MODIFIED** | Originally saved a single flat `.npy` (summed across all 46 crops). Now produces **both**: the legacy `cropland_05deg.npy` (backward compatible with `aggregate.py`) AND a new `cropland_05deg.npz` containing 46 individual crop arrays for crop-specific analyses. The `OUT` alias points to `.npy` so `aggregate.py` requires zero changes. |
| **`analysis/__init__.py`** | 1 | **NEW** | Package init for the analysis submodule. |
| **`analysis/response_paradox.py`** | 153 | **NEW** | Fully reproducible script investigating the Response Diversity Paradox. Computes `max_abs_beta` per country, runs Spearman/Pearson correlations, OLS regression controlling for climate variance, generates a 3-panel scatter figure (`04_response_diversity_paradox.png`), and saves results to `outputs/stats/response_paradox.csv`. |
| **`analysis/spatial_and_flash.py`** | 166 | **NEW** | Fully reproducible script proving Spatial Dilution and Flash Drought findings. Correlates country area with SPEI variance and crop sensitivity, compares `spei12_w` vs `spei6min_w` predictive power globally and during severe droughts, generates a 3-panel figure (`05_spatial_dilution_flash_droughts.png`), and saves results to `outputs/stats/spatial_flash.csv`. |

### 2.3 Data Artifacts Produced by Prathmesh

| File | Location | Description |
|---|---|---|
| `cropland_05deg.npy` | `data/raw/derived/` | Total cropland physical area (360×720), all 46 crops summed. 2 MB. |
| `cropland_05deg.npz` | `data/raw/derived/` | Per-crop physical area (46 arrays, each 360×720). 3.4 MB compressed. |
| `response_paradox.csv` | `outputs/stats/` | 8 rows: all correlations and OLS coefficients for the paradox analysis. |
| `spatial_flash.csv` | `outputs/stats/` | 6 rows: spatial dilution and flash drought correlation results. |

### 2.4 Figures Produced by Prathmesh

| Figure | Location | Description |
|---|---|---|
| `01_weighted_vs_unweighted.png` | `figures/climate/` | 2×3 panel: RUS, AUS, BRA, IND, USA, CHN comparing unweighted vs cropland-weighted SPEI-12 time series. Drought years shaded red. |
| `02_cropland_weights.png` | `figures/climate/` | Global heatmap of MapSPAM cropland area at 0.5° resolution (linear color scale). |
| `02_cropland_weights_lognorm.png` | `figures/climate/` | Same, but with logarithmic color scale — reveals sparse agricultural regions that linear scaling washes out. |
| `03_event_counts_map.png` | `figures/climate/` | Choropleth map: number of drought events per country (1993–2023). |
| `04_response_diversity_paradox.png` | `figures/climate/` | 3-panel scatterplot: (1) resp_div vs vulnerability, (2) resp_div vs max_abs_beta, (3) max_abs_beta vs mean_crop_cv. Resolves the paradox visually. |
| `05_spatial_dilution_flash_droughts.png` | `figures/climate/` | 3-panel: (1) log(area) vs SPEI variance, (2) log(area) vs max crop sensitivity, (3) bar chart comparing spei12_w and spei6min_w predictive power globally vs during severe droughts. |

### 2.5 What Prathmesh Added Beyond the Original Prompt

The original prompt (PROMPTS.md) assigned Steps 1–6 only. Everything below was additional research initiated by Prathmesh:

1. **LogNorm cropland figure** — A second version of the cropland weights map using logarithmic color scaling for better visual clarity.
2. **Per-crop `.npz` archive** — Extended `cropland.py` to preserve individual crop spatial footprints (46 arrays) rather than just the aggregate sum. This enables future crop-specific SPEI exposure analyses.
3. **Response Diversity Paradox investigation** — Designed and ran a multi-part statistical analysis proving that `resp_div` is a proxy for the presence of a single hyper-sensitive crop (`r = 0.91` with `max_abs_beta`), resolving a counter-intuitive finding in the core regressions.
4. **Spatial Dilution analysis** — Proved that geographic size artificially dampens measured drought indices (`r = -0.48` between log area and SPEI variance), identifying a systematic bias in the current pipeline.
5. **Flash Drought comparison** — Demonstrated that the seasonal metric `spei6min_w` is significantly more predictive of yield losses during severe droughts than the annual `spei12_w`.
6. **`report.md`** — Authored the comprehensive 10-section research report documenting all methodology, assumptions, and findings.
7. **`metadata.md`** — This file.

---

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
| `response.py` | 47 | `resp_beta`: OLS slope of yield anomaly on `spei12_w` per crop. `resp_div`: value-weighted SD of betas across crops. `resp_divergence`: the Ross et al. (2023) divergence formula, with the exact equation quoted in the docstring. |
| `decompose.py` | 26 | Portfolio decomposition following Loreau & de Mazancourt (2008). Computes `phi_sync` (crop synchrony), `mean_crop_cv`, and asserts $CV_\text{total} = \sqrt{\phi_\text{sync}} \times \text{mean\_crop\_cv}$ to $10^{-9}$. |
| `events.py` | 31 | Identifies drought events from `climate.parquet` using the rules in `CLAUDE.md` (threshold ≤ -1.0, 3-year pre/post windows). |
| `run.py` | 165 | The main orchestrator. For each included country: detrends national value, computes all metrics, extracts per-crop yield anomalies and response betas. Outputs `outputs/metrics.parquet`, `outputs/crop_anomalies.parquet`, `outputs/metrics_national.parquet`, and `outputs/metrics_diagnostics.csv`. |
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
| `crop_anomalies.parquet` | `outputs/` | ~15,000 rows. Per-crop yield anomalies with value_share and resp_beta for each country-year-item. |
| `metrics_national.parquet` | `outputs/` | ~3,300 rows. National-level detrended value index, SPEI, and drought flags per country-year. |
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
| `download.py` | 57 | Downloads FAOSTAT bulk files (QCL, QV, RL) and World Bank WDI indicators (GDP per capita, fertilizer usage). Logs all URLs to `data/raw/SOURCES.md`. |
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
| `countries.csv` | `data/clean/` | 232 rows. ISO3 codes, names, regions, merged entity membership. |
| `crops.parquet` | `data/clean/` | ~500K rows. iso3 × year × item_code with area_ha, prod_t, yield_t_ha, value_const, is_imputed. |
| `controls.parquet` | `data/clean/` | ~7K rows. iso3 × year with gdp_pc, fert_kg_ha, irrig_share. |
| `quality.csv` | `data/clean/` | 232 rows. Per-country: n_crops, share_imputed, included (boolean), reason for exclusion. |
| `flat_series.csv` | `data/clean/` | Flagged suspicious crop series with detrended CV, max identical run, imputation share. |
| `value_computed.csv` | `data/clean/` | Country-years where gross production value was computed as production × item price rather than taken from FAOSTAT directly. |

---

## 5. Unimplemented Sections

| Module | Owner | Status |
|---|---|---|
| `src/validation/` | Arushi | Contains only `__init__.py`. Case studies, hit rate analysis, and known droughts template are not implemented. |
| `src/explorer/` | Pragya | Contains only `__init__.py`. The interactive HTML dashboard, pipeline diagram generator, and deck assembler are not implemented. |

> **Note:** The `figures/deck/` directory contains assembled PNGs that were generated by a subagent during an earlier session. These include copies of Pranav's and Gursahib's figures, plus a `pipeline.svg` diagram. These were generated outside the normal pipeline and should be treated as preliminary.

---

## 6. Summary Statistics

| Metric | Value |
|---|---|
| Total Python source lines | 2,714 |
| Prathmesh's new code | 550 lines (figures.py + cropland.py modification + 2 analysis scripts) |
| Countries in dataset | 232 (107 included after quality filtering) |
| Crops tracked | 163 base-level items (11 aggregates dropped) |
| MapSPAM crop layers | 46 |
| Global cropland area | 1,274 Mha |
| Drought events identified | 431 across 230 countries |
| Analysis time window | 1993–2023 (31 years) |
