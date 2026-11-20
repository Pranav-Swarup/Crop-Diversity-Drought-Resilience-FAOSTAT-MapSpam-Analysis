# Research Report: Crop Diversity & Drought Resilience

## Post-presentation studies by Pmjarvis

The original report describes the delivered presentation. Pmjarvis’s separate studies compare that baseline with planting-to-harvest drought, hourly ERA5 heat and satellite vegetation, a 1961–2023 extreme-drought panel, and historical Indian district estimates using ICRISAT farm prices and crop-wise irrigation. [Follow-up findings and figure index](future_work/FINDINGS.md) · [Methods and source access](future_work/README.md)

- **Growing-season drought.** On 3,317 country-years, the presented definition classified 626 as dry. December drought on the matched crop basket classified 622, and the planting-to-harvest definition classified 524. At mean crop diversity the estimated output difference changed from −4.41% of trend under the presented definition to −5.48% under the seasonal definition. [Figure and values](future_work/results/figure_01_drought.csv)
- **Daily heat and crop stress.** Among the same 2,771 country-years in 99 countries, the presented heat and drought definitions identified 150 jointly hot and dry years. Seasonal ERA5 heat above 30°C and seasonal drought identified 194. Estimated crop output was 5.68% of trend lower under the presented classification and 8.48% lower under the seasonal classification, each relative to years meeting neither threshold. Satellite greenness was 2.64 Vegetation Condition Index points lower in the new compound group. [Figure and values](future_work/results/figure_03_heat.csv)
- **Power for extreme droughts.** Across the same 75 countries, adding 1961–1992 increased extreme observations from 30 to 49. The approximate effect detectable with 80% power fell from 7.50 to 4.89 percentage points of output trend per standard deviation of crop diversity. [Figure and values](future_work/results/figure_04_extremes.csv)

- **Better district data.** The 1966–2017 archived model includes 176 districts and 8,272 district-years. The interaction between moisture balance and prior diversity is −0.67 percentage points of output trend (95% interval −2.20 to +0.86). The later 1998–2017 fit gives +1.46 (−0.08 to +3.00). Both intervals include zero. [District model comparison and source checks](future_work/FINDINGS.md#official-indian-district-data)


**FAOSTAT × MapSPAM × SPEIbase Analysis (1993–2023)**

This document is the comprehensive record of the methodology, data pipelines, assumptions, reproducible analyses, and statistical findings for evaluating whether greater crop diversity makes national agricultural production more resilient to climatic variability.

---

## 1. Analysis Window & Geographical Standardization

*   **Time Window:** 1993–2023. Selected because it begins after the major geopolitical splits of the early 1990s (USSR, Czechoslovakia, Yugoslavia, Ethiopia PDR) and ends at the boundary of the SPEIbase v2.10 dataset (December 2023).
*   **Entity Merging:** To maintain continuous statistical series, certain countries were analytically merged:
    *   `BLX`: Belgium + Luxembourg
    *   `SCG`: Serbia + Montenegro (including Kosovo)
    *   `SDX`: Sudan (former) merged with South Sudan post-2011.
    *   `CHN`: Mainland China. Taiwan is tracked separately (`TWN`).
*   **Inclusion Rules:** A country is only included in the final analysis if it averages $\ge 100{,}000$ hectares of harvested area, grows $\ge 5$ distinct crops, and has $\ge 25$ non-imputed years of national production data. After filtering, **107 countries** are included.

---

## 2. Data Sources & Processing Pipelines

### 2.1 FAOSTAT (Agricultural Production & Economics)
*   **Datasets Used:** QCL (Crops & Livestock), QV (Value of Production), RL (Land Use).
*   **Pipeline:** `src/crops/crops.py`
*   **Filtering:** We strictly filter for base-level crops (those with an "Area harvested" element) and explicitly remove 11 aggregate categories (e.g., "Cereals, primary") to avoid double-counting diversity.
*   **Yield Calculation:** Yield is manually recalculated as `Production (tonnes) / Area (hectares)`.
*   **Value Standardization:** Because raw tonnage cannot be summed across different crops (e.g., tonnes of apples vs. tonnes of wheat), we calculate the portfolio's **Gross Production Value in constant 2014-2016 International Dollars (I$)**. Where FAOSTAT does not publish a value, we compute it as `production × item price`, where item price is recovered as the median of `value / production` across all published country-years for that item. This fill reproduces published values to within 0.01% of their total.
*   **Imputation tracking.** The crop table flags estimated or imputed area or production. The national inclusion screen uses the value share of flagged rows. Crop-response fitting uses non-imputed observations. The quality pipeline also lists flat series for review.

### 2.2 MapSPAM 2020 v2 (Spatial Cropland Weights)
*   **Purpose:** To solve the problem of spatial averaging. We cannot average climate indices across entire countries (e.g., including barren deserts), so we use MapSPAM as a geospatial mask.
*   **Pipeline:** `src/climate/cropland.py`
*   **Processing:** We extract the global physical area (in hectares) for all technologies (`_A_` layers) across 46 crops at a 5-arcminute resolution. We downsample each crop's raster to a 0.5° grid (matching the SPEI grid) and produce two outputs:
    *   `cropland_05deg.npy` — the total cropland footprint (sum of all 46 crops), shape `(360, 720)`. Used by `aggregate.py` for country-level climate weighting.
    *   `cropland_05deg.npz` — per-crop arrays, each `(360, 720)`. Supplies the crop-class exposures used for drought-response slopes.
*   **Validation:** Global physical crop area = **1,274 Mha**; cells with cropland = **38,162** of 259,200.

### 2.3 SPEIbase v2.10 (Climate & Drought Exposure)
*   **Index:** We use the Standardized Precipitation Evapotranspiration Index (SPEI), which measures precipitation minus potential evapotranspiration.
*   **Pipeline:** `src/climate/aggregate.py`
*   **Three indices computed per country-year:**
    *   `spei12_w` — Cropland-weighted mean of December SPEI-12 (standardized 12-month accumulated water balance).
    *   `spei12_unw` — Unweighted mean (by cell area only, i.e., cos-latitude weighting).
    *   `spei6min_w` — Minimum over the 12 months of cropland-weighted SPEI-6. This measures the driest six-month accumulation in each year.
*   **Defining a Drought Event:**
    *   An index of `0` is normal. `≤ -1.0` is a moderate drought, `≤ -1.5` is severe, and `≤ -2.0` is extreme.
    *   An **Event** is defined as the first year of a consecutive run of drought years (where December `spei12_w ≤ -1.0`).
    *   To accurately measure agricultural impact, an event must be preceded by at least 3 normal (drought-free) years, and followed by 3 years inside the analysis window.
*   **Coverage:** The catalogue contains 431 events in 187 countries or harmonized units.

### 2.4 Control Variables
To ensure our regressions are statistically robust, we join:
*   **GDP per capita** (World Bank WDI)
*   **Fertilizer usage** in kg/ha (World Bank WDI)
*   **Irrigation Share**, calculated as Land area equipped for irrigation divided by total Cropland (FAOSTAT RL).

---

## 3. Resilience & Diversity Metrics

All metrics are evaluated against a detrended national value index $I_t$ (using LOWESS smoothing with `frac=0.5` over the 1993-2023 series). Pipeline: `src/metrics/run.py`.

### 3.1 Diversity Metrics
*   **Crop Diversity (`hill1`):** The effective number of crops, calculated as the exponent of the Shannon entropy of the mean harvested-area shares over the time window.
*   **Response Diversity (`resp_div`):** Value-weighted standard deviation of crop-specific drought sensitivities (OLS slopes of yield anomaly on the crop’s MapSPAM-weighted December SPEI-12).
*   **Response Divergence (`resp_divergence`):** Ross et al. (2023) directional divergence metric: `(max β - min β - ||max β| - |min β||) / (max β - min β)`. Equals 0 when all slopes share the same sign, 1 when they are symmetric.

### 3.2 The Four Faces of Resilience (Urruty et al. 2016)
*   **Stability:** $1 / CV$ (Coefficient of Variation) of the detrended residuals of national value.
*   **Resistance:** Evaluated strictly during drought events. The median ratio of the value index during the drought year to the mean value of the 3 pre-drought years.
*   **Recovery:** Evaluated post-drought. The median ratio of the mean value over the 3 post-drought years to the value during the drought year.
*   **Vulnerability:** The share of years where the national value index drops below 0.9 (a loss > 10% below the trendline).

### 3.3 Portfolio Decomposition (Loreau & de Mazancourt 2008)
*   **Synchrony (`phi_sync`):** Measures how correlated crop value fluctuations are within a country.
*   **Mean Crop CV (`mean_crop_cv`):** Average coefficient of variation across individual crops.

---

## 4. Key Assumptions & Constraints

*   **Spatial Stationarity:** We use the MapSPAM 2020 v2 dataset to represent the physical distribution of crops over the entire 1993-2023 window. We assume the macro-geographic footprint of a country's cropland does not drastically migrate during this period.
*   **Constant Valuation:** The 2014-2016 constant FAOSTAT prices are assumed to hold relative value correctly across the entire window, ignoring severe market shocks that might temporarily alter a crop's value relative to others.
*   LOWESS provides an estimated trend. Smoothing can absorb both long-run changes and short-run variation.
*   Country-level events use the total cropland footprint. Crop-response slopes use the mapped crop class or a physical-area-weighted combination of classes for coffee and millet. A missing crop exposure produces a missing slope.

## 5. National results

Among 101 countries with complete controls, a one-standard-deviation increase in harvested-area diversity is associated with 0.399 standard deviations higher stability (HC3 p = 0.00544) and 0.423 standard deviations lower all-year vulnerability (p = 7.69 × 10⁻⁸). The resistance and recovery models each use 96 countries. Their diversity coefficients are +0.105 (p = 0.271) and −0.238 (p = 0.144). Higher irrigation share is associated with lower all-year vulnerability (coefficient −0.205, p = 0.0470). The controls use the World Bank release of 11 November 2026. [Adjusted national estimates](outputs/stats/ols.csv)

The country and year fixed-effects model estimates a SPEI × standardized Hill-1 interaction of −0.007619 (p = 0.02981, 3,286 country-years, 106 countries). The outcome is the national value anomaly. The holdout analysis uses 1993–2012 for trend fitting, crop eligibility and value shares. Against the zero-anomaly baseline, median crop skill in 2013–2023 is −0.01505 across 1,269 crop tests. The training and test slopes have Spearman ρ = 0.232, and 57.1% keep their sign. Across 263 holdout drought country-years the correlation between predicted and observed national anomalies is −0.079. The evaluation uses observed holdout weather. [Panel estimates](outputs/stats/panel_fe.csv) · [Holdout results](outputs/stats/oos_summary.csv)

## 6. Spatial exposure and severity

Across 107 countries, country log area correlates with the variance of cropland-weighted SPEI at −0.4841 (p = 1.27 × 10⁻⁷). Its correlation with the largest absolute crop slope is +0.1853 (p = 0.0560). MapSPAM describes 2020 crop locations throughout the analysis period. In the severity model the extreme-drought × Hill-1 coefficient is +0.0420 (clustered p = 0.107). The exposure and supply results include country-level crop classes, correlated climate exposure, and domestic and imported food sources. [Spatial results](outputs/stats/spatial_flash.csv) · [Severity](outputs/stats/severity_panel.csv) · [Exposure](outputs/exposure_diversity.parquet) · [Supply](outputs/supply_diversity.parquet)

## 7. Seasonal and annual drought measures

In the 240 country-years with annual SPEI below −1.5, the correlations with national value anomaly are +0.235 for the minimum six-month SPEI and +0.172 for annual SPEI. Across all 3,317 country-years the correlations are +0.133 and +0.142, respectively. Neither comparison measures flowering-stage damage. [Seasonal and annual estimates](outputs/stats/spatial_flash.csv)

## 8. Crop response diversity

Across 106 countries, the weighted spread of crop drought slopes has a rank correlation of +0.513 with vulnerability and −0.472 with stability. Its Pearson correlation with the largest absolute crop slope is +0.890. These estimates describe variation across crops and countries. Identifying which crop drives a national loss would require estimating its contribution to the country's value change in that year. [Response-diversity estimates](outputs/stats/response_paradox.csv)

## 9. Historical validation

For 12 selected country events, cropland-weighted SPEI-12 detected 6 in the listed year and 7 within one year. Area-weighted SPEI-12 detected 5 in the listed year, and the stricter event catalogue contained 4. The denominators describe these selected cases. [Validation table](outputs/validation/drought_hits_summary.csv)

## 10. Figures Generated

The following climate figures are in `figures/climate/`. The numbered collection also includes national results, district results and validation figures. [Complete figure index](figures/deck/INDEX.md)

| Figure | Description |
|---|---|
| `01_weighted_vs_unweighted.png` | Time-series for 6 key countries comparing unweighted SPEI-12 vs cropland-weighted SPEI-12 |
| `02_cropland_weights.png` | Global cropland heatmap (linear color scale) |
| `02_cropland_weights_lognorm.png` | Global cropland heatmap (logarithmic color scale) |
| `03_event_counts_map.png` | Choropleth of drought event counts per country (1993-2023) |
| `04_response_diversity_paradox.png` | Crop slope dispersion, vulnerability and crop volatility |
| `05_spatial_dilution_flash_droughts.png` | 3-panel analysis of country area and seasonal versus annual drought correlations |

---

## 11. Indian districts and the explorer

The core 1998–2022 district model uses 308 districts and 6,160 district-years after lagging the crop mix by five years. The SPEI × standardized lagged Hill-1 coefficient is +0.003390 (95% CI −0.006915 to +0.013694, p = 0.518). The five-crop 1993–2015 model uses 127 districts and 2,286 district-years. Its interaction is +0.008485 (95% CI −0.009912 to +0.026882, p = 0.363). Both intervals include zero. The core sample has 460 usable events in 274 districts. The five-crop sample has 174 in 123 districts. The model coefficients use current district matches, balanced area windows and paddy-equivalent valuation for milled rice. [District estimates](outputs/stats/india_panel.csv) · [Sample and events](outputs/stats/india_summary.csv) · [District sources and methods](src/india/README.md)

The standalone explorer includes national and district maps, crop time series, portfolio simulations, model results and source tables. The real and synthetic builds carry separate labels. Missing values remain gaps in lines and appear as unavailable in tables. The map geometry is simplified for display. [Explorer](explorer/index.html) · [Numbered figures](figures/deck/INDEX.md)

## 12. Reproducibility

The full run order and input requirements are in the [README](README.md). The following commands rebuild the climate inputs, selected analyses and explorer.

```bash
source .venv/bin/activate

# Core pipeline (requires raw data in data/raw/)
python -m src.climate.cropland       # → data/raw/derived/cropland_05deg.{npy,npz}
python -m src.climate.aggregate --per-crop # → data/clean/climate.parquet, crop_climate.parquet
python -m src.climate.events         # → data/clean/events.csv
python -m src.climate.figures        # → figures/climate/01-03*.png

# Extended analyses (runs on committed data/clean/ and outputs/)
python -m src.climate.analysis.response_paradox      # → outputs/stats/response_paradox.csv, figures/climate/04*.png
python -m src.climate.analysis.spatial_and_flash      # → outputs/stats/spatial_flash.csv, figures/climate/05*.png

# Phase 4 India district analysis
python -m src.india.panel            # → outputs/india/*.parquet, outputs/stats/india_*.csv, figures/india/*.png
python -m src.india.conley           # → outputs/stats/india_conley.csv
python -m src.india.kharif
python -m src.india.kharif_crops

# Published case studies and the offline explorer
python -m src.validation.case_studies --high COL --low FIN
python -m src.explorer               # → explorer/index.html, figures/deck/*
```

The presented analyses save their statistical tables in `outputs/stats/`. Follow-up figures and statistical tables are indexed in the [post-presentation findings](future_work/FINDINGS.md).

