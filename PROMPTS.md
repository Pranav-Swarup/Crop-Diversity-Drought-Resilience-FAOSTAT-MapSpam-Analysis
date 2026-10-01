# Claude Code prompts, one per person

Put `CLAUDE.md` at the repo root first; Claude Code reads it automatically. Paste your prompt into Claude Code from the repo root. Start in plan mode (review the plan before it writes code).

---

## Pranav — 0. Repo setup (Day 1 morning, before anyone else starts)

```
Read CLAUDE.md. Scaffold the repo exactly as it describes:
- folders: src/{common,crops,climate,metrics,validation,explorer}, data/{raw,clean}, outputs/{stats,validation}, figures/{data,climate,results,validation,deck}, explorer/
- __init__.py in every src package, .gitignore (data/raw/, .venv, __pycache__, .ipynb_checkpoints), requirements.txt with the listed packages
- src/common/contract.py: one dict per contract file mapping column -> dtype, plus a validate(df, name) function that asserts columns, dtypes, no duplicate keys (iso3+year, or iso3+year+item_code where relevant)
- src/common/make_dummy.py: writes realistic dummy versions of EVERY contract file (30 fake countries, 1993–2023, 8 crops each, plausible magnitudes, a few drought years), written to data/dummy/ and outputs/dummy/ (never to the real paths)
- src/common/paths.py with a USE_DUMMY env flag so every module can switch between dummy and real inputs
- a README with the run order
Run make_dummy and validate every dummy file. Commit.
```

## Pranav — 1. Metrics and results engine (Day 1–3)

```
Read CLAUDE.md, especially "Fixed analysis choices" and "Metric definitions". You own src/metrics/ and the outputs listed for Pranav.

Build, as small tested functions in src/metrics/:
1. detrend.py: LOWESS trend (frac=0.5) per series -> trend, relative anomaly a_t, index I_t. Unit tests on synthetic series (linear trend + noise recovers ~zero-mean anomalies).
2. diversity.py: hill1 from mean harvested-area shares.
3. faces.py: stability, resistance, recovery, vulnerability, vuln_drought, n_events, using events.csv and the event rules in CLAUDE.md. Unit-test resistance/recovery on a hand-made series with a known 30% drop and known rebound.
4. response.py: per-crop resp_beta (OLS of yield anomaly on spei12_w, with the inclusion thresholds), resp_div (value-weighted SD), resp_divergence (implement the divergence metric from Ross et al. 2023, Methods in Ecology and Evolution, "How to measure response diversity"; quote the formula you used in a docstring and flag if you are unsure of it).
5. decompose.py: phi_sync, mean_crop_cv, and assert CV_total == sqrt(phi_sync) * mean_crop_cv to 1e-9.
6. run.py: builds outputs/metrics.parquet and outputs/crop_anomalies.parquet for included countries only (quality.csv), validates against the contract.
7. stats.py -> outputs/stats/:
   - Spearman rho with bootstrap 95% CI (2000 resamples) of each face vs hill1 and vs resp_div
   - OLS per face: face ~ z(hill1) + z(resp_div) + z(log gdp_pc) + z(fert_kg_ha) + z(irrig_share), country-level window means of controls, HC3 robust SEs, report n
   - a small table comparing R² of hill1-only vs resp_div-only vs both
8. figures.py -> figures/results/:
   - 01_faces_2x2.png: hill1 vs each of the four faces, points coloured by region, Spearman rho + CI in each panel
   - 02_count_vs_response.png: coefficient plot (standardised betas + CIs) of hill1 vs resp_div for each face
   - 03_decomposition.png: for ~15 countries spanning the diversity range, stacked contribution of log(1/mean_crop_cv) and -0.5*log(phi_sync) to log(stability)
   - 04_maps.png: world maps of hill1 and resp_div using data/clean/countries.geojson

Develop everything against dummy data first (USE_DUMMY=1), then switch to real inputs when crops.parquet and climate.parquet land. At the end print a plain summary: n countries, n events, the headline correlations. Do not write interpretation text.
```

---

## Gursahib — Crop data pipeline and data quality (Day 1, quality audit Day 2)

```
Read CLAUDE.md. You own src/crops/, data/clean/{countries.csv, crops.parquet, controls.parquet, quality.csv}, figures/data/.

1. download.py: download the FAOSTAT bulk "normalized" files for Crops and livestock products (QCL), Value of Agricultural Production (QV), and Land Use (RL). Find the current bulk URLs from the FAOSTAT site (bulk downloads are under bulks-faostat.fao.org or fenixservices.fao.org/faostat/static/bulkdownloads); if neither works, STOP and ask me. Also pull World Bank WDI NY.GDP.PCAP.KD and AG.CON.FERT.ZS via the WDI API. Log every URL in data/raw/SOURCES.md.
2. countries.py -> countries.csv: map FAOSTAT areas to iso3 via the M49 column. Drop regional aggregates. Apply the merged entities in CLAUDE.md (BLX, SCG, SDX; China mainland -> CHN, drop "China" aggregate). Print any other area whose series starts or stops inside 1993–2023 and ask me before deciding.
3. crops.py -> crops.parquet:
   - keep only items that have an "Area harvested" element (drops livestock); drop aggregate items (e.g. "Cereals, primary", "... Total"; check the item list, do not rely on code ranges alone)
   - compute yield_t_ha = prod_t / area_ha yourself
   - join value_const = Gross Production Value, constant 2014-2016 thousand I$ (convert to I$)
   - is_imputed from flags E and I (confirm meanings from the flags file in the zip)
   - sum merged entities correctly (a merged country-year is imputed if any part is)
4. controls.py -> controls.parquet: gdp_pc, fert_kg_ha from WDI; irrig_share = "Land area equipped for irrigation" / "Cropland" from FAOSTAT RL.
5. quality.py -> quality.csv: per country n_crops, share_imputed (of value), included + reason, applying the inclusion rules in CLAUDE.md. Also find "suspiciously flat" series (crop series whose detrended CV is < 1% or which repeat the same value 3+ years running) and list them; flat series fake stability, so exclude them and record why.
6. Figures in figures/data/: 01_imputed_share_map.png, 02_example_flat_series.png (3 real examples of imputed flat series vs a real reported one).
Validate every file against src/common/contract.py. Print: countries included/excluded, crops per country (median), % value imputed overall.
```

---

## Prathmesh — Climate exposure pipeline (Day 1–2)

```
Read CLAUDE.md. You own src/climate/, data/clean/{climate.parquet, events.csv, countries.geojson}, figures/climate/.

1. download.py:
   - SPEIbase v2.10 global 0.5° netCDF, SPEI-6 and SPEI-12 (CSIC; https://spei.csic.es/database.html, DOI 10.20350/digitalCSIC/16497). If the direct file URL cannot be resolved, STOP and ask me to download manually into data/raw/spei/.
   - MapSPAM 2020 v2 global physical area GeoTIFFs, all-technologies layer for every crop (Harvard Dataverse DOI 10.7910/DVN/SWPENT). Read the ReadMe first and confirm the file naming.
   - Natural Earth 1:10m admin-0 countries.
   Log everything in data/raw/SOURCES.md.
2. geometry.py -> countries.geojson: use ISO_A3_EH (fall back to ADM0_A3 where it is -99); dissolve into our merged codes using member_iso3 from data/clean/countries.csv (if that file isn't there yet, use the merged entities listed in CLAUDE.md). Property: iso3.
3. cropland.py: sum MapSPAM per-crop physical area to one all-crops raster, then aggregate from 5 arcmin to the SPEI 0.5° grid by summation (6x6 blocks), aligned exactly to the SPEI grid. Assert total global cropland is in a plausible range and print it.
4. aggregate.py -> climate.parquet, 1993–2023:
   - spei12_w: cropland-weighted mean of SPEI-12 at December, per country-year (exactextract weighted mean with cropland as weights)
   - spei12_unw: same but weighted by cell area only (cos latitude)
   - spei6min_w: minimum over the 12 months of cropland-weighted SPEI-6
   Handle NaN cells (ocean, no-data) without dropping whole countries; report countries with < 90% valid cropland coverage.
5. events.py -> events.csv: event years per CLAUDE.md rules, with severity classes.
6. Figures in figures/climate/:
   - 01_weighted_vs_unweighted.png: spei12_w vs spei12_unw time series for RUS, AUS, BRA, IND, USA, CHN (6 panels), with drought years marked
   - 02_cropland_weights.png: global cropland raster used as weights
   - 03_event_counts_map.png: number of drought events per country, 1993–2023
Validate against the contract. Print: countries covered, total events, mean |spei12_w - spei12_unw| by country (top 10).
```

---

## Arushi — Validation and case studies (Day 2–3; Day 1: build on dummy data)

```
Read CLAUDE.md. You own src/validation/, outputs/validation/, figures/validation/. Import all metric functions from src/metrics/ (never reimplement them). Use USE_DUMMY=1 until the real files exist.

1. known_droughts.csv (I will fill in and check sources myself): create the template with columns iso3, year, event_name, source_citation, with these rows to be verified: IND 2002, IND 2009, USA 2012, AUS 2006, AUS 2007, ETH 2015, KEN 2011, SOM 2011, FRA 2003, DEU 2018, ZAF 2015, BRA 2015. Do not invent citations; leave source_citation blank for me.
2. hit_rate.py -> outputs/validation/drought_hits.csv: for each known event, whether our catalogue flags it (spei12_w <= -1.0) and whether spei6min_w <= -1.5 catches it; also compare against spei12_unw. Summary hit rate for weighted vs unweighted.
3. case_studies.py: for IND plus two contrast countries (pick one high-hill1 and one low-hill1 country with at least 3 drought events and similar event counts; print candidates and let me choose):
   - per-crop yield anomaly lines for the top 6 crops by value share, drought years shaded, national value index overlaid
   - a table of each crop's resp_beta
4. Figures in figures/validation/: 01_hit_rate.png (weighted vs unweighted), 02_case_<iso3>.png for each case study.
5. metric_map.md: a table skeleton with columns Metric | Definition (from CLAUDE.md) | Concept source | Paper in our Review 1 it relates to. Fill only the first two columns; I will fill the rest.
```

---

## Pragya — Interactive explorer and deck assets (Day 1–4)

```
Read CLAUDE.md. You own src/explorer/, explorer/, figures/deck/. Build against dummy data (USE_DUMMY=1) from Day 1 and switch to real files when they exist.

1. src/explorer/build.py -> explorer/index.html: ONE self-contained HTML file, Plotly.js loaded from a CDN, all data embedded as JSON (keep file < 15 MB).
   - left: world choropleth from data/clean/countries.geojson, dropdown to switch metric (hill1, resp_div, stability, resistance, recovery, vulnerability, phi_sync), colour-blind-safe scale, hover shows country + value + n_events
   - right, on country click: (a) top-6 crop yield anomaly lines from outputs/crop_anomalies.parquet, (b) spei12_w bars from climate.parquet underneath on a shared x-axis, drought event years shaded across both
   - a small "About the data" panel listing the sources from data/raw/SOURCES.md
   - works offline except for the CDN, readable on a laptop projector (min font 14px)
2. src/explorer/diagram.py -> figures/deck/pipeline.svg: a Mermaid (or graphviz) pipeline diagram: FAOSTAT / SPEIbase / MapSPAM / WDI -> cleaning -> country-year tables -> metrics -> results & explorer. Render to SVG and PNG.
3. src/explorer/collect.py: copies every final PNG from figures/*/ into figures/deck/ with a numbered index file (figures/deck/INDEX.md: filename, owner, one-line factual description).
Re-run build.py whenever outputs change. Print file size and the number of countries in the explorer.
```
