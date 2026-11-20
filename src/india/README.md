# src/india: phase 4, Indian districts (docs/phase4-design.md)

## Official apportioned-data follow-up

Pmjarvis implemented the follow-up on 311 historical ICRISAT/TCI districts. The archived crop and irrigation exports cover 1966–2017 and the farm-price export covers 1966–2016. The main model uses observed 2015 prices. The core and five-crop models below are the presented comparators. [Results and figures](../../future_work/FINDINGS.md#official-indian-district-data) · [Sources and limits](../../future_work/DISTRICT.md)

The extension audits the crop export against a World Bank deposit and compares crop-specific irrigated area with the government land-use series on aligned districts. The data retain conflicting source values and flag uncertain units and boundaries. The release dates visible in the archive do not establish the current live edition. [Boundary and unit rules](../../future_work/DISTRICT.md#what-can-be-compared-across-district-boundaries) · [Access record](../../future_work/results/district_status.json)

The DES core covers 308 eligible districts from 1998–2022, with 6,160 observations in the lagged-diversity model. The separate five-crop analysis covers 127 districts from 1993–2015, with 2,286 model observations. The explorer displays each sample on a separate map. [Sample counts](../../outputs/stats/india_summary.csv) · [Panel estimates](../../outputs/stats/india_panel.csv)

## Run order
```
python -m src.india.download_docs   # ICRISAT DLD documentation (official Dataverse record; the record has no data files)
python -m src.india.download        # ICRISAT mirror on Mendeley (5 crops, 1990-2015), THIRD-PARTY
python -m src.india.download_apy    # DES district APY (Ministry of Agriculture) via the India Data Portal CKAN dev instance
python -m src.india.download_geo    # geoBoundaries India ADM1 (2011), ADM2 (2021), SPEI-4 and SPEI-12
python -m src.india.clean           # DES -> data/clean/india/{districts.csv,crops.parquet}
python -m src.india.geo             # district name -> polygon match, districts.geojson
python -m src.india.climate         # district SPEI-12 (December) + kharif SPEI-4 (September), climate.parquet
python -m src.india.extended        # mirror -> 5-crop table 1993-2015 + irrigation/fertiliser controls
python -m src.india.panel           # faces, panel, severity, tables, figures
python -m src.india.conley          # robustness: Conley spatial-HAC standard errors -> india_conley.csv
python -m src.india.kharif          # robustness: kharif drought variable (SPEI-4 of September) -> india_kharif.csv
python -m src.india.kharif_crops    # seasonal crop totals and strict kharif sensitivity
python -m pytest tests/test_india.py   # unit tests (synthetic data, no downloads)
```
District climate aggregation requires both SPEI files and the MapSPAM cropland grid. From the checked-in district tables, start at `src.india.panel` to reproduce the models.

## Sources and why they differ from the design
| Design | Used | Why |
|---|---|---|
| ICRISAT DLD apportioned (1966 boundaries), all crops, prices, irrigation by crop | DES district APY 1998-99 to 2022-23 (1997-98 dropped: thin first-year coverage), unapportioned | `data.icrisat.org/dld` unreachable on 2026-11-14 **[team]** |
| 1993 start | core 1998-2022; 5-crop window 1993-2015 from the Mendeley mirror, reported separately | DES starts in 1997 **[team]** |
| district polygons from ICRISAT | geoBoundaries ADM2 2021 (ODbL), matched by name, aliases for known renames | ICRISAT ships none **[team]** |
| farm harvest prices | constant FAOSTAT India price per crop (`value_const / prod_t`, exact per item) | DES has no prices |
| irrigated area by crop | district gross irrigated / gross cropped area from the mirror, years <= 2015, shares outside [0, 1] excluded | not in DES |

The archived version-one mirror contains 560 district codes. Its 23,739,904-byte workbook has SHA-256 `e972b837b29298b8f06cda6fe7ea982d90a7a97a66e9a07fc72df3b66680318d`. Running `src.india.extended` on that workbook reproduces the 39,954 clean crop rows and 10,620 district-year controls. [Publisher record](https://doi.org/10.17632/ywp3y5j9vv.1)

## Implemented rules
- Year = start year of the agricultural year (1998 = 1998-99), matched to December SPEI-12 of that year (covers the monsoon).
- Season: use the "Total" row if present, else sum seasons. Fixed crop list: 47 crops reported in every year of the window, no "Other ...".
- Diversity uses crops with reported areas in every year of the relevant full or lagged window. Unreported crop areas remain missing.
- Coconut production is in nuts (blanked, area kept). Yields above 10x the crop's national median are blanked as source errors.
- Cotton, coconut, turmeric, cardamom, garlic, horse gram, khesari, mixed pulses, sannhamp have no clean FAOSTAT price: in diversity, not in the value index.
- Rice tonnes are retained as published. The output value uses the paddy price divided by the 0.67 milling outturn. Prices are constant 2014–2016 international dollars per tonne. [NABARD, Table 8](https://www.nabard.org/auth/writereaddata/File/Water%20Productivity%20Mapping%20of%20Major%20Indian%20Crops.pdf)
- Stable district: data in all 25 years. `split_flag`: a parent that kept its code but lost territory, detected relative to the state median.
- Value index uses crops with complete series in the district; a district needs >= 3 such crops holding >= 70% of its priced value.

## Robustness checks (added after the core run; all with `spei12_w` unless stated)
- **Conley standard errors** (`conley.py`, `outputs/stats/india_conley.csv`): Bartlett kernel on centroid distance (100, 200, 400 km), serial lag 5 years,
  next to district and state clustering. District and state cluster counts are reported separately from the number of districts.
- **Kharif drought variable** (`kharif.py`, `india_kharif.csv`, `india_kharif_perm.csv`): SPEI-4 of September (June-September), cropland weighted. Same districts
  and design; district totals still contain rabi crops, so this changes the exposure measure only, not the crop set.
- **Kharif-only crops** (`kharif_crops.py`, `india_kharif_crops*.csv`): table from DES monsoon-season rows only (Kharif + Autumn + Winter; a strict Kharif-only
  variant), no Total/Rabi/Summer/Whole-Year rows, 27 (strict: 24) crops. It covers 53-62% (strict 42-54%) of the core table's area and only states that publish
  a season split (17 states in the usable sample), so it is a different, smaller sample than the core, not a subset to compare one-to-one.
- The validation table records source-backed monsoon rainfall departures for 2002, 2004, 2009, 2014 and 2015. Other years have unknown documentation status. Source URLs accompany the observations in [the validation table](../../outputs/validation/india_drought_years.csv).
- **Tests** (`tests/test_india.py`): hill1_tv timing, balanced-crop rule, season rule, split flag, name matching, `fast_t` against `fe_fit`, Conley against the
  heteroskedasticity-robust variance, distances, citation table.
- `fe_basis` uses an SVD basis because district and state-year dummies are collinear; the first version used a plain QR, which over-projected
  and changed the studentised permutation p-values (fixed, tests added).

## Not done
- Closing 2050 CMIP6 scenario (design section 6.5): needs a separate precipitation and PET download and a team decision.
