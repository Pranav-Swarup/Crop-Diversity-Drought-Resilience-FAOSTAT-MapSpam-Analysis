# src/india: phase 4, Indian districts (docs/phase4-design.md)

Unreviewed. Not agreed by the team: every source switch and rule marked **[team]** below. Nothing in "Fixed analysis choices" of `CLAUDE.md` was changed; the national analysis is untouched.

## Run order
```
python -m src.india.download_docs   # ICRISAT DLD documentation (official Dataverse record; the record has no data files)
python -m src.india.download        # ICRISAT mirror on Mendeley (5 crops, 1990-2015), THIRD-PARTY
python -m src.india.download_apy    # DES district APY (Ministry of Agriculture) via the India Data Portal CKAN dev instance
python -m src.india.download_geo    # geoBoundaries India ADM1/ADM2 (2021) + SPEI-12
python -m src.india.clean           # DES -> data/clean/india/{districts.csv,crops.parquet}
python -m src.india.geo             # district name -> polygon match, districts.geojson
python -m src.india.climate         # district SPEI-12 (December) + kharif SPEI-4 (September), climate.parquet
python -m src.india.extended        # mirror -> 5-crop table 1993-2015 + irrigation/fertiliser controls
python -m src.india.panel           # faces, panel, severity, tables, figures
python -m src.india.conley          # robustness: Conley spatial-HAC standard errors -> india_conley.csv
python -m src.india.kharif          # robustness: kharif drought variable (SPEI-4 of September) -> india_kharif.csv
python -m pytest tests/test_india.py   # unit tests (synthetic data, no downloads)
```
Needs `data/raw/spei/spei12.nc` and MapSPAM (`src.climate.download` after the manual Dataverse guestbook download, then
`src.climate.cropland`) for the design's drought variable `spei12_w`; without the cropland raster `src.india.climate` leaves `spei12_w`
empty and the analysis falls back to the area-weighted `spei12_unw` (the label is in every output). The kharif variant needs
`data/raw/spei/spei04.nc` (SPEIbase v2.10, CSIC digital repository, sequence 4 of handle 10261/364137; SPEI-4 is not fetched by
`src.climate.download`, it was downloaded in 8 ranged chunks and logged in `data/raw/SOURCES.md`).

## Sources and why they differ from the design
| Design | Used | Why |
|---|---|---|
| ICRISAT DLD apportioned (1966 boundaries), all crops, prices, irrigation by crop | DES district APY 1998-99 to 2022-23 (1997-98 dropped: thin first-year coverage), unapportioned | `data.icrisat.org/dld` unreachable on 2026-11-14 **[team]** |
| 1993 start | core 1998-2022; 5-crop window 1993-2015 from the Mendeley mirror, reported separately | DES starts in 1997 **[team]** |
| district polygons from ICRISAT | geoBoundaries ADM2 2021 (ODbL), matched by name, aliases for known renames | ICRISAT ships none **[team]** |
| farm harvest prices | constant FAOSTAT India price per crop (`value_const / prod_t`, exact per item) | DES has no prices |
| irrigated area by crop | district gross irrigated / gross cropped area from the mirror, years <= 2015 only | not in DES |

## Rules (proposals)
- Year = start year of the agricultural year (1998 = 1998-99), matched to December SPEI-12 of that year (covers the monsoon).
- Season: use the "Total" row if present, else sum seasons. Fixed crop list: 47 crops reported in every year of the window, no "Other ...".
- A missing row is not filled. Diversity counts it as zero area; value and yield anomalies leave it missing.
- Coconut production is in nuts (blanked, area kept). Yields above 10x the crop's national median are blanked as source errors.
- Cotton, coconut, turmeric, cardamom, garlic, horse gram, khesari, mixed pulses, sannhamp have no clean FAOSTAT price: in diversity, not in the value index.
- DES rice production is ~0.6 of FAOSTAT's (milled vs paddy) with matching area; kept as published.
- Stable district: data in all 25 years. `split_flag`: a parent that kept its code but lost territory, detected relative to the state median.
- Value index uses crops with complete series in the district; a district needs >= 3 such crops holding >= 70% of its priced value.

## Robustness checks (added after the core run; all with `spei12_w` unless stated)
- **Conley standard errors** (`conley.py`, `outputs/stats/india_conley.csv`): Bartlett kernel on centroid distance (100, 200, 400 km), serial lag 5 years,
  next to district and state clustering. They are no larger than the district-clustered errors here, so spatial correlation is not what
  limits the diversity result.
- **Kharif drought variable** (`kharif.py`, `india_kharif.csv`, `india_kharif_perm.csv`): SPEI-4 of September (June-September), cropland weighted. Same districts
  and design; district totals still contain rabi crops, so this changes the exposure measure only, not the crop set.
- **Kharif-only crops** (`kharif_crops.py`, `india_kharif_crops*.csv`): table from DES monsoon-season rows only (Kharif + Autumn + Winter; a strict Kharif-only
  variant), no Total/Rabi/Summer/Whole-Year rows, 27 (strict: 24) crops. It covers 53-62% (strict 42-54%) of the core table's area and only states that publish
  a season split (17 states in the usable sample), so it is a different, smaller sample than the core, not a subset to compare one-to-one.
- **Documented drought years** (`outputs/validation/india_drought_years.csv`): 2002, 2009, 2014, 2015 with the all-India June-September rainfall departure
  and source. The sources are secondary (ORF, RBI Bulletin, a news copy of a Lok Sabha reply) and differ by 1-3 points for 2002 and 2009; no IMD
  end-of-season report could be read. The validation owner should replace them with the IMD reports.
- **Tests** (`tests/test_india.py`): hill1_tv timing, balanced-crop rule, season rule, split flag, name matching, `fast_t` against `fe_fit`, Conley against the
  heteroskedasticity-robust variance, distances, citation table.
- `fe_basis` uses an SVD basis because district and state-year dummies are collinear; the first version used a plain QR, which over-projected
  and changed the studentised permutation p-values (fixed, tests added).

## Not done
- Closing 2050 CMIP6 scenario (design section 6.5): needs a separate precipitation and PET download and a team decision.
