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
python -m src.india.climate         # district SPEI-12 (December), climate.parquet
python -m src.india.extended        # mirror -> 5-crop table 1993-2015 + irrigation/fertiliser controls
python -m src.india.panel           # faces, panel, severity, tables, figures
```
Needs `data/raw/spei/spei12.nc` and, for the design's drought variable `spei12_w`, MapSpam via `src.climate.cropland` (manual download).

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

## Not done
- Closing 2050 CMIP6 scenario (design section 6.5): needs a separate precipitation and PET download and a team decision.
- Conley spatial standard errors (state-clustered errors instead).
- Kharif-season SPEI variant (needs SPEI-4 or a daily/monthly rebuild; SPEI-6 and SPEI-12 only are downloaded).
- Documented drought years are given without citations; the validation owner adds them.
