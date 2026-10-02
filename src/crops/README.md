# Crop data notes (src/crops)

Run order: `download`, `countries`, `crops`, `controls`, `quality`, `figures` (each as `python -m src.crops.<script>`). `countries` needs the Natural Earth file from `src.climate.download`.

## crops.parquet

- 3.8% of rows (10,203) have `value_const` computed as production x item price because FAOSTAT did not publish it. That is 1.9% of total value, but it is concentrated: 0.2% of value in 1993-2017 and 7.1% in 2018-2023, when FAOSTAT publishes the constant-I$ value for only a few items in EU countries. 29 countries have more than 10% of their value computed.
- The item price is FAOSTAT's own: value / production is constant per item to within 0.81%.
- Per-country list: `data/raw/faostat/value_filled_from_price.csv` (written by `crops.py`).
- 730 rows have no price (flax, jojoba seeds, tallowtree seeds) and `value_const` is left missing.

## quality.csv

- A year counts as non-imputed when at most half of its value comes from rows flagged E or I. The 50% cut-off is not in CLAUDE.md; 25% gives 85 included countries, 50% gives 107, 75% gives 126.

## flat_series.csv (not a contract file)

- Lists crop series that fake stability: detrended CV < 1%, or the same value 3+ years running with more than half of the years imputed. Detrending and CV are imported from `src/metrics/`.
- It is a record only; `crops.parquet` and `quality.csv` are not changed by it.
- 719 series, 347 of them in included countries. Among included countries they hold more than 10% of value in GHA, BFA, TKM and JAM.

## controls.parquet

- `irrig_share` is FAOSTAT "Land area equipped for irrigation" / "Cropland". It exceeds 1 in 112 country-years (GUY and TKM in every year, NZL in 22 years from 2001, six others in 1 to 11 years). The cause is not confirmed. Values are as published, not capped.
- For a merged entity `irrig_share` is given only when both items come from the same members. FAOSTAT has no irrigation series for Luxembourg, so `BLX` is missing in 2000-2023.
- `gdp_pc` and `fert_kg_ha` are missing for `BLX`, `SCG`, `SDX` and `TWN` (no WDI series).

## countries.csv

- French Guiana, Guadeloupe, Martinique and Réunion (series end in 2006, each under 100,000 ha) are dropped because Natural Earth has no separate iso3 for them.
