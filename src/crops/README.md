# Crop data notes (src/crops)

Run order: `download`, `countries`, `crops`, `controls`, `quality`, `figures` (each as `python -m src.crops.<script>`). `countries` needs the Natural Earth file from `src.climate.download`.

## Team decisions (2026-10-02)

The team left these three choices to the crop-data owner. They are not in CLAUDE.md.

1. Keep `value_const` computed as production x item price where FAOSTAT does not publish it, with a committed record (`data/clean/value_computed.csv`).
2. Keep the 50% cut-off in the inclusion rule: a year counts as non-imputed when at most half of its value comes from rows flagged E or I.
3. Flat series are recorded in `data/clean/flat_series.csv` and not removed from `crops.parquet`.

## crops.parquet

- 3.8% of rows (10,203) have `value_const` computed as production x item price because FAOSTAT did not publish it. That is 1.9% of total value, but it is concentrated: 0.2% of value in 1993-2017 and 7.1% in 2018-2023, when FAOSTAT publishes the constant-I$ value for only a few items in EU countries. 29 countries have more than 10% of their value computed.
- The item price is FAOSTAT's own: value / production is constant per item to within 0.81%. `crops.py` asserts that production x item price reproduces the 263,450 values FAOSTAT does publish; the difference is 0.0001% of their total.
- `data/clean/value_computed.csv` (not a contract file) lists the 250 country-years with any computed value: `n_rows`, `n_computed`, `value_const`, `value_computed`, `share_value_computed`.
- 730 rows have no price (flax, jojoba seeds, tallowtree seeds) and `value_const` is left missing.
- FAOSTAT stops reporting some minor crops for EU countries from 2018 (official figures, flag A). In 20 included countries, crops holding more than 1% of 2014-2017 value have no rows from 2018 (BLX 9.6%, LVA 8.6%, LTU 7.7%, EST 5.4%, HUN 4.7%, GRC 4.7%). Nothing is filled in. `quality.py` prints this check.

## quality.csv

- Inclusion at other imputed-value cut-offs: 25% gives 85 countries, 50% gives 107, 75% gives 126 (`quality.py` prints this).

## flat_series.csv (not a contract file)

- Lists crop series that fake stability: detrended CV < 1%, or the same value 3+ consecutive years with more than half of the years imputed. Detrending and CV are imported from `src/metrics/`.
- 719 series, 347 of them in included countries. Among included countries they hold more than 10% of value in GHA, BFA, TKM and JAM.
- Most listed series (635) are on the list for a run of identical imputed values, not for being flat throughout, so the list is a record for robustness checks and whole series are not removed.

## controls.parquet

- `irrig_share` is FAOSTAT "Land area equipped for irrigation" / "Cropland". It exceeds 1 in 112 country-years (GUY and TKM in every year, NZL in 22 years from 2001, six others in 1 to 11 years). The cause is not confirmed. Values are as published, not capped.
- For a merged entity `irrig_share` is given only when both items come from the same members. FAOSTAT has no irrigation series for Luxembourg, so `BLX` is missing in 2000-2023.
- `gdp_pc` and `fert_kg_ha` are missing for `BLX`, `SCG`, `SDX` and `TWN` (no WDI series).

## countries.csv

- French Guiana, Guadeloupe, Martinique and Réunion (series end in 2006, each under 100,000 ha) are dropped because Natural Earth has no separate iso3 for them.
