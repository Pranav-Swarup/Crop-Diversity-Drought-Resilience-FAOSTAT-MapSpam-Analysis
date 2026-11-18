# EVST: Crop Diversity & Drought Resilience (POC)

The study compares crop diversity with four measures of resilience across 107 countries during 1993–2023 and within Indian districts in two agricultural panels. Open the downloaded HTML file in a browser to explore the maps, crop histories, portfolio scenarios and underlying results offline. [Explorer](explorer/index.html) · [Research results](report.md) · [Methods and contracts](CLAUDE.md) · [November verification](metadata.md#november-2026-verification)

## Setup

```
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Dummy data

Generate the separate development fixture with:

```
python -m src.common.make_dummy
```

This writes to `data/dummy/` and `outputs/dummy/` only. It has 30 countries (real iso3 codes, made-up numbers), 1993–2023, 8 crops each. Three countries are built to fail the inclusion rules (JAM, MNG, LBY). `metrics.parquet` and `crop_anomalies.parquet` in `outputs/dummy/` are placeholders until you run `USE_DUMMY=1 python -m src.metrics.run`.

## Switching between dummy and real inputs

Take every path from `src/common/paths.py`. With `USE_DUMMY=1` they point at the dummy folders, otherwise at `data/clean/` and `outputs/`.

```python
from src.common import paths
from src.common.contract import load, validate

crops = load("crops")        # reads paths.CROPS and validates it
validate(my_df, "climate")   # call at the end of every pipeline, before writing
```

```
USE_DUMMY=1 python -m src.metrics.run    # dummy inputs
python -m src.metrics.run                # real inputs
```

Check every contract file that exists:

```
python -m src.common.contract
USE_DUMMY=1 python -m src.common.contract
```

## Run order

All scripts run from the repo root as `python -m src.<module>.<script>`.

1. `src.common.make_dummy` (once, for development)
2. Download national source files with `src.climate.download` and `src.crops.download`. The climate downloader documents the MapSPAM Dataverse guestbook. The crops downloader restores the 11 November 2026 WDI snapshot bundled under `data/sources/wdi/`.
3. Run `src.crops.countries`, `src.crops.crops`, `src.crops.controls` and `src.crops.quality`.
4. Run `src.climate.geometry`, `src.climate.cropland`, `src.climate.aggregate --per-crop`, `src.climate.events` and `src.climate.mapspam_area` before fitting crop slopes. Run `src.climate.figures` and `src.crops.figures` for the base figures.
5. Run `src.metrics.all`, then `src.validation.hit_rate` and `src.validation.case_studies` to reproduce the published India, Colombia and Finland cases and their shared crop-slope table. See [metric definitions](src/metrics/METHODS.md).
6. Run the extensions below before building the complete explorer.

### Phase 2 extensions

Definitions and the data contract are in CLAUDE.md, "Phase 2 extensions". Run after the national pipeline in steps 2–5:

1. `src.crops`: `download` (adds FAOSTAT Food Balances, SUA and Producer Prices), `trade`, `prices`, `calories`
2. `src.climate`: `wet_events`, `download_cru`, `heat` (`heat` needs the gridded files from `src.climate.cropland`; CRU TS v4.10 downloads without a login)
3. `src.metrics`: `wet`, `heat`, `panel`, `oos`, `portfolio`, `calorie_faces` (`panel` and `portfolio` take about 2 minutes each)
4. `src.crops`: `buffers` (needs `outputs/metrics.parquet` and `outputs/crop_anomalies.parquet`)
5. `src.climate.analysis`: `response_paradox`, `spatial_and_flash`

### Phase 3

Definitions and the data contract are in CLAUDE.md, "Phase 3". Run after phase 2:

1. `src.climate`: `mapspam_area`, `exposure_pre`, `download_calendar`, `exposure_season` (need the gridded raw files)
2. `src.metrics`: `exposure_diversity`, `severity` (`severity` takes about 12 minutes)
3. `src.crops`: `download_trade_matrix` (420 MB), `trade_matrix`, `supply_diversity`

### Phase 4 (Indian districts)

The district module documents the implemented sources, selection rules and model checks. The original proposal records the planned design. [Implemented methods](src/india/README.md) · [Original proposal](docs/phase4-design.md)

1. `src.india`: `download_docs`, `download`, `download_apy`, `download_geo`, `clean`, `geo`, `climate`, `extended`, `panel`, `conley`, `kharif`, `kharif_crops`
2. `src.explorer`: `build`, `collect` (embeds Phase 4 Indian district map, district timeseries, and statistical tables into `explorer/index.html` and collects `figures/india/` into `figures/deck/`)

Rebuild the real dashboard and numbered figure collection with `python -m src.explorer`. Rebuild the synthetic dashboard separately with `USE_DUMMY=1 python -m src.explorer.build`. Both HTML files bundle their map and JavaScript dependencies. Run `python -m pytest` for the regression suite. The offline rendering tests also run when Chrome or Chromium and the optional Python package `selenium` are installed.

## Layout

```
src/{common,crops,climate,metrics,validation,india,explorer}/
data/raw/      downloads, gitignored; sources logged in data/raw/SOURCES.md
data/clean/    contract inputs (including data/clean/india/)
data/dummy/    synthetic contract inputs
outputs/       metrics, india/, stats/, validation/, dummy/
figures/{data,climate,results,validation,india,deck}/
explorer/      index.html
```

