# EVST: Crop Diversity & Drought Resilience (POC)

Project rules, fixed analysis choices, the data contract and folder ownership are in [CLAUDE.md](CLAUDE.md). Per-person prompts are in [PROMPTS.md](PROMPTS.md).

## Setup

```
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Dummy data

Until the real files exist, build against synthetic files with the contract's exact columns:

```
python -m src.common.make_dummy
```

This writes to `data/dummy/` and `outputs/dummy/` only. It has 30 countries (real iso3 codes, made-up numbers), 1993–2023, 8 crops each. Three countries are built to fail the inclusion rules (JAM, MNG, LBY). `metrics.parquet` and `crop_anomalies.parquet` in `outputs/dummy/` are placeholders until `src.metrics.run` exists.

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
2. `src.crops`: `download`, `countries`, `crops`, `controls`, `quality`
3. `src.climate`: `download`, `geometry`, `cropland`, `aggregate`, `events`. The MapSPAM file must be downloaded by hand (Dataverse guestbook); `download` prints the steps. `src.crops.countries` needs the Natural Earth file from `src.climate.download`.
4. `src.metrics`: `all` (runs `run`, `stats`, `robustness`, `figures`); methods in `src/metrics/METHODS.md`
5. `src.validation`: `hit_rate`, `case_studies`
6. `src.explorer`: `build`, `diagram`, `collect`

## Layout

```
src/{common,crops,climate,metrics,validation,explorer}/
data/raw/      downloads, gitignored; sources logged in data/raw/SOURCES.md
data/clean/    contract inputs
data/dummy/    synthetic contract inputs
outputs/       metrics, stats/, validation/, dummy/
figures/{data,climate,results,validation,deck}/
explorer/      index.html
```
