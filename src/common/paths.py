"""Central paths. Set USE_DUMMY=1 to read/write the dummy copies instead of the real files."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
USE_DUMMY = os.environ.get("USE_DUMMY") == "1"

RAW = ROOT / "data" / "raw"
CLEAN_REAL = ROOT / "data" / "clean"
OUT_REAL = ROOT / "outputs"
CLEAN_DUMMY = ROOT / "data" / "dummy"
OUT_DUMMY = ROOT / "outputs" / "dummy"

CLEAN = CLEAN_DUMMY if USE_DUMMY else CLEAN_REAL
OUT = OUT_DUMMY if USE_DUMMY else OUT_REAL

COUNTRIES = CLEAN / "countries.csv"
CROPS = CLEAN / "crops.parquet"
CONTROLS = CLEAN / "controls.parquet"
QUALITY = CLEAN / "quality.csv"
CLIMATE = CLEAN / "climate.parquet"
EVENTS = CLEAN / "events.csv"
COUNTRIES_GEOJSON = CLEAN / "countries.geojson"

METRICS = OUT / "metrics.parquet"
CROP_ANOMALIES = OUT / "crop_anomalies.parquet"
STATS_DIR = OUT / "stats"
VALIDATION_DIR = OUT / "validation"

# Not switched by USE_DUMMY.
FIGURES = ROOT / "figures"
EXPLORER = ROOT / "explorer"
SOURCES_MD = RAW / "SOURCES.md"

NATIONAL = OUT / "metrics_national.parquet"

# Phase 2 extensions (CLAUDE.md "Phase 2 extensions").
CROP_CLIMATE = CLEAN / "crop_climate.parquet"
WET_EVENTS = CLEAN / "wet_events.csv"
CALORIES = CLEAN / "calories.parquet"
TRADE = CLEAN / "trade.parquet"
PRICES = CLEAN / "prices.parquet"
METRICS_WET = OUT / "metrics_wet.parquet"
METRICS_KCAL = OUT / "metrics_kcal.parquet"
PORTFOLIO_CROPS = OUT / "portfolio_crops.parquet"
PORTFOLIO_FRONTIER = OUT / "portfolio_frontier.parquet"
HEAT = CLEAN / "heat.parquet"
HEAT_EVENTS = CLEAN / "heat_events.csv"
METRICS_HEAT = OUT / "metrics_heat.parquet"

# Phase 3 (CLAUDE.md "Phase 3: what makes a portfolio diversified").
MAPSPAM_AREA = CLEAN / "mapspam_area.parquet"
CROP_CLIMATE_SEASON = CLEAN / "crop_climate_season.parquet"
TRADE_PARTNERS = CLEAN / "trade_partners.parquet"
TRADE_PARTNERS_MIRROR = CLEAN / "trade_partners_mirror.parquet"
EXPOSURE_DIVERSITY = OUT / "exposure_diversity.parquet"
SUPPLY_DIVERSITY = OUT / "supply_diversity.parquet"


def fig_dir(owner):
    """figures/<owner>/, or figures/<owner>/dummy/ under USE_DUMMY so dummy plots never sit beside real ones."""
    d = FIGURES / owner / "dummy" if USE_DUMMY else FIGURES / owner
    d.mkdir(parents=True, exist_ok=True)
    return d
