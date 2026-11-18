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

# Phase 4 (docs/phase4-design.md, src/india/): Indian district panel.
INDIA_CLEAN = CLEAN_REAL / "india"
INDIA_OUT = OUT_REAL / "india"
INDIA_DISTRICTS = INDIA_CLEAN / "districts.csv"
INDIA_DISTRICTS_EXT5 = INDIA_CLEAN / "districts_ext5.csv"
INDIA_DISTRICT_MATCH = INDIA_CLEAN / "district_match.csv"
INDIA_GEOJSON = INDIA_CLEAN / "districts.geojson"
INDIA_CROPS = INDIA_CLEAN / "crops.parquet"
INDIA_CROPS_EXT5 = INDIA_CLEAN / "crops_ext5.parquet"
INDIA_CLIMATE = INDIA_CLEAN / "climate.parquet"
INDIA_CONTROLS = INDIA_CLEAN / "mirror_controls.parquet"
INDIA_METRICS = INDIA_OUT / "metrics.parquet"
INDIA_METRICS_EXT5 = INDIA_OUT / "metrics_ext5.parquet"
INDIA_PANEL = INDIA_OUT / "panel.parquet"
INDIA_PANEL_EXT5 = INDIA_OUT / "panel_ext5.parquet"
INDIA_CROP_ANOMALIES = INDIA_OUT / "crop_anomalies.parquet"
INDIA_DROPPED = INDIA_OUT / "dropped_districts.csv"
INDIA_DROUGHT_YEARS = VALIDATION_DIR / "india_drought_years.csv"


def fig_dir(owner):
    """figures/<owner>/, or figures/<owner>/dummy/ under USE_DUMMY so dummy plots never sit beside real ones."""
    d = FIGURES / owner / "dummy" if USE_DUMMY else FIGURES / owner
    d.mkdir(parents=True, exist_ok=True)
    return d

