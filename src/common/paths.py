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
