"""Statistical tables with their original columns and numeric precision."""
import pandas as pd

from src.common import paths


def result_tables():
    tables = {}
    files = sorted(paths.STATS_DIR.glob("*.csv"))
    files.extend(sorted(paths.VALIDATION_DIR.glob("*.csv")))
    if not paths.USE_DUMMY:
        files.extend(paths.OUT / name for name in ("exposure_diversity.parquet", "supply_diversity.parquet"))
    for path in files:
        frame = (pd.read_csv(path, float_precision="round_trip") if path.suffix == ".csv"
                 else pd.read_parquet(path))
        tables[path.stem] = {"columns": frame.columns.tolist(),
                             "rows": frame.astype(object).where(frame.notna(), None).values.tolist()}
    return tables
