"""Independent checks from source rasters, hourly archives and actual crop footprints."""

import json
from datetime import date, timedelta

import netCDF4
import numpy as np
import pandas as pd

from . import era5
from .heat import era_footprints
from .seasons import KEY, footprints, indices_for
from .storage import RESULT, spei_path


def source_spei():
    foot = footprints()
    stored = pd.read_parquet(RESULT / "season_crop_year.parquet").set_index(KEY + ["year"])
    rows = []
    for iso, crop, year in [("IND", "WHEA", 1994), ("BRA", "SOYB", 2015), ("NGA", "PMIL", 2005)]:
        f = foot[(foot.iso3 == iso) & (foot.mapspam_crop == crop)]
        num = 0.0
        den = 0.0
        for (scale, month), g in f.groupby(["scale", "month"]):
            with netCDF4.Dataset(spei_path(int(scale))) as ds:
                idx = int(indices_for(ds, [year], month)[0])
                grid = ds["spei"][idx, :, :].filled(np.nan)
                source_row = 359 - g.cell.to_numpy() // 720
                source_col = g.cell.to_numpy() % 720
                values = grid[source_row, source_col]
                keep = np.isfinite(values)
                h = g.ha.to_numpy()[keep]
                num += np.dot(values[keep], h)
                den += h.sum()
        ref = stored.loc[(iso, crop, year)]
        actual = num / den
        np.testing.assert_allclose([actual, den], [ref.season, ref.observed_ha], rtol=1e-10, atol=1e-10)
        rows.append(
            {
                "iso3": iso,
                "mapspam_crop": crop,
                "year": year,
                "season_spei_from_source": actual,
                "observed_ha": den,
            }
        )
    return rows


def source_season_heat(crops):
    rows = []
    pooled = crops.set_index(KEY + ["year"])
    f = era_footprints(footprints())
    for iso, crop, year in [
        ("JAM", "SUGC", 1994),
        ("GMB", "POTA", 1994),
        ("DOM", "COTT", 1994),
        ("DOM", "COTT", 2023),
    ]:
        d = f[(f.iso3 == iso) & (f.mapspam_crop == crop)]
        numerator = np.zeros(3)
        dates = set()
        for row in d.itertuples():
            # Month and day independently mapped from the calendar's non-leap reference year.
            p = date(2001, 1, 1) + timedelta(days=row.plant - 1)
            h = date(2001, 1, 1) + timedelta(days=row.harvest - 1)
            first = date(year - int(p > h), p.month, p.day)
            last = date(year, h.month, h.day)
            amount = np.zeros(3)
            for i in range((last - first).days + 1):
                day = first + timedelta(days=i)
                dates.add(day)
                with np.load(era5.DAY_DIR / f"{day}.npz") as s:
                    amount += s["ddd"].reshape(3, -1)[:, row.grid]
            numerator += amount * row.ha
        actual = numerator / d.ha.sum()
        ref = pooled.loc[(iso, crop, year), ["edd28", "edd30", "edd32"]]
        np.testing.assert_allclose(actual, ref.to_numpy(float), atol=1e-9, rtol=1e-10)
        rows.append(
            {
                "iso3": iso,
                "mapspam_crop": crop,
                "year": year,
                "calendar_ha": d.ha.sum(),
                "first_day": min(dates).isoformat(),
                "last_day": max(dates).isoformat(),
                "weatherbench_days": sum(t.year < 2023 for t in dates),
                "arco_days": sum(t.year >= 2023 for t in dates),
                **dict(zip(["edd28", "edd30", "edd32"], actual)),
            }
        )
    return rows


def source_hours():
    checks = []
    for day in [date(1994, 6, 17), date(2023, 1, 4)]:
        if day.year == 2023:
            hourly = np.stack([era5.native_hour(day, h) for h in range(24)])
        else:
            offset = (day - era5.WB_START).days
            t = era5.wb_hours(offset // 2)
            hourly = t[24 * (offset % 2) : 24 * (offset % 2 + 1)]
        # Independent threshold implementation, using source float precision and float64 sums.
        deg = np.array(
            [
                np.clip((hourly - np.float32(273.15)) - threshold, 0, None).sum(axis=0, dtype=float)
                / 24.0
                for threshold in (28.0, 30.0, 32.0)
            ],
            dtype=np.float32,
        )
        with np.load(era5.DAY_DIR / f"{day}.npz") as cached:
            np.testing.assert_array_equal(deg, cached["ddd"])
            assert str(cached["source"]) == era5.source_signature(day)
        checks.append(
            {
                "day": day.isoformat(),
                "utc_hours": 24,
                "thresholds_c": [28, 30, 32],
                "grid_cells": 121 * 240,
                "max_abs_difference_degree_days": 0.0,
            }
        )
    return checks


def run(crops=None):
    if crops is None:
        crops = pd.read_parquet(RESULT / "heat_crop_year.parquet")
    result = {
        "source_spei": source_spei(),
        "source_heat_seasons": source_season_heat(crops),
        "hourly_recalculations": source_hours(),
    }
    path = RESULT / "source_measurement_checks.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2) + "\n")
    tmp.replace(path)


if __name__ == "__main__":
    run()
