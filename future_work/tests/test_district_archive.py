"""Regression checks for the units and sample restrictions in the archived district study."""

import numpy as np
import pandas as pd
import pytest

from future_work import district, district_archive


def test_positive_production_below_area_precision_preserves_the_observation():
    k = {"state_code": "S", "district_code": "D", "crop": "C"}
    apy = pd.DataFrame([dict(k, year=2015, area_ha=0.0, prod_t=12.0)])
    prices = apy[district.KEY].assign(price_inr_t=250.0)
    irrig = apy[district.KEY].assign(irrigated_ha=3.0)
    row = district.prepare_crop(apy, prices, irrig, 2015, rounded_areas=True).iloc[0]
    assert row.area_below_reporting_precision and row.irrigation_invalid
    assert row.area_ha_reported == 0 and row.prod_t == 12 and row.value_const == 3000
    assert np.isnan(row.area_ha) and np.isnan(row.irrigation_share) and np.isnan(row.yield_t_ha)


def test_nonproducing_crop_cannot_qualify_or_block_an_irrigation_observation():
    d = pd.DataFrame(
        {
            "dist_code": ["D"] * 4,
            "year": [2015, 2016] * 2,
            "crop": ["A", "A", "B", "B"],
            "area_ha": [4.0, 4.0, 5.0, 5.0],
            "value_const": [10.0, 0.0, 0.0, 0.0],
            "irrigated_ha": [2.0, 0.0, np.nan, np.nan],
            "irrigation_invalid": [False] * 4,
            "irrigation_share": [0.5, 0.0, np.nan, np.nan],
        }
    )
    years = [2015, 2016]
    eligible = district.value_sample(d, years)
    assert eligible.loc[eligible.crop.eq("B"), "area_ha"].eq(5).all()
    assert eligible.loc[eligible.crop.eq("B"), "value_const"].isna().all()
    assert eligible.loc[eligible.crop.eq("A"), "value_const"].tolist() == [10.0, 0.0]
    assert district.lagged_irrigation(d, years).irrig_lag.tolist() == [0.5, 0.0]


def test_native_headers_units_and_negative_sentinel_are_strict():
    d = pd.DataFrame({"WHEAT AREA": ["-1.0", "0.0", "2.35"]})
    units = {"WHEAT AREA": "1000 ha"}
    amounts = district_archive.measurements(d, units, "WHEAT AREA", "1000 ha", 1000)
    np.testing.assert_allclose(amounts, [np.nan, 0.0, 2350.0], equal_nan=True)
    with pytest.raises(ValueError, match="units"):
        district_archive.measurements(d, units, "WHEAT AREA", "ha", 1000)
    d.loc[0, "WHEAT AREA"] = "-9"
    with pytest.raises(ValueError, match="negative"):
        district_archive.measurements(d, units, "WHEAT AREA", "1000 ha", 1000)


def test_land_use_datastore_reconciliation_joins_ids_and_preserves_blank(tmp_path):
    import csv
    import hashlib

    from future_work.district_land_use import FIELDS, canonical, compare_export

    def record(key, area):
        return [
            key,
            "2000-2001",
            "State",
            7,
            "District",
            12,
            "Food",
            "Cereal",
            "Cereal",
            "Wheat",
            8,
            "nan",
            area,
        ]

    pages = [
        {
            "offset": 0,
            "rows": 2,
            "canonical_sha256": hashlib.sha256(
                canonical(record(0, None)) + canonical(record(1, 12.50))
            ).hexdigest(),
        }
    ]
    path = tmp_path / "land_use.csv"

    def write(rows):
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(FIELDS)
            writer.writerows(rows)

    write([record(1, "12.500"), record(0, "")])
    assert compare_export(path, pages) == 2
    write([record(1, "12.500"), record(0, "0")])
    with pytest.raises(ValueError, match="fields differ"):
        compare_export(path, pages)
    write([record(0, ""), record(0, "")])
    with pytest.raises(ValueError, match="repeated"):
        compare_export(path, pages)
