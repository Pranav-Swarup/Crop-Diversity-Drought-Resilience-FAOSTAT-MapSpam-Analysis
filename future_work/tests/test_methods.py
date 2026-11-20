"""Local tests for calendar cutoffs, observed data handling, source bytes and geography."""

import copy
import hashlib
from datetime import date

import numpy as np
import pandas as pd
import pytest

from future_work import (
    district,
    era5,
    heat,
    seasons,
    source_objects,
    vegetation,
    vegetation_grid,
)
from future_work.history import detectable


def test_inclusive_months_and_leap_years():
    month, span = seasons.windows([1, 365, 20, 365, 59], [1, 1, 19, 365, 60])
    assert list(month) == [1, 1, 1, 12, 3]
    assert list(span) == [1, 2, 13, 1, 2]
    with pytest.raises(ValueError):
        seasons.windows(0, 10)
    for year, n in [(2019, 2), (2020, 3)]:
        a, b = heat.bounds([59, 365, 60], [60, 1, 60], year)
        assert list(b - a) == [n, 2, 1]
    assert heat.ordinal(60, 2020) == (date(2020, 3, 1) - date(2020, 1, 1)).days


def test_hourly_extremes_are_accumulated_before_averaging():
    temp = np.full((24, 121, 240), 273.15 + 24, dtype="float32")
    temp[-6:] = 273.15 + 34
    for actual, expected in zip(era5.daily(temp), [1.5, 1.0, 0.5]):
        np.testing.assert_allclose(actual, expected, atol=1e-5)
    with pytest.raises(ValueError):
        era5.daily(temp[:-1])
    temp[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        era5.daily(temp)


def test_constant_heat_exposure_has_no_standardized_anomaly():
    # A smoother can introduce roundoff even when every observed value is identical.
    for level in [0.0, 2.125]:
        assert np.isnan(heat.standardized_excess(pd.Series([level] * 31))).all()
    annual = pd.Series(np.tile([0.0, 3.0, 9.0, 2.0], 8)[:31])
    assert np.std(heat.standardized_excess(annual), ddof=1) == pytest.approx(1)


def test_geographic_overlap_conserves_area_and_wraps_longitude():
    d = pd.DataFrame(
        {
            "iso3": "AAA",
            "mapspam_crop": "WHEA",
            "plant": 20,
            "harvest": 65,
            "ha": [7.0, 5.0, 13.0],
            "cell": [0, 719, 180 * 720 + 360],
        }
    )
    result = heat.era_footprints(d)
    np.testing.assert_allclose(result.ha.sum(), 25.0)
    assert result.grid.between(0, 121 * 240 - 1).all()
    wy, wx = era5.regrid_weights()
    np.testing.assert_allclose(wy.sum(axis=1), 1)
    np.testing.assert_allclose(wx.sum(axis=1), 1)
    assert wx[0, -1] > 0
    grid = vegetation_grid.overlap([-1.5, -0.5, 0.5, 1.5], [-1, 1], True)
    np.testing.assert_allclose(grid.sum(axis=1), 1)


def test_packed_noaa_values_and_quality_bits():
    class Var:
        scale_factor = 0.01
        add_offset = 0
        _FillValue = -9999

        def __init__(self, values):
            self.values = np.asarray(values)

        def set_auto_maskandscale(self, value):
            assert not value

        def __getitem__(self, key):
            return self.values[key]

    actual = vegetation_grid.unpack(
        Var([5500, 7300, 3010, 10001, -9999, 7100, 6000, 5000]), Var([4, 6, 8, 4, 4, 20, 5, 0])
    )
    np.testing.assert_allclose(
        actual, [55, 73, 30.1, np.nan, np.nan, np.nan, np.nan, np.nan], equal_nan=True
    )


def test_noaa_declaration_and_negative_fill():
    header = "Mean data for IND ,  from 1992 to 2023, weekly; version='GC_current' area with 'WHEA' year,week, SMN,SMT,VCI,TCI, VHI"
    html = header + "<pre>1997,20,0,0,62.2,0,0\n1997,21,0,0,-1,0,0</pre>"
    np.testing.assert_allclose(
        vegetation.parse_series(html, "IND", "WHEA").vci, [62.2, np.nan], equal_nan=True
    )
    with pytest.raises(ValueError):
        vegetation.parse_series(html, "BLX", "WHEA")
    with pytest.raises(ValueError):
        vegetation.parse_series(html.replace("1997,21", "1997,20"), "IND", "WHEA")


def test_source_pins_cover_all_era_objects_and_reject_corruption():
    pins = source_objects.pinned()
    assert len(pins) == 14423
    assert sum(a == "arco" for a, _ in pins) == 365 * 24
    overlap_key = f"{(era5.OVERLAP_DAY - era5.WB_START).days // 2}.0.0"
    assert ("weatherbench", overlap_key) in pins
    assert len(era5.source_signature(date(1992, 1, 1))) == 64
    assert era5.source_signature(date(1992, 1, 1)) != era5.source_signature(date(1992, 1, 2))
    with pytest.raises(ValueError):
        source_objects.checked_bytes(b"bad", next(iter(pins.values())))


def test_minimum_detectable_effect_scales_with_observed_uncertainty():
    assert detectable(0.01, 74) == pytest.approx(detectable(0.02, 74) / 2)
    assert 2.8 < detectable(1, 74) < 2.9


def test_observed_farm_price_and_lagged_crop_irrigation():
    k = {"state_code": "01", "district_code": "009", "crop": "Paddy"}
    apy = pd.DataFrame([dict(k, year=y, area_ha=10.0, prod_t=20.0) for y in [2015, 2016]])
    price = apy[district.KEY].assign(price_inr_t=[2000.0, 3000.0])
    irr = apy[district.KEY].assign(irrigated_ha=[6.0, 12.0])
    d = district.prepare_crop(apy, price, irr, 2015)
    np.testing.assert_allclose(d.value_const, [40000, 40000])
    assert d.irrigation_invalid.tolist() == [False, True]
    lag = district.lagged_irrigation(d, [2015, 2016])
    np.testing.assert_allclose(lag.irrig_lag, [0.6, np.nan], equal_nan=True)
    assert lag.year.tolist() == [2016, 2017]
    partial = d.assign(crop="Wheat", area_ha=[1000.0, np.nan], irrigated_ha=0.0, irrigation_share=0.0)
    paired = district.lagged_irrigation(pd.concat([d, partial]), [2015, 2016])
    pd.testing.assert_frame_equal(paired, lag)
    with pytest.raises(ValueError):
        district.prepare_crop(apy, price, irr, 1993)


def test_official_wide_and_long_inputs_keep_ids_and_missing_values(tmp_path):
    f = tmp_path / "prices.csv"
    f.write_text("S,D,Y,P\n01,009,2015-16,25\n01,009,2016-17,-999\n")
    spec = {
        "source_url": district.PORTAL,
        "file": f.name,
        "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
        "keys": {"state_code": "S", "district_code": "D", "year": "Y"},
        "missing_values": ["-999"],
        "crops": {"Paddy": {"price_inr_t": {"column": "P", "unit": "INR/quintal"}}},
    }
    out = district.normalize("prices", spec, tmp_path)
    assert out.district_code.tolist() == ["009", "009"]
    np.testing.assert_allclose(out.price_inr_t, [250, np.nan], equal_nan=True)
    with pytest.raises(ValueError):
        district.crop_year("2015-19")
    with pytest.raises(ValueError):
        district.normalize("prices", dict(spec, source_url="https://example.org"), tmp_path)
    with pytest.raises(ValueError):
        district.normalize("prices", dict(spec, sha256="wrong"), tmp_path)
    invalid = copy.deepcopy(spec)
    invalid["crops"]["Paddy"]["price_inr_t"]["unit"] = "international dollars"
    with pytest.raises(ValueError):
        district.normalize("prices", invalid, tmp_path)
    f.write_text("S,D,Y,C,P\n01,009,2015,raw rice,20\n")
    spec.update(
        sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
        crop_column="C",
        crop_labels={"raw rice": "Paddy"},
        values=spec.pop("crops")["Paddy"],
    )
    assert district.normalize("prices", spec, tmp_path).price_inr_t.item() == 200


def test_district_release_reaches_models_and_plots_only_in_temporary_directory(tmp_path, monkeypatch):
    import json

    import geopandas as gpd
    from shapely.geometry import box

    from future_work import figures
    from future_work.freeze import digest

    rng = np.random.default_rng(512)
    inputs = tmp_path / "raw"
    results = tmp_path / "results"
    fig = tmp_path / "figures"
    for f in [inputs, results, fig]:
        f.mkdir()
    monkeypatch.setattr(district, "INPUT", inputs)
    monkeypatch.setattr(district, "RESULT", results)
    monkeypatch.setattr(figures, "FIG", fig)
    monkeypatch.setattr(district, "discovery", lambda: [])
    states = ["01", "02"]
    years = list(range(1998, 2018))
    geo = []
    obs = []
    for i in range(10):
        state = states[i // 5]
        d = f"{i + 1:03d}"
        longitude = 74 + i
        latitude = 18 + i % 3
        geo.append(
            {"S": state, "D": d, "geometry": box(longitude, latitude, longitude + 1, latitude + 1)}
        )
        for year in years:
            for j, name in enumerate(["a", "b", "c", "d"]):
                area = 100 * (j + 1) + rng.uniform(1, 70)
                prod = area * (1.1 + 0.12 * j + rng.uniform(0, 0.6))
                obs.append(
                    [state, d, year, name, area, prod, 200 + 20 * j + i, area * rng.uniform(0.1, 0.9)]
                )
    d = pd.DataFrame(obs, columns=["S", "D", "Y", "crop", "ha", "tons", "price", "irrigation"])
    d.to_csv(inputs / "test.csv", index=False)
    g = gpd.GeoDataFrame(geo, crs=4326)
    g.to_file(inputs / "geometry.geojson", driver="GeoJSON")

    def record(name):
        return {"file": name, "sha256": digest(inputs / name), "source_url": district.PORTAL}

    (inputs / "evidence.txt").write_text("TEST-ONLY-1 is the latest test fixture.")
    common = {
        **record("test.csv"),
        "keys": {"state_code": "S", "district_code": "D", "year": "Y"},
        "crop_column": "crop",
        "boundary_vintage": "test polygons",
    }
    manifest = {
        "evidence": record("evidence.txt"),
        "latest_evidence_quote": "latest test fixture",
        "boundary_vintage": "test polygons",
        "release_id": "TEST-ONLY-1",
        "series": "apportioned",
        "analysis_years": [years[0], years[-1]],
        "price_reference_year": 2015,
        "geography": {
            **record("geometry.geojson"),
            "keys": {"state_code": "S", "district_code": "D"},
            "boundary_vintage": "test polygons",
        },
        "apy": dict(
            common,
            values={
                "area_ha": {"column": "ha", "unit": "ha"},
                "prod_t": {"column": "tons", "unit": "t"},
            },
        ),
        "prices": dict(common, values={"price_inr_t": {"column": "price", "unit": "INR/quintal"}}),
        "irrigation": dict(common, values={"irrigated_ha": {"column": "irrigation", "unit": "ha"}}),
    }
    (inputs / "release.json").write_text(json.dumps(manifest))
    assert district.run()
    district.render_if_ready()
    sample = pd.read_csv(results / "district_comparison.csv")
    official = sample[sample.model.eq("official_crop_irrigation")]
    assert len(official) == 5 and set(official.n_districts) == {10}
    assert (fig / "05_district_comparison.png").stat().st_size > 1000
    assert pd.read_parquet(results / "district_paired_quantities.parquet").empty
    climate = pd.read_parquet(results / "district_climate.parquet")
    assert len(climate) == 200 and climate.spei12_w.notna().all()
    assert json.loads((results / "district_status.json").read_text())["release_id"] == "TEST-ONLY-1"


def test_archived_observed_satellite_reading_and_missing_periods():
    from future_work.storage import RESULT

    weeks = pd.read_parquet(RESULT / "vci_weekly.parquet")
    wheat = weeks.query('iso3=="IND" and mapspam_crop=="WHEA" and year==1997 and week==20')
    assert wheat.vci.item() == pytest.approx(62.20)
    years = pd.read_parquet(RESULT / "vci_seasonal.parquet")
    coverage = years.groupby("year").vci.count()
    assert coverage.loc[2004] < coverage.loc[2003] and coverage.loc[1994] < coverage.loc[1993]
    assert weeks.query("year==1994 and week==40").vci.isna().all()


def test_xlsx_district_import_on_a_named_sheet(tmp_path):
    import openpyxl

    from future_work.freeze import digest

    book = openpyxl.Workbook()
    sheet = book.active
    sheet.title = "Farm prices"
    sheet.append(["State", "District", "Year", "Commodity", "Rupees"])
    sheet.append(["02", "007", "2015-16", "Wheat", 24])
    file = tmp_path / "prices.xlsx"
    book.save(file)
    spec = {
        "file": file.name,
        "sha256": digest(file),
        "source_url": district.PORTAL,
        "sheet": "Farm prices",
        "keys": {"state_code": "State", "district_code": "District", "year": "Year"},
        "crop_column": "Commodity",
        "values": {"price_inr_t": {"column": "Rupees", "unit": "INR/kg"}},
    }
    d = district.normalize("prices", spec, tmp_path)
    assert d[["state_code", "district_code", "year", "crop", "price_inr_t"]].iloc[0].tolist() == [
        "02",
        "007",
        2015,
        "Wheat",
        24000.0,
    ]


def test_real_crop_irrigation_geography_compares_only_complete_child_totals(tmp_path, monkeypatch):
    """Exercise physical comparison joins on published districts in an explicit test crosswalk."""
    from future_work.freeze import digest
    from future_work.storage import ROOT

    clean = ROOT / "data/clean/india"
    cross = pd.read_csv(clean / "district_match.csv", dtype={"dist_code": str, "poly_id": str}).query(
        "matched"
    )
    observed = pd.read_parquet(clean / "crops.parquet")
    r = observed.query('crop=="Rice" and year==2000').dropna(subset=["area_ha", "prod_t"])
    two = r.merge(cross[["dist_code", "poly_id"]], on="dist_code", validate="m:1").head(2)
    specfile = tmp_path / "members.csv"
    pd.DataFrame({"State": ["01", "01"], "Parent": ["01", "01"], "ADM": two.poly_id}).to_csv(
        specfile, index=False
    )
    spec = {
        "file": specfile.name,
        "source_url": district.PORTAL,
        "sha256": digest(specfile),
        "keys": {"state_code": "State", "district_code": "Parent"},
        "poly_id": "ADM",
    }
    official = pd.DataFrame(
        [
            {
                "dist_code": "01/01",
                "crop": "Rice at the presented stage",
                "year": 2000,
                "area_ha": two.area_ha.sum() + 10,
                "prod_t": two.prod_t.sum() - 5,
            }
        ]
    )
    manifest = {"geography": spec, "comparison_crops": {"DES": {"Rice": "Rice at the presented stage"}}}
    pair = district.comparable_quantities(official, None, manifest, tmp_path)
    assert len(pair) == 1
    assert pair.area_ha_difference.item() == pytest.approx(10)
    assert pair.prod_t_difference.item() == pytest.approx(-5)
    # A missing child's production must invalidate that total even when its area is available.
    read = pd.read_parquet

    def without_production(path, *args, **kwargs):
        x = read(path, *args, **kwargs)
        x.loc[
            (x.dist_code == two.dist_code.iloc[0]) & x.year.eq(2000) & x.crop.eq("Rice"), "prod_t"
        ] = np.nan
        return x

    monkeypatch.setattr(pd, "read_parquet", without_production)
    incomplete = district.comparable_quantities(official, None, manifest, tmp_path)
    assert (
        incomplete.prod_t_difference.isna().all()
        and incomplete.area_ha_difference.item() == pytest.approx(10)
    )
