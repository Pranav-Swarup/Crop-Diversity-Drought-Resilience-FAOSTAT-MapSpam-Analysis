"""Unit tests for exposure-weighted diversity. Run: python -m pytest src/metrics/tests/test_exposure_diversity.py"""
import numpy as np
import pandas as pd
import pytest

from src.common import paths
from src.metrics import crop_classes
from src.metrics.diversity import hill1
from src.metrics.exposure_diversity import YEARS, class_areas, class_series, compute

RNG = np.random.default_rng(7)


def crops_frame(areas, iso3="AAA"):
    """areas: dict item_code -> constant area_ha in every year."""
    rows = [{"iso3": iso3, "year": y, "item_code": k, "item": crop_classes.ITEMS[k][0], "area_ha": a,
             "value_const": 1.0, "is_imputed": False} for k, a in areas.items() for y in YEARS]
    return pd.DataFrame(rows)


def climate_frame(series, iso3="AAA"):
    """series: dict mapspam_crop -> array over YEARS. Returns long crop_climate frame."""
    return pd.concat([pd.DataFrame({"iso3": iso3, "year": YEARS, "mapspam_crop": c, "crop_spei12_w": v})
                      for c, v in series.items()], ignore_index=True)


def national(values, iso3="AAA"):
    return pd.DataFrame({"iso3": iso3, "year": YEARS, "spei12_w": values})


def test_class_areas_sum_items_within_a_class_and_average_over_years():
    # Oats + Rye -> OCER; Wheat alone; one year without a value is dropped (same filter as run.py)
    c = crops_frame({15: 100.0, 71: 30.0, 75: 20.0})
    c.loc[(c["item_code"] == 71) & (c["year"] == 2000), "value_const"] = np.nan
    a = class_areas(c)
    assert set(a.index) == {"WHEA", "OCER"}
    assert np.isclose(a["WHEA"], 100.0)
    assert np.isclose(a["OCER"], 20.0 + 30.0 * 30 / 31)


def test_merged_class_is_area_weighted_mean_of_members():
    wide = pd.DataFrame({"COFF": np.arange(31.0), "RCOF": np.zeros(31)}, index=YEARS)
    nat = pd.Series(np.ones(31), index=YEARS)
    s, fb, w = class_series("COFF+RCOF", wide, nat, {"COFF": 300.0, "RCOF": 100.0})
    assert not fb and np.isclose(w["COFF"], 0.75)
    assert np.allclose(s.to_numpy(), 0.75 * np.arange(31.0))
    s_eq, _, w_eq = class_series("COFF+RCOF", wide, nat, None)
    assert np.allclose(s_eq.to_numpy(), 0.5 * np.arange(31.0)) and np.isclose(w_eq["RCOF"], 0.5)
    # a member without a series is left out
    s1, _, w1 = class_series("COFF+RCOF", wide[["RCOF"]], nat, {"COFF": 300.0, "RCOF": 100.0})
    assert w1 == {"RCOF": 1.0} and np.allclose(s1.to_numpy(), 0.0)


def test_class_without_series_falls_back_to_national():
    nat = RNG.normal(size=31)
    crops = crops_frame({15: 60.0, 56: 40.0})                   # wheat has a series, maize does not
    cc = climate_frame({"WHEA": RNG.normal(size=31)})
    ed, detail = compute(crops, national(nat), cc, ["AAA"])
    assert np.isclose(ed.loc[0, "fallback_share"], 0.4)
    d = detail.set_index("exposure_class")
    assert d.loc["MAIZ", "fallback"] and not d.loc["WHEA", "fallback"]
    assert np.isclose(d.loc["MAIZ", "r_national"], 1.0)


def test_identical_exposure_gives_one_effective_exposure():
    s = RNG.normal(size=31)
    crops = crops_frame({15: 50.0, 56: 30.0, 27: 20.0})
    cc = climate_frame({"WHEA": s, "MAIZ": s, "RICE": s})
    ed, _ = compute(crops, national(s), cc, ["AAA"])
    r = ed.iloc[0]
    assert np.isclose(r["hill1_exp"], 1.0) and np.isclose(r["exp_corr"], 1.0) and np.isclose(r["hill2_exp"], 1.0)
    assert np.isclose(r["hill1_ms"], hill1([50, 30, 20]))
    assert np.isclose(r["exp_ratio"], 1 / r["hill1_ms"])


def test_uncorrelated_exposure_gives_close_to_plain_hill_number():
    a = RNG.normal(size=(31, 4))
    q, _ = np.linalg.qr(a - a.mean(axis=0))                     # centred orthogonal columns: correlation exactly 0
    crops = crops_frame({15: 40.0, 56: 30.0, 27: 20.0, 116: 10.0})
    cc = climate_frame({"WHEA": q[:, 0], "MAIZ": q[:, 1], "RICE": q[:, 2], "POTA": q[:, 3]})
    ed, _ = compute(crops, national(q[:, 0]), cc, ["AAA"])
    r = ed.iloc[0]
    assert abs(r["exp_corr"]) < 1e-9
    assert r["hill1_exp"] == pytest.approx(r["hill1_ms"], rel=1e-9)
    # partly correlated exposures sit strictly between 1 and the plain Hill number
    cc2 = climate_frame({"WHEA": q[:, 0], "MAIZ": q[:, 0] + q[:, 1], "RICE": q[:, 2], "POTA": q[:, 3]})
    r2 = compute(crops, national(q[:, 0]), cc2, ["AAA"])[0].iloc[0]
    assert 1 < r2["hill1_exp"] < r2["hill1_ms"]


def test_mapping_covers_every_item_in_crops():
    if not paths.CROPS.exists() or not paths.CROP_CLIMATE.exists():
        pytest.skip("real crops.parquet / crop_climate.parquet not present")
    crops = pd.read_parquet(paths.CROPS, columns=["item_code", "item"])
    codes = pd.read_parquet(paths.CROP_CLIMATE, columns=["mapspam_crop"])["mapspam_crop"].unique()
    assert crop_classes.check(crops, codes)
    assert set(crops["item_code"]) <= set(crop_classes.ITEMS)
