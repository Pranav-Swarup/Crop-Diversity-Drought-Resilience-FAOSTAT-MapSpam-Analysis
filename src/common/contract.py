"""Data contract from CLAUDE.md: columns, dtypes and keys of every shared table.

    from src.common.contract import load, validate
    crops = load("crops")            # reads from paths (dummy or real) and validates
    validate(my_df, "climate")       # call at the end of every pipeline

`python -m src.common.contract` validates every contract file that exists.
"""
import json

import pandas as pd
from pandas.api import types as pdt

from . import paths

YEAR_MIN, YEAR_MAX = 1993, 2023
SEVERITIES = {"moderate", "severe", "extreme"}

CONTRACT = {
    "countries": {
        "path": paths.COUNTRIES,
        "keys": ["iso3"],
        "columns": {"iso3": str, "name": str, "region": str, "member_iso3": str},
    },
    "crops": {
        "path": paths.CROPS,
        "keys": ["iso3", "year", "item_code"],
        "columns": {
            "iso3": str, "year": int, "item_code": int, "item": str,
            "area_ha": float, "prod_t": float, "yield_t_ha": float,
            "value_const": float, "is_imputed": bool,
        },
    },
    "controls": {
        "path": paths.CONTROLS,
        "keys": ["iso3", "year"],
        "columns": {"iso3": str, "year": int, "gdp_pc": float, "fert_kg_ha": float, "irrig_share": float},
    },
    "quality": {
        "path": paths.QUALITY,
        "keys": ["iso3"],
        "columns": {"iso3": str, "n_crops": int, "share_imputed": float, "included": bool, "reason": str},
    },
    "climate": {
        "path": paths.CLIMATE,
        "keys": ["iso3", "year"],
        "columns": {"iso3": str, "year": int, "spei12_w": float, "spei12_unw": float, "spei6min_w": float},
    },
    "events": {
        "path": paths.EVENTS,
        "keys": ["iso3", "year"],
        "columns": {"iso3": str, "year": int, "spei12_w": float, "severity": str},
    },
    "metrics": {
        "path": paths.METRICS,
        "keys": ["iso3"],
        "columns": {
            "iso3": str, "hill1": float, "stability": float, "resistance": float,
            "recovery": float, "vulnerability": float, "vuln_drought": float,
            "n_events": int, "resp_div": float, "resp_divergence": float,
            "phi_sync": float, "mean_crop_cv": float,
        },
    },
    "crop_anomalies": {
        "path": paths.CROP_ANOMALIES,
        "keys": ["iso3", "year", "item"],
        "columns": {
            "iso3": str, "year": int, "item": str,
            "yield_anom": float, "value_share": float, "resp_beta": float,
        },
    },
}

# Inclusive (lo, hi) bounds; None = unbounded. NaN is ignored.
RANGES = {
    "area_ha": (0, None), "prod_t": (0, None), "yield_t_ha": (0, None), "value_const": (0, None),
    "gdp_pc": (0, None), "fert_kg_ha": (0, None), "irrig_share": (0, None),
    "n_crops": (0, None), "share_imputed": (0, 1),
    "hill1": (1, None), "stability": (0, None), "resistance": (0, None), "recovery": (0, None),
    "vulnerability": (0, 1), "vuln_drought": (0, 1), "n_events": (0, None),
    "resp_div": (0, None), "phi_sync": (0, 1), "mean_crop_cv": (0, None),
    "value_share": (0, 1),
}

_DTYPE_CHECKS = {
    str: lambda s: pdt.is_string_dtype(s) or pdt.is_object_dtype(s),
    int: lambda s: pdt.is_integer_dtype(s),
    float: lambda s: pdt.is_float_dtype(s),
    bool: lambda s: pdt.is_bool_dtype(s),
}


def validate(df, name):
    """Assert that df matches the contract for `name`. Returns df unchanged."""
    spec = CONTRACT[name]
    cols, keys = spec["columns"], spec["keys"]

    missing = [c for c in cols if c not in df.columns]
    extra = [c for c in df.columns if c not in cols]
    assert not missing and not extra, f"{name}: missing columns {missing}, unexpected columns {extra}"

    for col, kind in cols.items():
        assert _DTYPE_CHECKS[kind](df[col]), f"{name}.{col}: expected {kind.__name__}, got {df[col].dtype}"

    for k in keys:
        assert df[k].notna().all(), f"{name}.{k}: nulls in key column"
    dup = df.duplicated(keys, keep=False)
    assert not dup.any(), f"{name}: {dup.sum()} rows with duplicate keys {keys}\n{df.loc[dup, keys].head()}"

    bad_iso = df.loc[~df["iso3"].str.fullmatch(r"[A-Z]{3}"), "iso3"].unique()
    assert len(bad_iso) == 0, f"{name}.iso3: not 3 upper-case letters: {list(bad_iso)[:10]}"
    if "year" in cols:
        assert df["year"].between(YEAR_MIN, YEAR_MAX).all(), f"{name}.year: outside {YEAR_MIN}-{YEAR_MAX}"

    for col, (lo, hi) in RANGES.items():
        if col in cols:
            v = df[col].dropna()
            assert lo is None or (v >= lo).all(), f"{name}.{col}: values below {lo} (min {v.min()})"
            assert hi is None or (v <= hi).all(), f"{name}.{col}: values above {hi} (max {v.max()})"

    if name == "events":
        bad = set(df["severity"]) - SEVERITIES
        assert not bad, f"events.severity: unexpected values {bad}"
        assert (df["spei12_w"] <= -1.0).all(), "events.spei12_w: event with spei12_w > -1.0"
    return df


def validate_geojson(path=None):
    """Assert every feature has a unique 3-letter `iso3` property. Returns the list of iso3."""
    path = path or paths.COUNTRIES_GEOJSON
    with open(path) as f:
        gj = json.load(f)
    assert gj.get("type") == "FeatureCollection", "countries.geojson: not a FeatureCollection"
    iso = [ft.get("properties", {}).get("iso3") for ft in gj["features"]]
    assert all(isinstance(i, str) and len(i) == 3 and i.isupper() for i in iso), \
        "countries.geojson: feature without a valid iso3 property"
    assert len(iso) == len(set(iso)), "countries.geojson: duplicate iso3"
    assert all(ft.get("geometry") for ft in gj["features"]), "countries.geojson: feature without geometry"
    return iso


def read(name, path=None):
    """Read a contract table with the contract dtypes, without validating."""
    spec = CONTRACT[name]
    path = path or spec["path"]
    if str(path).endswith(".parquet"):
        return pd.read_parquet(path)
    dtypes = {c: k for c, k in spec["columns"].items() if k is not float}
    return pd.read_csv(path, dtype=dtypes)


def load(name, path=None):
    """Read a contract table and validate it."""
    return validate(read(name, path), name)


def summary(df, name):
    """One-line summary: rows, countries, years, % missing cells."""
    years = f"{df['year'].min()}-{df['year'].max()}" if "year" in df.columns else "-"
    pct_missing = 100 * df.isna().to_numpy().mean() if len(df) else 0.0
    return (f"{name:<15} rows={len(df):>6}  countries={df['iso3'].nunique():>3}  "
            f"years={years:<9}  missing={pct_missing:.1f}%")


def validate_all():
    """Validate every contract file that exists at the current paths. Returns the names checked."""
    print(f"USE_DUMMY={'1' if paths.USE_DUMMY else '0'}  clean={paths.CLEAN}  outputs={paths.OUT}")
    checked = []
    for name, spec in CONTRACT.items():
        if not spec["path"].exists():
            print(f"{name:<15} not found ({spec['path'].relative_to(paths.ROOT)})")
            continue
        print(summary(load(name), name) + "  OK")
        checked.append(name)
    if paths.COUNTRIES_GEOJSON.exists():
        print(f"{'geojson':<15} features={len(validate_geojson()):>3}  OK")
        checked.append("geojson")
    else:
        print(f"{'geojson':<15} not found ({paths.COUNTRIES_GEOJSON.relative_to(paths.ROOT)})")
    return checked


if __name__ == "__main__":
    validate_all()
