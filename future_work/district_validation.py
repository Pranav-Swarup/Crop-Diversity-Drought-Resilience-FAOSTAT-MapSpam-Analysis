"""Independent checks of the archived district observations, spatial exposure, prices and models. Author: Pmjarvis."""

import json

import geopandas as gpd
import netCDF4
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from shapely.geometry import box
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.sandwich_covariance import cov_cluster

from .district_archive import ARCHIVE, PINS
from .district_land_use import FIELDS as LAND_USE_FIELDS
from .district_land_use import PAGES, verify_export
from .freeze import digest
from .storage import RESULT, ROOT, spei_path


def equal(a, b):
    np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-10, equal_nan=True)


def reestimate(p, saved, irrig=False):
    d = p.dropna(subset=["anom", "spei", "hill1_tv"]).copy()
    d["z"] = (d.hill1_tv - d.hill1_tv.mean()) / d.hill1_tv.std(ddof=1)
    d["spei_z"] = d.spei * d.z
    terms = ["spei", "spei_z", "z"]
    if irrig:
        d = d.dropna(subset=["irrig_lag"]).copy()
    if "irrig_lag" in set(saved.term):
        terms += ["irrig_lag", "spei_irrig_lag"]
        d["spei_irrig_lag"] = d.spei * d.irrig_lag
    # State-year effects and districts are nested in states. Project separately within each
    # state, then use statsmodels' cluster score covariance with the presented CR1 correction.
    matrices, clusters = [], []
    for _, g in d.groupby("state_name"):
        fe = np.column_stack(
            [
                pd.get_dummies(g.dist_code, dtype=float),
                pd.get_dummies(g.year, drop_first=True, dtype=float),
            ]
        )
        yx = g[["anom", *terms]].to_numpy()
        matrices.append(yx - fe @ np.linalg.lstsq(fe, yx, rcond=None)[0])
        clusters.extend(g.dist_code)
    yx = np.vstack(matrices)
    fitted = OLS(yx[:, 0], yx[:, 1:]).fit()
    n, k = len(d), len(terms)
    g = d.dist_code.nunique()
    k += d[["state_name", "year"]].drop_duplicates().shape[0] - 1
    cov = (
        cov_cluster(fitted, np.array(clusters), use_correction=False)
        * (g / (g - 1))
        * ((n - 1) / (n - k))
    )
    se = np.sqrt(np.diag(cov))
    b = fitted.params
    crit = student_t.ppf(0.975, g - 1)
    expect = saved.set_index("term").loc[terms]
    assert expect.n_obs.eq(n).all() and expect.n_districts.eq(g).all()
    equal(expect.beta, b)
    equal(expect.se, se)
    equal(expect.ci_lo, b - crit * se)
    equal(expect.ci_hi, b + crit * se)
    equal(expect.p, 2 * student_t.sf(np.abs(b / se), g - 1))


def check_window(crops, panel):
    years = sorted(panel.year.unique())
    selected = crops[crops.dist_code.isin(panel.dist_code) & crops.year.isin(years)]
    indexed = panel.set_index(["dist_code", "year"]).sort_index()
    areas = selected.pivot(index=["dist_code", "year"], columns="crop", values="area_ha")
    values = selected.assign(v=selected.value_const.where(selected.area_ha.notna()))
    values = values.pivot(index=["dist_code", "year"], columns="crop", values="v")
    ratios = selected.pivot(index=["dist_code", "year"], columns="crop", values="irrigation_share")
    invalid = selected.pivot(
        index=["dist_code", "year"], columns="crop", values="irrigation_invalid"
    )
    for code, v in values.groupby(level="dist_code"):
        v = v.droplevel("dist_code").reindex(years)
        keep = v.notna().all() & v.gt(0).any()
        v = v.loc[:, keep]
        assert v.shape[1] >= 3
        out = v.sum(axis=1).to_numpy()
        trend = lowess(out, years, frac=0.5, return_sorted=False)
        p = indexed.loc[code]
        equal(p["index"], out / trend)
        equal(p.anom, (out - trend) / trend)
        area = areas.loc[code].reindex(years)
        for length, col in [(5, "hill1_tv"), (3, "hill1_tv3")]:
            pooled = area.rolling(length, min_periods=length).sum().shift()
            shares = pooled.div(pooled.sum(axis=1).replace(0, np.nan), axis=0)
            shares = shares.where(shares > 0)
            entropy = -(shares * np.log(shares)).sum(axis=1, min_count=1)
            equal(p[col], np.exp(entropy))
        a = area[v.columns]
        r = ratios.loc[code, v.columns].reindex(years)
        missing = (a.gt(0) & r.isna()) | invalid.loc[code, v.columns].reindex(years).eq(True)
        ratio = (a * r).sum(axis=1) / a.sum(axis=1).replace(0, np.nan)
        equal(p.irrig_lag, ratio.mask(missing.any(axis=1)).shift())


def verify_native(crop, codebook, status):
    pins = json.loads(PINS.read_text())
    assert pins == status["source_files"]
    for role, spec in pins.items():
        path = ARCHIVE / spec["file"]
        if path.exists() and digest(path) != spec["sha256"]:
            raise ValueError(f"District source digest differs for {role}")
    land = status["external_land_use"]
    pages = pd.read_csv(PAGES)
    assert pages.offset.tolist() == [0, *pages.rows.cumsum().iloc[:-1].tolist()]
    assert pages.canonical_sha256.str.fullmatch("[0-9a-f]{64}").all()
    assert land["rows"] == land["datastore_records_checked"] == pages.rows.sum()
    assert land["datastore_fields_checked"] == len(LAND_USE_FIELDS) * land["rows"]
    export = ARCHIVE / pins["des_irrigation"]["file"]
    if export.is_file():
        assert verify_export(export) == land["rows"]
    fields = {
        "apy": [("area_ha_reported", " AREA", 1000), ("prod_t", " PRODUCTION", 1000)],
        "prices": [("price_inr_t", "price_field", 10)],
        "irrigation": [("irrigated_ha", "irrigation_field", 1000)],
    }
    for role, measures in fields.items():
        path = ARCHIVE / pins[role]["file"]
        if not path.exists():
            continue
        data = json.loads(path.read_bytes())
        source = pd.DataFrame(data["data"], columns=[h["header"] for h in data["headers"]])
        source["Year"] = source["Year"].astype(int)
        source = source.set_index(["State Code", "Dist Code", "Year"])
        for name, rows in crop.groupby("crop"):
            index = pd.MultiIndex.from_frame(rows[["state_code", "district_code", "year"]])
            for out, column, factor in measures:
                column = name + column if role == "apy" else codebook.loc[name, column]
                if pd.isna(column):
                    assert rows[out].isna().all()
                    continue
                numbers = pd.to_numeric(source[column]).reindex(index)
                numbers = numbers.mask(numbers.eq(-1) | (numbers.eq(0) & (role == "prices")))
                equal(rows[out], numbers * factor)


def verify_climate(climate):
    map_path = ARCHIVE / "boundaries.geojson"
    if not map_path.exists():
        return
    geo = gpd.read_file(map_path)
    hectares = np.load(ROOT / "data/raw/derived/cropland_05deg.npy").reshape(360, 720)
    with netCDF4.Dataset(spei_path(12)) as ds:
        y, x = ds["lat"][:], ds["lon"][:]
        dates = netCDF4.num2date(
            ds["time"][:], ds["time"].units, calendar=getattr(ds["time"], "calendar", "standard")
        )
        dec = {t.year: i for i, t in enumerate(dates) if t.month == 12}
        for code in ["14/1", "10/152", "19/661"]:
            geom = geo[geo.ID.astype(str).eq(code.split("/")[1])].geometry.union_all()
            lo, bot, hi, top = geom.bounds
            weights, coords = [], []
            for i in np.where((x >= lo - 0.25) & (x <= hi + 0.25))[0]:
                for j in np.where((y >= bot - 0.25) & (y <= top + 0.25))[0]:
                    rect = box(x[i] - 0.25, y[j] - 0.25, x[i] + 0.25, y[j] + 0.25)
                    w = (
                        geom.intersection(rect).area
                        / rect.area
                        * hectares[round((89.75 - y[j]) * 2), round((x[i] + 179.75) * 2)]
                    )
                    if w > 0:
                        weights.append(w)
                        coords.append((j, i))
            for year in [1966, 1993, 2017]:
                sample = np.ma.filled(ds["spei"][dec[year]], np.nan)
                values = np.array([sample[j, i] for j, i in coords])
                ok = np.isfinite(values)
                w = np.asarray(weights)
                observed = climate[climate.dist_code.eq(code) & climate.year.eq(year)].iloc[0]
                np.testing.assert_allclose(
                    observed.spei12_w, np.average(values[ok], weights=w[ok]), atol=3e-6
                )
                np.testing.assert_allclose(
                    observed.climate_coverage, w[ok].sum() / w.sum(), atol=3e-6
                )


def validate(status):
    crop = pd.read_parquet(RESULT / "district_crop_year.parquet")
    assert not crop.duplicated(["dist_code", "year", "crop"]).any()
    definitions = pd.read_csv(RESULT / "district_crop_definitions.csv").set_index("crop")
    verify_native(crop, definitions, status)
    clim = pd.read_parquet(RESULT / "district_climate.parquet")
    assert len(clim) == status["mapped_apy_districts"] * 52
    verify_climate(clim)
    assert set(crop.dist_code) == set(clim.dist_code)
    rounded = crop.area_ha_reported.eq(0) & (crop.prod_t > 0)
    assert crop.area_below_reporting_precision.eq(rounded).all()
    equal(crop.area_ha, crop.area_ha_reported.mask(rounded))
    fixed = crop[crop.year.eq(2015)].set_index(["dist_code", "crop"]).price_inr_t
    expected_price = fixed.reindex(pd.MultiIndex.from_frame(crop[["dist_code", "crop"]])).to_numpy()
    equal(crop.reference_price_inr_t_reported, expected_price)
    equal(
        crop.reference_price_inr_t, np.where(crop.state_name.eq("Orissa"), np.nan, expected_price)
    )
    equal(
        crop.irrigation_share,
        (crop.irrigated_ha / crop.area_ha.where(crop.area_ha.gt(0))).mask(
            crop.irrigation_invalid | crop.irrigation_manual_conflict
        ),
    )
    audit = pd.read_csv(RESULT / "district_source_reconciliation.csv")
    assert audit.cells.sum() == status["source_cells_checked"] and audit.different.eq(0).all()
    cross = pd.read_csv(RESULT / "district_boundary_comparison.csv")
    assert cross.included.eq(
        ~cross.history_change & ~cross.ambiguous_name & cross.intersection_over_union.ge(0.8)
    ).all()
    pairs = pd.read_parquet(RESULT / "district_paired_quantities.parquet")
    assert set(pairs.dist_code) <= set(cross[cross.included].dist_code)
    assert (
        pairs[~pairs.crop.isin(["SUGARCANE", "COTTON"]) | pairs.measure.ne("production (t)")].shape
        == pairs.shape
    )
    equal(pairs.difference, pairs.archived - pairs.reference)
    agreement = pd.read_csv(RESULT / "district_quantity_agreement.csv")
    for row in agreement.itertuples():
        d = pairs[pairs.source.eq(row.source) & pairs.measure.eq(row.measure)]
        if row.crop != "All comparable crops":
            d = d[d.crop.eq(row.crop)]
        valid = d.dropna(subset=["reference", "archived"])
        positive = valid[valid[["reference", "archived"]].max(axis=1).gt(0)]
        assert (len(d), len(valid), len(positive)) == (
            row.candidate_rows,
            row.observed_pairs,
            row.positive_pairs,
        )
        assert (
            int(positive.difference.abs().le(np.maximum(positive.reference * 0.01, 10)).sum())
            == row.within_tolerance
        )
    stats = pd.read_csv(RESULT / "district_comparison.csv")
    coverage = pd.read_csv(RESULT / "district_coverage.csv")
    dropped = pd.read_csv(RESULT / "district_exclusions.csv")
    for tag, suffix in [("archive", ""), ("overlap", "_overlap")]:
        p = pd.read_parquet(RESULT / f"district_panel{suffix}.parquet")
        book = coverage[coverage.window.eq(tag)]
        assert set(book[book.eligible].dist_code) == set(p.dist_code)
        reasons = dropped[dropped.window.eq(tag)].set_index("dist_code")
        assert set(book[~book.eligible].dist_code) == set(reasons.index)
        source = crop[crop.year.isin(p.year.unique()) & crop.dist_code.isin(reasons.index)]
        reference_counts = (
            source[source.reference_price_inr_t.notna()].groupby("dist_code").crop.nunique()
        )
        equal(
            reasons.crops_with_reference_price,
            reference_counts.reindex(reasons.index, fill_value=0),
        )
        prices_missing = reasons.reason.eq("no usable 2015 district-crop price")
        assert reasons.crops_with_reference_price.eq(0).equals(prices_missing)
        assert (
            reasons.loc[reasons.reason.str.startswith("fewer than three"), "balanced_crops"]
            .lt(3)
            .all()
        )
        low_share = reasons[reasons.reason.str.startswith("complete crops cover")]
        assert (
            low_share.balanced_crops.ge(3).all()
            and low_share.balanced_share_of_priced_value.lt(0.7).all()
        )
        years = set(p.year)
        for code, g in source.groupby("dist_code"):
            missing = ", ".join(map(str, sorted(years - set(g.year))))
            recorded = reasons.at[code, "missing_apy_start_years"]
            assert missing == ("" if pd.isna(recorded) else recorded)
            if reasons.at[code, "reason"].startswith("output index undefined"):
                values = (
                    g.dropna(subset=["area_ha"])
                    .pivot(index="year", columns="crop", values="value_const")
                    .reindex(sorted(years))
                    .dropna(axis=1)
                )
                totals = values.sum(axis=1)
                trend = lowess(totals, totals.index, frac=0.5, return_sorted=False)
                assert (~np.isfinite(trend) | (trend <= 0)).any()
        check_window(crop, p)
        for model in [tag, tag + "_irrigation_sample", tag + "_crop_irrigation"]:
            reestimate(p, stats[stats.model.eq(model)], model != tag)
    rates = pd.read_csv(RESULT / "district_price_basket.csv")
    pp = pd.read_parquet(RESULT / "district_price_panel.parquet")
    ps = pd.read_csv(RESULT / "district_price_sensitivity.csv")
    selected = crop.merge(rates, on=["dist_code", "crop"], validate="m:1")
    for name, group in ps.groupby("model"):
        raw = (
            (selected.prod_t * selected[name])
            .groupby([selected.dist_code, selected.year])
            .sum()
            .sort_index()
        )
        equal(pp.set_index(["dist_code", "year"])[name].sort_index(), raw)
        reestimate(pp.assign(anom=pp["anom_" + name]), group)
    raw_panel = pd.read_parquet(RESULT / "district_manual_sensitivity_panel.parquet")
    raw_stats = pd.read_csv(RESULT / "district_manual_sensitivity.csv")
    for name, group in raw_stats.groupby("model"):
        reestimate(raw_panel, group, name != "as_printed")
