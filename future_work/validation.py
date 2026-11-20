"""Independently recalculate figure values and audit sample coverage using the published tables."""

import json

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.sandwich_covariance import cov_cluster

from .storage import RESULT, ROOT


def equal(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10, equal_nan=True)


def explicit_effects(frame, outcome, terms):
    """Independent fit with explicit country/year indicators and country-clustered intervals."""
    countries = pd.get_dummies(frame.iso3, prefix="country", dtype=float)
    years = pd.get_dummies(frame.year, prefix="year", dtype=float, drop_first=True)
    matrix = pd.concat([frame[terms], countries, years], axis=1)
    fit = OLS(frame[outcome], matrix).fit()
    covariance = cov_cluster(fit, frame.iso3.astype("category").cat.codes, use_correction=False)
    n, groups = len(frame), len(countries.columns)
    # The original CR1 convention counts year effects and slopes; countries are nested in clusters.
    correction = groups / (groups - 1) * (n - 1) / (n - len(terms) - len(years.columns))
    beta = fit.params.loc[terms]
    se = np.sqrt(np.diagonal(covariance)[: len(terms)] * correction)
    margin = student_t.ppf(0.975, groups - 1) * se
    return pd.DataFrame(
        {
            "beta": beta,
            "se": se,
            "ci_lo": beta - margin,
            "ci_hi": beta + margin,
            "p": 2 * student_t.sf(abs(beta / se), groups - 1),
        }
    )


def check_crop_responses(matched):
    from src.metrics.crop_classes import ITEMS

    measures = ["december12", "harvest6", "season"]
    items = pd.read_parquet(
        ROOT / "data/clean/crops.parquet", columns=["item_code", "item"]
    ).drop_duplicates()
    items["mapspam_crop"] = items.item_code.map({code: members for code, (_, members) in ITEMS.items()})
    items = items.explode("mapspam_crop").dropna().drop(columns="item_code")
    keys = ["iso3", "item"]
    areas = pd.read_parquet(ROOT / "data/clean/mapspam_area.parquet").query("area_ha > 0")
    required = areas.merge(items, on="mapspam_crop").groupby(keys).mapspam_crop.nunique()
    obs = matched.merge(items, on="mapspam_crop").set_index(keys)
    observed = obs.groupby(level=keys).mapspam_crop.transform("nunique")
    obs = obs[observed.eq(required.reindex(obs.index))].reset_index()
    group = keys + ["year"]
    ha = obs.groupby(group).area_ha.sum()
    climate = (
        obs[measures].mul(obs.area_ha, axis=0).groupby([obs[k] for k in group]).sum().div(ha, axis=0)
    )
    yields = pd.read_parquet(ROOT / "outputs/crop_anomalies.parquet").dropna(subset=["yield_anom"])
    observations = yields.merge(climate, on=group, validate="1:1")
    slopes = pd.read_csv(RESULT / "season_crop_slopes.csv").set_index(keys)
    sizes = observations.groupby(keys).size()
    sizes = sizes[sizes >= 20]
    assert sizes.index.equals(slopes.index)
    equal(slopes.n_years, sizes)
    for key, g in observations.groupby(keys):
        if key not in sizes.index:
            continue
        x = g[measures] - g[measures].mean()
        b = x.mul(g.yield_anom - g.yield_anom.mean(), axis=0).sum() / x.pow(2).sum()
        equal(slopes.loc[key, measures], b)
        assert g.value_share.nunique() == 1
        equal(slopes.loc[key, "value_share"], g.value_share.iloc[0])
    diversity = pd.read_csv(RESULT / "season_response_diversity.csv").set_index("iso3")
    for iso, g in slopes.groupby(level="iso3"):
        assert diversity.loc[iso, "n_crops"] == len(g)
        for name in measures:
            good = g[g[name].notna() & g.value_share.gt(0)]
            expected = np.nan
            if len(good) >= 2:
                mean = np.average(good[name], weights=good.value_share)
                expected = np.sqrt(np.average((good[name] - mean) ** 2, weights=good.value_share))
            equal(diversity.loc[iso, name], expected)


def seasonal():
    key = ["iso3", "year"]
    ckey = ["iso3", "mapspam_crop", "year"]
    crops = pd.read_parquet(RESULT / "season_crop_year.parquet")
    years = pd.read_csv(RESULT / "season_national_year.csv").set_index(key).sort_index()
    assert not crops.duplicated(ckey).any() and (crops.groupby(ckey[:2]).size() == 31).all()
    assert crops.coverage.between(0, 1 + 1e-6).all()
    equal(crops.coverage, crops.observed_ha / crops.regime_area_ha)
    assert crops.season.notna().equals(crops.coverage.ge(0.9))
    matched = crops[crops.complete]
    for name in ["december12", "harvest6", "season"]:
        g = matched.assign(numerator=matched.area_ha * matched[name]).groupby(key)
        weighted = g.numerator.sum() / g.area_ha.sum()
        equal(years[name], weighted.reindex(years.index))
    baseline = (
        pd.read_parquet(ROOT / "outputs/metrics_national.parquet").set_index(key).loc[years.index]
    )
    for col in ["spei12_w", "index", "anom"]:
        equal(years[col], baseline[col])
    fig = pd.read_csv(RESULT / "figure_01_drought.csv").set_index("measure")
    loss_models = pd.read_csv(RESULT / "season_loss_panel.csv").set_index(["measure", "term"])
    diversity = pd.read_parquet(ROOT / "outputs/metrics.parquet").set_index("iso3").hill1
    frame = years.reset_index()
    country_z = (diversity - diversity.mean()) / diversity.std()
    frame["diversity"] = frame.iso3.map(country_z)
    for name, row in fig.iterrows():
        dry = years[name].le(-1)
        old = years.spei12_w.le(-1)
        assert (row.n_drought, row.newly_dry, row.no_longer_dry) == (
            dry.sum(),
            (dry & ~old).sum(),
            (~dry & old).sum(),
        )
        equal(
            [row.output_difference_pct_trend, row.lo_pct, row.hi_pct],
            100 * loss_models.loc[(name, "dry"), ["beta", "ci_lo", "ci_hi"]].to_numpy(dtype=float),
        )
        frame["dry"] = frame[name].le(-1).astype(float)
        frame["dry_x_hill1"] = frame.dry * frame.diversity
        check = explicit_effects(frame, "anom", ["dry", "dry_x_hill1"])
        equal(check, loss_models.loc[name].loc[check.index, check.columns])
    points = pd.read_csv(RESULT / "figure_02_responses.csv")
    faces = pd.read_csv(RESULT / "season_faces.csv")
    summary = faces.set_index(["iso3", "measure"])
    for iso, g in years.groupby(level="iso3"):
        values = np.lib.stride_tricks.sliding_window_view(g["index"].to_numpy(), 7)
        for name in fig.index:
            dry = g[name].to_numpy() <= -1
            windows = np.lib.stride_tricks.sliding_window_view(dry, 7)
            events = values[windows[:, 3] & ~windows[:, :3].any(axis=1)]
            resistance = np.median(events[:, 3] / events[:, :3].mean(axis=1)) if len(events) else np.nan
            recovery = np.median(events[:, 4:].mean(axis=1) / events[:, 3]) if len(events) else np.nan
            row = summary.loc[iso, name]
            equal(
                [row.n_dry, row.n_events, row.resistance, row.recovery],
                [dry.sum(), len(events), resistance, recovery],
            )
    check_crop_responses(matched)
    for label, col in [("Resistance", "resistance"), ("Recovery", "recovery")]:
        source = (
            faces.pivot(index="iso3", columns="measure", values=col)[["december12", "season"]]
            .dropna()
            .sort_index()
        )
        plotted = points[points.metric.eq(label)].set_index("iso3").sort_index()
        assert source.index.equals(plotted.index)
        equal(plotted[["december", "season"]], source)
    for label, filename, keys in [
        ("Crop yield response", "season_crop_slopes.csv", ["iso3", "item"]),
        ("Response diversity", "season_response_diversity.csv", ["iso3"]),
    ]:
        source = (
            pd.read_csv(RESULT / filename)
            .set_index(keys)[["december12", "season"]]
            .dropna()
            .sort_index()
        )
        plotted = points[points.metric.eq(label)].set_index(keys).sort_index()
        assert source.index.equals(plotted.index)
        equal(plotted[["december", "season"]], source)


def historical():
    data = pd.read_parquet(RESULT / "history_country_year.parquet")
    table = pd.read_csv(RESULT / "history_severity.csv")
    fig = pd.read_csv(RESULT / "figure_04_extremes.csv").set_index("model")
    assert (
        not data.duplicated(["iso3", "year"]).any() and data.groupby("iso3").year.nunique().eq(63).all()
    )
    long = fig.loc["eligible_1961_2023"]
    extreme = data.loc[data.spei.le(-2)]
    assert (long.n_countries, long.n_obs, long.n_extreme, long.n_exposed) == (
        data.iso3.nunique(),
        len(data),
        len(extreme),
        extreme.iso3.nunique(),
    )
    base = (
        pd.read_csv(ROOT / "outputs/stats/severity_panel.csv")
        .query('model=="bins_hill1"')
        .set_index("term")
    )
    found = table[table.model.eq("presented_1993_2023")].set_index("term")
    columns = ["beta", "se", "ci_lo", "ci_hi"]
    equal(found[columns], base.loc[found.index, columns])
    p = (
        pd.read_csv(ROOT / "outputs/stats/severity_robustness.csv")
        .query('model=="bins_hill1" and term=="extreme_x_hill1" and statistic=="p_perm_t"')
        .value.item()
    )
    equal(fig.loc["presented_1993_2023", "p_perm_t"], p)
    for model, row in fig.iterrows():
        src = table[table.model.eq(model) & table.term.eq("extreme_x_hill1")].iloc[0]
        equal(
            [row.effect_pp, row.lo_pp, row.hi_pp, row.mde_pp],
            100 * np.array([src.beta, src.ci_lo, src.ci_hi, src.mde_80]),
        )
    for name, mask in {
        "moderate": data.spei.le(-1) & data.spei.gt(-1.5),
        "severe": data.spei.le(-1.5) & data.spei.gt(-2),
        "extreme": data.spei.le(-2),
    }.items():
        data[name] = mask.astype(float)
    diversity = data.groupby("iso3").hill1.first()
    z = data.iso3.map((diversity - diversity.mean()) / diversity.std())
    bins = ["moderate", "severe", "extreme"]
    interactions = [f"{name}_x_hill1" for name in bins]
    data[interactions] = data[bins].mul(z, axis=0)
    check = explicit_effects(data, "anom", bins + interactions)
    published_long = table[table.model.eq("eligible_1961_2023")].set_index("term")
    equal(check, published_long.loc[check.index, check.columns])


def vegetation():
    rows = pd.read_parquet(RESULT / "vci_footprints_paired.parquet")
    keys = ["iso3", "year"]
    table = pd.read_csv(RESULT / "figure_03b_vegetation.csv").set_index(keys).sort_index()
    assert rows.groupby(["iso3", "mapspam_crop"]).year.nunique().eq(4).all()
    assert rows[["vci", "grid_vci"]].apply(lambda s: s.between(0, 100).all()).all()
    for field in ["vci", "grid_vci"]:
        g = rows.assign(value=rows[field] * rows.area_ha).groupby(keys)
        equal(table[field], (g.value.sum() / g.area_ha.sum()).loc[table.index])
    summary = pd.read_csv(RESULT / "vci_footprint_agreement.csv").query('level=="country-year"').iloc[0]
    equal(
        [summary.pearson, summary.mean_absolute_difference],
        [table.vci.corr(table.grid_vci), (table.vci - table.grid_vci).abs().mean()],
    )
    assert summary.n == len(table)


def temperature():
    from statsmodels.nonparametric.smoothers_lowess import lowess

    crops = pd.read_parquet(RESULT / "heat_crop_year.parquet")
    d = pd.read_csv(RESULT / "heat_country_year.csv", float_precision="round_trip")
    ckeys = ["iso3", "mapspam_crop"]
    keys = ["iso3", "year"]
    threshold = ["edd28", "edd30", "edd32"]
    assert not crops.duplicated(ckeys + ["year"]).any() and np.isfinite(crops[threshold]).all().all()
    assert ((crops.edd28 >= crops.edd30) & (crops.edd30 >= crops.edd32) & (crops.edd32 >= 0)).all()
    assert (
        crops.vci.dropna().between(0, 100).all() and d.vci_basket_area_share.between(0, 1 + 1e-10).all()
    )
    fixed = crops[crops.complete & crops.groupby(ckeys).vci.transform("count").gt(0)]
    nat = d.set_index(keys).sort_index()
    coverage = fixed.groupby(keys).vci.agg(["size", "count"])
    assert coverage.index.equals(nat.index) and nat.vci.notna().equals(
        coverage["size"].eq(coverage["count"])
    )
    audit = pd.read_csv(RESULT / "heat_sample_coverage.csv").set_index(keys).sort_index()
    assert audit.index.equals(nat.index)
    equal(audit.basket_crops, coverage["size"])
    equal(audit.observed_vci_crops, coverage["count"])
    observed_ha = fixed.area_ha.where(fixed.vci.notna(), 0).groupby([fixed[k] for k in keys]).sum()
    equal(audit.vci_basket_area_share, observed_ha / fixed.groupby(keys).area_ha.sum())
    assert audit.satellite_complete.equals(nat.vci.notna())
    for _, g in nat.groupby(level="iso3"):
        years = g.index.get_level_values("year").to_numpy()
        np.testing.assert_array_equal(years, np.arange(1993, 2024))
        for col in threshold:
            values = g[col].to_numpy()
            residual = values - lowess(values, years - years[0], frac=0.5, return_sorted=False)
            sd = residual.std(ddof=1)
            expected = residual / sd if np.ptp(values) > 0 and sd > 0 else np.full(len(g), np.nan)
            equal(g[col + "_z"], expected)
    assert audit.temperature_standardized.equals(nat[[m + "_z" for m in threshold]].notna().all(axis=1))
    all_types = pd.read_csv(RESULT / "heat_year_types.csv").set_index(keys)
    published_switches = pd.read_csv(RESULT / "heat_year_type_transitions.csv", index_col=0)
    recalculated = pd.crosstab(all_types.presented_CRU_Dec12, all_types.ERA5_30_season)
    pd.testing.assert_frame_equal(published_switches, recalculated, check_names=False)
    for col in [*threshold, "season", "vci"]:
        g = fixed.assign(value=fixed[col] * fixed.area_ha).groupby(keys)
        est = g.value.sum() / g.area_ha.sum()
        if col == "vci":
            est = est.where(coverage["size"].eq(coverage["count"]))
        equal(nat[col], est.reindex(nat.index))
    valid = d.dropna(
        subset=["vci", "anom", "heat_z_w", "spei12_w", "season", *[m + "_z" for m in threshold]]
    ).copy()
    model_index = valid.set_index(keys).index
    assert audit.index[audit.in_model].equals(model_index) and all_types.index.equals(model_index)
    all_effects = pd.read_csv(RESULT / "heat_effects.csv").set_index(["model", "outcome", "term"])
    chart = pd.read_csv(RESULT / "figure_03_heat.csv").set_index(["model", "outcome", "term"])
    baseline = (
        pd.read_csv(ROOT / "outputs/stats/heat_panel.csv")
        .query('model=="year_types"')
        .set_index("term")
    )
    for term in ["dry_only", "hot_only", "compound"]:
        r = chart.loc["presented_full_sample", "anom", term]
        equal(
            [r.beta_plot, r.ci_lo_plot, r.ci_hi_plot],
            100 * baseline.loc[term, ["beta", "ci_lo", "ci_hi"]].to_numpy(float),
        )
    for model in all_effects.index.get_level_values("model").unique():
        h = "heat_z_w" if model.startswith("presented") else "edd" + model.split("_")[1] + "_z"
        hot = valid[h].ge(1)
        dry = valid.season.le(-1) if model.endswith("_season") else valid.spei12_w.le(-1)
        terms = {"dry_only": dry & ~hot, "hot_only": hot & ~dry, "compound": dry & hot}
        flags = np.select(list(terms.values()), list(terms), default="neither")
        np.testing.assert_array_equal(all_types[model].to_numpy(), flags)
        design = valid.assign(**{t: v.astype(float) for t, v in terms.items()})
        for output, scale in [("anom", 100), ("vci", 1)]:
            check = explicit_effects(design, output, list(terms))
            for term, value in check.iterrows():
                ident = (model, output, term)
                effect = all_effects.loc[ident]
                assert (effect.n_obs, effect.n_countries, effect.n_exposed, effect.n_neither) == (
                    len(design),
                    design.iso3.nunique(),
                    terms[term].sum(),
                    (~dry & ~hot).sum(),
                )
                columns = ["beta", "ci_lo", "ci_hi"]
                values = value[columns].to_numpy(dtype=float)
                equal(effect[check.columns].to_numpy(dtype=float), value.to_numpy(dtype=float))
                if ident in chart.index:
                    equal(
                        chart.loc[ident, [c + "_plot" for c in columns]].to_numpy(dtype=float),
                        scale * values,
                    )


def district():
    status = json.loads((RESULT / "district_status.json").read_text())
    assert status["status"] in ["complete", "archived_complete", "awaiting_official_release"]
    b = pd.read_csv(RESULT / "district_presented_benchmarks.csv").set_index(["model", "term"])
    orig = pd.read_csv(ROOT / "outputs/stats/india_panel.csv").set_index(["model", "term"]).loc[b.index]
    equal(b[["beta", "se", "n_obs", "n_districts"]], orig[["beta", "se", "n_obs", "n_districts"]])
    if status["status"] in {"complete", "archived_complete"}:
        data = pd.read_parquet(RESULT / "district_crop_year.parquet")
        valid = data.irrigation_share.dropna()
        assert valid.between(0, 1).all()
        equal(data.value_const, data.prod_t * data.reference_price_inr_t)
        assert (RESULT / "district_comparison.csv").exists()
        if status["status"] == "archived_complete":
            from .district_validation import validate
            validate(status)
    else:
        assert not (RESULT / "district_comparison.csv").exists()


def validate_results():
    seasonal()
    historical()
    vegetation()
    temperature()
    district()
    print("Reconciled figure data, fixed crop baskets, uncertainty intervals and district status.")
