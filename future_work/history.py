"""Extend the published extreme-drought design to 1961–2023. Author: Pmjarvis."""

from zipfile import ZipFile

import netCDF4
import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.sparse import csr_matrix

from src.crops import crops as original
from src.crops.countries import CHINA_AGGREGATE, MERGED, ne_table
from src.crops.quality import (
    MAX_IMPUTED_VALUE_SHARE,
    MIN_AREA_HA,
    MIN_CROPS,
    MIN_GOOD_YEARS,
)
from src.metrics import severity
from src.metrics.detrend import detrend
from src.metrics.diversity import hill1

from .seasons import grid_cells, indices_for, spei_grid
from .storage import CACHE, RESULT, ROOT, spei_path

YEARS = np.arange(1961, 2024)
KEY = ["Area Code", "Item Code", "Year"]


def bulk(domain, codes):
    folder = ROOT / "data/raw/faostat"
    with ZipFile(next(folder.glob(domain + "*zip"))) as z:
        name = next(n for n in z.namelist() if n.endswith("_(Normalized).csv"))
        cols = KEY + ["Area Code (M49)", "Area", "Item", "Element Code", "Value", "Unit", "Flag"]
        pieces = []
        with z.open(name) as f:
            for ch in pd.read_csv(
                f, usecols=cols, chunksize=400000, dtype={"Area Code (M49)": str}, low_memory=False
            ):
                pieces.append(
                    ch.loc[
                        ch["Element Code"].isin(codes)
                        & ch.Year.between(1961, 2023)
                        & ch["Area Code"].lt(5000)
                    ]
                )
    return pd.concat(pieces, ignore_index=True)


def historical_crops():
    cache = CACHE / "historical_crops.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    q = bulk("Production_Crops_Livestock", [5312, 5510])
    qv = bulk("Value_of_Production", [152])
    crops = q.loc[q["Element Code"].eq(5312), ["Item Code", "Item"]].drop_duplicates()
    aggregates = crops.Item.str.contains(r"Total$|Primary$|, primary$|Equivalent$")
    if set(crops.loc[aggregates, "Item Code"]) != original.AGGREGATE_ITEMS:
        raise ValueError("FAOSTAT aggregate definition changed")
    keep = set(crops["Item Code"]) - original.AGGREGATE_ITEMS
    q = q[q["Item Code"].isin(keep)]
    area = q[q["Element Code"].eq(5312)]
    prod = q[q["Element Code"].eq(5510)]
    if set(area.Unit) != {"ha"} or set(prod.Unit) != {"t"} or set(qv.Unit) != {"1000 Int$"}:
        raise ValueError("Unexpected FAOSTAT units")
    v = qv[KEY + ["Value"]].rename(columns={"Value": "value_fao"})
    v.value_fao *= 1000.0
    df = (
        area[KEY + ["Area", "Area Code (M49)", "Value", "Flag"]]
        .rename(columns={"Value": "area_ha", "Flag": "area_flag"})
        .merge(
            prod[KEY + ["Value", "Flag"]].rename(columns={"Value": "prod_t", "Flag": "prod_flag"}),
            on=KEY,
            validate="1:1",
        )
    )
    df = df[df.area_ha.gt(0) & df.prod_t.notna()].copy()
    prices = original.item_prices(df.loc[df.Year.ge(1993), KEY + ["prod_t"]], v[v.Year.ge(1993)])
    df = df.merge(v, on=KEY, how="left", validate="1:1")
    df["value_const"] = df.value_fao.fillna(df.prod_t * df["Item Code"].map(prices))
    df["computed"] = df.value_fao.isna() & df.value_const.notna()
    df["unpriced"] = df.value_const.isna()
    df["is_imputed"] = df.area_flag.isin(original.IMPUTED_FLAGS) | df.prod_flag.isin(
        original.IMPUTED_FLAGS
    )
    df["m49"] = df["Area Code (M49)"].str.strip("'")
    df = df.merge(ne_table()[["m49", "iso3"]], on="m49", how="left", validate="m:1")
    df["iso3"] = df["Area Code"].map(MERGED).fillna(df.iso3)
    df.loc[df["Area Code"].eq(CHINA_AGGREGATE), "iso3"] = None
    valid = set(pd.read_csv(ROOT / "data/clean/countries.csv").iso3)
    df["mapped"] = df.iso3.isin(valid)
    loc = (
        df.groupby(["Area Code", "Area", "iso3", "mapped"], dropna=False)
        .agg(first=("Year", "min"), last=("Year", "max"))
        .reset_index()
    )
    loc.to_csv(RESULT / "history_territories.csv", index=False)
    df = df[df.mapped]
    for _, g in df[df.iso3.isin(set(MERGED.values()))].groupby(["iso3", "Year"]):
        codes = set(g["Area Code"])
        if codes.intersection({15, 186, 206}) and len(codes) > 1:
            raise ValueError("Parent-successor overlap")
    g = df.groupby(["iso3", "Year", "Item Code"], sort=True)
    out = g.agg(
        area_ha=("area_ha", "sum"),
        prod_t=("prod_t", "sum"),
        value_const=("value_const", "sum"),
        is_imputed=("is_imputed", "any"),
        computed=("computed", "any"),
        unpriced=("unpriced", "any"),
    ).reset_index()
    out.loc[out.unpriced, "value_const"] = np.nan
    out = out.rename(columns={"Year": "year", "Item Code": "item_code"})
    # Check the complete reconstructed overlap against the original crop table, including missingness.
    old = (
        pd.read_parquet(ROOT / "data/clean/crops.parquet")
        .set_index(["iso3", "year", "item_code"])
        .sort_index()
    )
    chk = out[out.year.ge(1993)].set_index(old.index.names).sort_index()
    if not old.index.equals(chk.index):
        raise ValueError("Historical reader disagrees on overlapping FAOSTAT keys")
    for col in ("area_ha", "prod_t", "value_const", "is_imputed"):
        np.testing.assert_allclose(chk[col], old[col], rtol=1e-12, atol=1e-8, equal_nan=True)
    out.to_parquet(cache, index=False)
    return out


def history_climate():
    cache = CACHE / "history_climate.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    g = grid_cells()
    codes, labels = pd.factorize(g.iso3, sort=True)
    w = (
        g.land_fraction.to_numpy()
        * np.load(ROOT / "data/raw/derived/cropland_05deg.npy").ravel()[g.cell]
    )
    mat = csr_matrix((w, (codes, g.cell)), shape=(len(labels), 360 * 720))
    with netCDF4.Dataset(spei_path(12)) as ds:
        x = spei_grid(ds, indices_for(ds, YEARS, 12))
    num = mat @ np.nan_to_num(x).T
    den = mat @ np.isfinite(x).T
    cover = den / np.asarray(mat.sum(axis=1))
    mean = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
    out = pd.DataFrame(
        {
            "iso3": np.repeat(labels, len(YEARS)),
            "year": np.tile(YEARS, len(labels)),
            "spei": mean.ravel(),
            "climate_coverage": cover.ravel(),
        }
    )
    old = pd.read_parquet(ROOT / "data/clean/climate.parquet")[["iso3", "year", "spei12_w"]]
    chk = out[out.year.ge(1993)].merge(old, on=["iso3", "year"], validate="1:1")
    np.testing.assert_allclose(chk.spei, chk.spei12_w, rtol=1e-9, atol=1e-9, equal_nan=True)
    out.to_parquet(cache, index=False)
    return out


def build_panel():
    c = historical_crops()
    g = c[c.value_const.notna()].copy()
    g["imputed_value"] = g.value_const.where(g.is_imputed, 0)
    g["computed_value"] = g.value_const.where(g.computed, 0)
    by = (
        g.groupby(["iso3", "year"])[["value_const", "area_ha", "imputed_value", "computed_value"]]
        .sum()
        .reset_index()
    )
    d = by.merge(history_climate(), on=["iso3", "year"], validate="1:1")
    crop_count = g.groupby("iso3").item_code.nunique()
    diversity = g.groupby(["iso3", "item_code"]).area_ha.sum().groupby("iso3").apply(hill1)
    old = set(pd.read_parquet(ROOT / "outputs/metrics.parquet").iso3)
    criteria = []
    panels = []
    for iso, v in d.groupby("iso3", sort=True):
        v = v.sort_values("year").copy()
        good = int(v.imputed_value.div(v.value_const).le(MAX_IMPUTED_VALUE_SHARE).sum())
        reasons = []
        if not np.array_equal(v.year, YEARS):
            reasons.append("incomplete observations on a mapped territory")
        if (v.spei.isna() | v.climate_coverage.lt(0.9)).any():
            reasons.append("insufficient climate coverage")
        if good < int(np.ceil(MIN_GOOD_YEARS * len(YEARS) / 31)):
            reasons.append("insufficient years meeting the reporting rule")
        if crop_count[iso] < MIN_CROPS or v.area_ha.mean() < MIN_AREA_HA:
            reasons.append("crop count or area below baseline rule")
        criteria.append(
            dict(
                iso3=iso,
                first_year=v.year.min(),
                last_year=v.year.max(),
                n_years=len(v),
                good_years=good,
                crops=crop_count[iso],
                mean_ha=v.area_ha.mean(),
                old_cohort=iso in old,
                eligible=not reasons,
                reason=" | ".join(reasons),
            )
        )
        if reasons:
            continue
        v["hill1"] = diversity[iso]
        v["anom"] = detrend(v.value_const, v.year)["anom"].to_numpy()
        if v.anom.isna().any():
            raise ValueError(f"Missing trend for {iso}")
        panels.append(v)
    pd.DataFrame(criteria).to_csv(RESULT / "history_eligibility.csv", index=False)
    panel = pd.concat(panels, ignore_index=True)
    panel.to_parquet(RESULT / "history_country_year.parquet", index=False)
    return panel


def detectable(se, dof):
    """Two-sided alpha .05, 80% power, conditional on the estimated clustered standard error."""
    crit = stats.t.ppf(0.975, dof)

    def gap(ncp):
        return stats.nct.sf(crit, dof, ncp) + stats.nct.cdf(-crit, dof, ncp) - 0.8

    return optimize.brentq(gap, 0, 5) * se


def permutation(d, fit):
    """Exact balanced-panel demeaning, checked against the original clustered estimator on real data."""
    d = d.sort_values(["iso3", "year"])
    G, T = d.iso3.nunique(), d.year.nunique()
    if len(d) != G * T or d.duplicated(["iso3", "year"]).any():
        raise ValueError("Expected a balanced panel")

    def within(a):
        return a - a.mean(axis=1, keepdims=True) - a.mean(axis=0, keepdims=True) + a.mean(axis=(0, 1))

    bins = d[severity.DROUGHT_BINS].to_numpy().reshape(G, T, 3)
    x0 = within(bins)
    yc = within(d.anom.to_numpy().reshape(G, T))
    z = d.groupby("iso3").hill1.first().to_numpy()
    z = (z - z.mean()) / z.std(ddof=1)
    N = G * T
    corr = G / (G - 1) * (N - 1) / (N - (6 + T - 1))

    def estimate(weights):
        x = np.concatenate([x0, within(bins * weights[:, None, None])], axis=2)
        xf = x.reshape(N, 6)
        bread = np.linalg.inv(xf.T @ xf)
        beta = bread @ xf.T @ yc.ravel()
        scores = np.einsum("gtk,gt->gk", x, yc - x @ beta)
        se = np.sqrt(corr * np.sum((scores @ bread[:, -1]) ** 2))
        return beta[-1], beta[-1] / se

    b, t = estimate(z)
    np.testing.assert_allclose(b, fit["params"][severity.FOCUS], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(
        b / t, np.sqrt(fit["cov"].loc[severity.FOCUS, severity.FOCUS]), rtol=1e-8
    )
    rng = np.random.default_rng(severity.SEED)
    draws = np.array([estimate(rng.permutation(z)) for _ in range(severity.N_PERM)])
    return tuple((1 + (np.abs(draws) >= np.abs([b, t])).sum(axis=0)) / (1 + len(draws)))


def analyze():
    hist = build_panel()
    old = (
        pd.read_parquet(ROOT / "outputs/metrics_national.parquet")
        .rename(columns={"spei12_w": "spei"})
        .merge(
            pd.read_parquet(ROOT / "outputs/metrics.parquet")[["iso3", "hill1"]],
            on="iso3",
            validate="m:1",
        )
    )
    common = set(old.iso3) & set(hist.iso3)
    samples = {
        "presented_1993_2023": old,
        "common_1993_2023": old[old.iso3.isin(common)],
        "common_1961_2023": hist[hist.iso3.isin(common)],
        "eligible_1961_2023": hist,
    }
    rows = []
    for label, d in samples.items():
        d = d.reset_index(drop=True)
        d = pd.concat([d, severity.drought_bins(d.spei)], axis=1)
        fit, design, scale = severity.fit_outcome(d, "anom", severity.DROUGHT_BINS, ["hill1"])
        table = severity.coef_table(
            label, fit, severity.derived_rows(fit, design, scale, severity.DROUGHT_BINS)
        )
        p_raw, p_t = permutation(d, fit)
        extreme = d[d.extreme.eq(1)]
        for row in table:
            row.update(n_extreme=len(extreme), n_exposed=extreme.iso3.nunique())
            if row["term"] == severity.FOCUS:
                row.update(mde_80=detectable(row["se"], fit["df_t"]), p_perm=p_raw, p_perm_t=p_t)
            rows.append(row)
    out = pd.DataFrame(rows)
    base = out[out.model.eq("presented_1993_2023")].set_index("term")
    official = (
        pd.read_csv(ROOT / "outputs/stats/severity_panel.csv")
        .query("model == 'bins_hill1'")
        .set_index("term")
        .loc[base.index]
    )
    np.testing.assert_allclose(
        base[["beta", "se", "ci_lo", "ci_hi"]], official[["beta", "se", "ci_lo", "ci_hi"]], atol=1e-10
    )
    ref = pd.read_csv(ROOT / "outputs/stats/severity_robustness.csv")
    perm = ref[
        (ref.model == "bins_hill1") & (ref.term == severity.FOCUS) & (ref.statistic == "p_perm_t")
    ].value.item()
    if (base.n_extreme.iloc[0], base.n_exposed.iloc[0]) != (58, 41) or abs(
        base.loc[severity.FOCUS, "p_perm_t"] - perm
    ) > 1e-12:
        raise ValueError("Frozen extreme-drought benchmark mismatch")
    out.to_csv(RESULT / "history_severity.csv", index=False)
    # Keep country membership fixed while separating period-specific diversity from the
    # LOWESS smoothing span. The baseline frac=0.5 covers roughly 15.5 years in 31.
    common_old = old[old.iso3.isin(common)]
    historical = hist[hist.iso3.isin(common)].copy()
    historical["frozen_hill1"] = historical.iso3.map(common_old.groupby("iso3").hill1.first())
    fixed_span = 0.5 * 31 / len(YEARS)
    by_iso = historical.groupby("iso3", sort=True)
    historical["fixed_span_anom"] = by_iso.value_const.transform(
        lambda s: detrend(s.to_numpy(), YEARS, frac=fixed_span)["anom"].to_numpy()
    )
    sensitivity = []
    for h in ["hill1", "frozen_hill1"]:
        for outcome in ["anom", "fixed_span_anom"]:
            d = historical.copy()
            d["hill1"] = d[h]
            d = pd.concat([d, severity.drought_bins(d.spei)], axis=1)
            fit, _, _ = severity.fit_outcome(d, outcome, severity.DROUGHT_BINS, ["hill1"])
            row = next(r for r in severity.panel.coef_rows(fit) if r["term"] == severity.FOCUS)
            sensitivity.append(
                dict(
                    diversity_window="1993–2023" if h == "frozen_hill1" else "1961–2023",
                    trend_span_years=15.5 if outcome == "fixed_span_anom" else 31.5,
                    **row,
                    n_countries=fit["n_clusters"],
                    n_obs=fit["nobs"],
                )
            )
    pd.DataFrame(sensitivity).to_csv(RESULT / "history_sensitivity.csv", index=False)
    print(out[out.term.eq(severity.FOCUS)].to_string(index=False), flush=True)


if __name__ == "__main__":
    analyze()
