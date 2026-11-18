"""Within-country panel test: does the response of national crop value to SPEI-12 vary with crop diversity?

  outputs/stats/panel_fe.csv        model, term, beta, se, ci_lo, ci_hi, p, n_obs, n_countries, r2_within
  outputs/stats/panel_summary.csv   per model (and per slope variable): n_obs, n_countries, years, r2_within,
                                    marginal effect of SPEI on anom at the 10th / 50th / 90th percentile of hill1
  figures/results/05_panel_interaction.png   marginal effect of SPEI across the observed hill1 range, 95% band

Unit of observation: country-year (outputs/metrics_national.parquet). Outcome: `anom`, the LOWESS-detrended
relative anomaly of national crop value (value_const summed over crops). `spei` = spei12_w.
z(.) = z-score across the countries of each model's estimation sample (one value per country, ddof=1).
Every model has country and year fixed effects; standard errors are clustered by country.

Models (term names as in panel_fe.csv):
  core           anom_it = a_i + g_t + b1 spei_it + b2 spei_it z(hill1_i) + b3 spei_it z(resp_div_i) + e_it
  asymmetric     dry_it = min(spei_it, 0), wet_it = max(spei_it, 0);
                 anom_it = a_i + g_t + d1 dry + d2 dry z(hill1) + d3 dry z(resp_div)
                                     + w1 wet + w2 wet z(hill1) + w3 wet z(resp_div) + e_it
  controls       core + spei z(log_gdp_pc_i) + spei z(fert_kg_ha_i) + spei z(irrig_share_i)
                 (controls are country window means, as in src.metrics.data.analysis_table)
  hill1_only     a_i + g_t + b1 spei + b2 spei z(hill1)
  resp_div_only  a_i + g_t + b1 spei + b3 spei z(resp_div)
  hill1_tv_lag   a_i + g_t + b1 spei + c1 z(h_i,t-1) + c2 spei z(h_i,t-1) + b3 spei z(resp_div)
                 (terms z_hill1_lag, spei_x_hill1_lag; sample 1994-2023 because of the lag)
                 h_it = hill1 of year t's harvested-area shares; z over the pooled country-years of the sample
  even_years     core, fitted on even years only (reference sample for split_resp_div)
  split_resp_div a_i + g_t + b1 spei + b2 spei z(hill1) + b3 spei z(resp_div_odd), fitted on even years only;
                 resp_div_odd is computed by src.metrics.run.compute from odd-year crop rows only
  excl_imputed   core, dropping country-years where > 50% of national value_const comes from imputed rows

Identification: hill1_i, resp_div_i and the controls are time-invariant (window values), so their main effects
are absorbed by the country fixed effects; only their interactions with spei are identified. Year fixed effects
absorb shocks common to all countries in a year, so the spei terms are identified from within-country
deviations of spei relative to the cross-country pattern of that year.

resp_div and the outcome share data. resp_div is the value-weighted SD of per-crop OLS slopes of yield anomaly
on SPEI, estimated over 1993-2023, i.e. from the same yields and SPEI that make up anom_it and spei_it in the
core model. Sampling noise in the crop slopes inflates resp_div and is correlated with the noise in the
national anom-spei relation, so spei x z(resp_div) is partly mechanical. split_resp_div removes the shared
years: slopes and resp_div come only from odd years (crop yields detrended on odd years), the panel is fitted
on even years. Odd years give at most 16 observations per crop, below the 20 required by
response.MIN_YEARS, so for this variant MIN_YEARS is set to 10 (half the window rule, for half the years)
while run.compute runs. The national outcome anom is still detrended on all years (outputs/metrics_national).

Estimator: two-way fixed effects by Frisch-Waugh-Lovell. anom and the terms are residualised on country and
year dummies (exact also for unbalanced samples), then OLS without constant. r2_within = R2 of that
residualised regression (variance explained beyond both sets of fixed effects). Cluster-robust covariance
(CR1): V = c (X'X)^-1 [sum_g X_g' u_g u_g' X_g] (X'X)^-1, c = G/(G-1) (N-1)/(N-K), K = number of terms +
(number of years - 1); country dummies are nested in the clusters and not counted (reghdfe convention).
p-values and 95% CIs use t with G-1 degrees of freedom. Marginal effect of spei at hill1 = h:
b1 + b2 z(h), with the other interacted variables at their sample mean (z = 0); its SE is the delta method,
sqrt(V11 + z^2 V22 + 2 z V12), exact because the effect is linear in the coefficients. hill1 percentiles are
of the window hill1 across the countries of each model's sample.

Run: python -m src.metrics.panel     (after src.metrics.run; USE_DUMMY=1 for dummy inputs)
"""
from unittest import mock

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as st

from ..common import paths
from ..common.contract import load
from . import figures as figs
from . import response
from . import run as metrics_run
from .data import analysis_table
from .diversity import hill1

FE_FILE = "panel_fe.csv"
SUMMARY_FILE = "panel_summary.csv"
FIG_NAME = "05_panel_interaction.png"
SPLIT_MIN_YEARS = 10      # per-crop minimum for the odd-year slopes (see module docstring)
IMPUTED_MAX = 0.5         # excl_imputed: max share of national value from imputed rows
PCTS = (10, 50, 90)
MIN_COUNTRIES = 10
N_COUNTRIES_MAX = 107
COUNTRY_VARS = ["hill1", "resp_div", "log_gdp_pc", "fert_kg_ha", "irrig_share"]


# ---------------------------------------------------------------- estimator

def residualise(df, cols, fe=("iso3", "year")):
    """Residuals of each column in `cols` on dummies for every fixed effect in `fe` (FWL). Returns array."""
    dummies = [pd.get_dummies(df[f], drop_first=i > 0, dtype=float) for i, f in enumerate(fe)]
    d = pd.concat(dummies, axis=1).to_numpy()
    y = df[list(cols)].to_numpy(dtype=float)
    coef, *_ = np.linalg.lstsq(d, y, rcond=None)
    return y - d @ coef


def fe_fit(df, y, terms, cluster="iso3", fe=("iso3", "year")):
    """Two-way FE OLS of y on terms with CR1 cluster-robust covariance. Returns dict:
    params, cov (cluster), cov_naive (homoskedastic), nobs, n_clusters, r2_within, df_t (G - 1)."""
    terms = list(terms)
    r = residualise(df, [y] + terms, fe)
    yt, xt = r[:, 0], r[:, 1:]
    n, k = xt.shape
    bread = np.linalg.inv(xt.T @ xt)
    beta = bread @ xt.T @ yt
    u = yt - xt @ beta
    groups = df[cluster].to_numpy()
    g_codes, g_idx = np.unique(groups, return_inverse=True)
    n_g = len(g_codes)
    scores = np.zeros((n_g, k))
    np.add.at(scores, g_idx, xt * u[:, None])
    k_full = k + df[fe[1]].nunique() - 1 if len(fe) > 1 else k
    c = n_g / (n_g - 1) * (n - 1) / (n - k_full)
    cov = c * bread @ (scores.T @ scores) @ bread
    sigma2 = (u @ u) / (n - k - (n_g + (df[fe[1]].nunique() - 1 if len(fe) > 1 else 0)))
    cov_naive = sigma2 * bread
    sst = yt @ yt
    return {
        "params": pd.Series(beta, index=terms),
        "cov": pd.DataFrame(cov, index=terms, columns=terms),
        "cov_naive": pd.DataFrame(cov_naive, index=terms, columns=terms),
        "nobs": n, "n_clusters": n_g, "df_t": n_g - 1,
        "r2_within": float(1 - (u @ u) / sst) if sst > 0 else np.nan,
    }


def coef_rows(fit):
    """One dict per term: beta, se, ci_lo, ci_hi, p (t with G-1 df)."""
    se = np.sqrt(np.diag(fit["cov"].to_numpy()))
    tcrit = st.t.ppf(0.975, fit["df_t"])
    rows = []
    for t, b, s in zip(fit["params"].index, fit["params"].to_numpy(), se):
        p = 2 * st.t.sf(abs(b / s), fit["df_t"]) if s > 0 else np.nan
        rows.append({"term": t, "beta": b, "se": s, "ci_lo": b - tcrit * s, "ci_hi": b + tcrit * s, "p": p})
    return rows


def marginal_effect(fit, slope, inter, z):
    """Marginal effect of `slope` at z (array) of the variable in `inter`, and its delta-method SE."""
    z = np.asarray(z, dtype=float)
    b, v = fit["params"], fit["cov"]
    me = b[slope] + b[inter] * z
    var = v.loc[slope, slope] + z ** 2 * v.loc[inter, inter] + 2 * z * v.loc[slope, inter]
    return me, np.sqrt(var)


# ---------------------------------------------------------------- data

def annual_hill1(crops, iso3s):
    """hill1 of each country-year's harvested-area shares, and its one-year lag. Columns iso3, year, hill1_t,
    hill1_lag (lag taken on consecutive calendar years only)."""
    c = crops[crops["iso3"].isin(iso3s) & crops["value_const"].notna()]  # same row filter as run.country_metrics
    h = c.groupby(["iso3", "year"])["area_ha"].apply(hill1).rename("hill1_t").reset_index()
    prev = h.assign(year=h["year"] + 1).rename(columns={"hill1_t": "hill1_lag"})
    return h.merge(prev, on=["iso3", "year"], how="left")


def imputed_value_share(crops, iso3s):
    """Share of each country-year's value_const that comes from rows with is_imputed. Columns iso3, year, share."""
    c = crops[crops["iso3"].isin(iso3s) & crops["value_const"].notna()]
    tot = c.groupby(["iso3", "year"])["value_const"].sum()
    imp = c[c["is_imputed"]].groupby(["iso3", "year"])["value_const"].sum().reindex(tot.index, fill_value=0.0)
    return (imp / tot).rename("imputed_share").reset_index()


def split_resp_div():
    """resp_div from odd-year crop rows only, through src.metrics.run.compute with response.MIN_YEARS set to
    SPLIT_MIN_YEARS for the duration of the call. Returns DataFrame iso3, resp_div_odd."""
    crops, climate, crop_climate, events, included = metrics_run.load_inputs()
    odd = crops[crops["year"] % 2 == 1]
    with mock.patch.object(response, "MIN_YEARS", SPLIT_MIN_YEARS):
        out = metrics_run.compute(odd, climate, crop_climate, events, included)
    return out["metrics"][["iso3", "resp_div"]].rename(columns={"resp_div": "resp_div_odd"})


def load_panel():
    """Country-year frame: iso3, year, anom, spei, dry, wet, window-level country variables, hill1_t,
    hill1_lag, imputed_share."""
    nat = pd.read_parquet(paths.NATIONAL)
    expect = {"iso3", "year", "anom", "spei12_w"}
    assert expect <= set(nat.columns), f"metrics_national.parquet lacks {expect - set(nat.columns)}"
    assert not nat.duplicated(["iso3", "year"]).any(), "metrics_national.parquet has duplicate iso3-year rows"
    country = analysis_table()[["iso3"] + COUNTRY_VARS]
    assert set(nat["iso3"]) == set(country["iso3"]), "metrics_national and metrics.parquet cover different countries"
    crops = load("crops")
    iso3s = nat["iso3"].unique()
    df = nat[["iso3", "year", "anom", "spei12_w"]].rename(columns={"spei12_w": "spei"})
    df = df.merge(country, on="iso3", how="left") \
           .merge(annual_hill1(crops, iso3s), on=["iso3", "year"], how="left") \
           .merge(imputed_value_share(crops, iso3s), on=["iso3", "year"], how="left")
    df["dry"] = np.minimum(df["spei"], 0.0)
    df["wet"] = np.maximum(df["spei"], 0.0)
    return df


# ---------------------------------------------------------------- models

def zc(df, col):
    """z-score of a country-level column across the countries in df. Returns (Series aligned to df, mean, sd)."""
    per = df.groupby("iso3")[col].first()
    mu, sd = per.mean(), per.std(ddof=1)
    return (df[col] - mu) / sd, mu, sd


def design(df, slopes, inter, main=(), pooled=()):
    """Add interaction terms `<slope>_x_<var>` for every slope x var. z-scores for country-level vars use one
    value per country; vars in `pooled` (time-varying) use the pooled rows. `main` vars enter as z main
    effects. Returns (df with terms, list of terms, dict var -> (mean, sd))."""
    df = df.copy()
    scale = {}
    for v in set(inter) | set(main):
        if v in pooled:
            mu, sd = df[v].mean(), df[v].std(ddof=1)
            df[f"z_{v}"] = (df[v] - mu) / sd
        else:
            df[f"z_{v}"], mu, sd = zc(df, v)
        scale[v] = (mu, sd)
    terms = list(slopes) + [f"z_{v}" for v in main]
    for s in slopes:
        for v in inter:
            name = f"{s}_x_{v}"
            df[name] = df[s] * df[f"z_{v}"]
            terms.append(name)
    return df, terms, scale


def model_specs(df, resp_odd):
    """dict name -> (sample DataFrame, slopes, interacted vars, main-effect vars, pooled vars, hill1 var)."""
    even = df[df["year"] % 2 == 0]
    split = even.merge(resp_odd, on="iso3", how="left") if resp_odd is not None else None
    specs = {
        "core": (df, ["spei"], ["hill1", "resp_div"], [], [], "hill1"),
        "asymmetric": (df, ["dry", "wet"], ["hill1", "resp_div"], [], [], "hill1"),
        "controls": (df, ["spei"], ["hill1", "resp_div", "log_gdp_pc", "fert_kg_ha", "irrig_share"], [], [], "hill1"),
        "hill1_only": (df, ["spei"], ["hill1"], [], [], "hill1"),
        "resp_div_only": (df, ["spei"], ["resp_div"], [], [], None),
        "hill1_tv_lag": (df, ["spei"], ["hill1_lag", "resp_div"], ["hill1_lag"], ["hill1_lag"], "hill1_lag"),
        "even_years": (even, ["spei"], ["hill1", "resp_div"], [], [], "hill1"),
        "split_resp_div": (split, ["spei"], ["hill1", "resp_div_odd"], [], [], "hill1"),
        "excl_imputed": (df[df["imputed_share"] <= IMPUTED_MAX], ["spei"], ["hill1", "resp_div"], [], [], "hill1"),
    }
    return {k: v for k, v in specs.items() if v[0] is not None}


def fit_model(sample, slopes, inter, main, pooled):
    """Complete cases, z-scores within the sample, FE fit. Returns (fit, design df, terms, scale) or None."""
    need = ["iso3", "year", "anom"] + list(slopes) + list(set(inter) | set(main)) + (["hill1"] if "hill1" not in inter else [])
    d = sample[need].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    if d["iso3"].nunique() < MIN_COUNTRIES:
        return None
    d, terms, scale = design(d, slopes, inter, main, pooled)
    return fe_fit(d, "anom", terms), d, terms, scale


def marginal_rows(name, fit, d, slopes, hvar, scale):
    """Summary rows: marginal effect of each slope variable at hill1 percentiles of the sample countries."""
    h = d.groupby("iso3")["hill1"].first()
    qs = np.percentile(h, PCTS)
    rows = []
    for s in slopes:
        row = {"model": name, "slope": s}
        for q, pct in zip(qs, PCTS):
            row[f"hill1_p{pct}"] = q
            if hvar is None:
                row[f"me_p{pct}"], row[f"se_p{pct}"] = np.nan, np.nan
                continue
            mu, sd = scale[hvar]
            me, se = marginal_effect(fit, s, f"{s}_x_{hvar}", (q - mu) / sd)
            row[f"me_p{pct}"], row[f"se_p{pct}"] = float(me), float(se)
        rows.append(row)
    return rows


def run_models(df, resp_odd):
    """Fit every model. Returns (coefficient table, summary table, dict of fits for the figure)."""
    coef, summ, fits = [], [], {}
    for name, (sample, slopes, inter, main, pooled, hvar) in model_specs(df, resp_odd).items():
        res = fit_model(sample, slopes, inter, main, pooled)
        if res is None:
            print(f"skipped {name}: fewer than {MIN_COUNTRIES} countries with complete data")
            continue
        fit, d, terms, scale = res
        fits[name] = (fit, d, scale)
        meta = {"n_obs": fit["nobs"], "n_countries": fit["n_clusters"], "r2_within": fit["r2_within"]}
        coef += [{"model": name, **r, **meta} for r in coef_rows(fit)]
        years = f"{d['year'].min()}-{d['year'].max()}"
        for r in marginal_rows(name, fit, d, slopes, hvar, scale):
            summ.append({**r, **meta, "years": years, "n_years": d["year"].nunique()})
    coef = pd.DataFrame(coef)[["model", "term", "beta", "se", "ci_lo", "ci_hi", "p", "n_obs", "n_countries", "r2_within"]]
    lead = ["model", "slope", "n_obs", "n_countries", "years", "n_years", "r2_within"]
    summ = pd.DataFrame(summ)
    summ = summ[lead + [c for c in summ.columns if c not in lead]]
    return coef, summ, fits


# ---------------------------------------------------------------- figure

def fig_interaction(fits):
    """Marginal effect of SPEI on anom (x100) across the observed hill1 range: core (left), asymmetric (right)."""
    c = plt.get_cmap("tab10").colors
    fig, axes = plt.subplots(1, 2, figsize=figs.FIGSIZE, sharey=True)
    panels = [("core", [("spei", c[0], "SPEI-12 (all years)")]),
              ("asymmetric", [("dry", c[1], "Dry side: min(SPEI-12, 0)"), ("wet", c[0], "Wet side: max(SPEI-12, 0)")])]
    n = 0
    for ax, (name, lines) in zip(axes, panels):
        if name not in fits:
            ax.set_title(f"{name}: not estimated", fontsize=9, color=figs.INK, loc="left")
            continue
        fit, d, scale = fits[name]
        n = max(n, fit["n_clusters"])
        h = d.groupby("iso3")["hill1"].first()
        grid = np.linspace(h.min(), h.max(), 200)
        tcrit = st.t.ppf(0.975, fit["df_t"])
        mu, sd = scale["hill1"]
        for slope, color, label in lines:
            me, se = marginal_effect(fit, slope, f"{slope}_x_hill1", (grid - mu) / sd)
            ax.fill_between(grid, 100 * (me - tcrit * se), 100 * (me + tcrit * se), color=color, alpha=0.18, linewidth=0)
            ax.plot(grid, 100 * me, color=color, linewidth=1.8, label=label)
        ax.plot(h, np.zeros_like(h), "|", color=figs.MUTED, markersize=6, alpha=0.6)
        ax.axhline(0, color=figs.MUTED, linewidth=0.8)
        ax.set_xlabel("Crop diversity, hill1 (effective number of crops)")
        ax.set_title(f"{name} model: n = {fit['n_clusters']} countries, {fit['nobs']} country-years",
                     fontsize=8, color=figs.INK, loc="left")
        ax.legend(frameon=False, fontsize=7, loc="upper right")
    axes[0].set_ylabel("Change in national value anomaly per +1 SPEI-12\n(% of trend value), 95% CI")
    fig.suptitle("Marginal effect of SPEI-12 on the national crop value anomaly across crop diversity\n"
                 "Country and year fixed effects, SEs clustered by country, resp_div held at its sample mean; "
                 "ticks = countries",
                 fontsize=9, color=figs.INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    figs.save(fig, FIG_NAME, n)


# ---------------------------------------------------------------- main

def main():
    df = load_panel()
    resp_odd = split_resp_div()
    coef, summ, fits = run_models(df, resp_odd)

    assert not coef.duplicated(["model", "term"]).any(), "duplicate model/term rows"
    assert np.isfinite(coef["beta"]).all() and np.isfinite(coef["se"]).all(), "non-finite beta or se"
    assert (coef["se"] > 0).all()
    assert coef["r2_within"].between(0, 1).all() and summ["r2_within"].between(0, 1).all()
    assert (coef["n_countries"] <= N_COUNTRIES_MAX).all() and (coef["n_countries"] >= MIN_COUNTRIES).all()
    assert (coef["ci_lo"] <= coef["beta"]).all() and (coef["beta"] <= coef["ci_hi"]).all()
    assert coef["p"].between(0, 1).all()
    assert not summ.duplicated(["model", "slope"]).any()
    assert set(summ["model"]) == set(coef["model"])
    assert "core" in fits, "core model not estimated"

    paths.STATS_DIR.mkdir(parents=True, exist_ok=True)
    coef.to_csv(paths.STATS_DIR / FE_FILE, index=False)
    summ.to_csv(paths.STATS_DIR / SUMMARY_FILE, index=False)
    print(f"wrote {(paths.STATS_DIR / FE_FILE).relative_to(paths.ROOT)}, {(paths.STATS_DIR / SUMMARY_FILE).relative_to(paths.ROOT)}")
    fig_interaction(fits)

    pd.set_option("display.width", 220)
    print(f"\npanel: {len(df)} country-years, {df['iso3'].nunique()} countries, "
          f"{df['year'].min()}-{df['year'].max()}; resp_div_odd finite for {int(resp_odd['resp_div_odd'].notna().sum())} countries")
    print("\ncoefficients (anom in fractions of trend; spei in SPEI units; interacted vars as z-scores):")
    print(coef.round(4).to_string(index=False))
    print("\nmarginal effect of the slope variable at hill1 percentiles:")
    print(summ.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
