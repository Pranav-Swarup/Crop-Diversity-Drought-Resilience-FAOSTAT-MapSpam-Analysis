"""Correlation and regression tables -> outputs/stats/.

  spearman.csv       Spearman rho (bootstrap 95% CI, 2000 resamples) of each face vs hill1 and resp_div
  ols.csv            face ~ z(hill1) + z(resp_div) + z(log gdp_pc) + z(fert_kg_ha) + z(irrig_share), HC3
  r2_comparison.csv  R2 of hill1-only vs resp_div-only vs both, on the same countries
  summary.csv        sample sizes and metric medians

Run: python -m src.metrics.stats
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr

from ..common import paths
from .data import CONTROLS, FACES, PREDICTORS, analysis_table

N_BOOT = 2000
SEED = 20231231
MIN_N = 10
OUTCOMES = FACES + ["vuln_drought"]


def spearman_boot(x, y, n_boot=N_BOOT, seed=SEED):
    """Spearman rho with a percentile bootstrap 95% CI over country pairs. Returns dict."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, n = x[ok], y[ok], int(ok.sum())
    if n < MIN_N or np.ptp(x) == 0 or np.ptp(y) == 0:
        return {"rho": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "p": np.nan, "n": n}
    rho, p = spearmanr(x, y)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = np.array([spearmanr(x[i], y[i])[0] for i in idx])
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return {"rho": float(rho), "ci_lo": float(lo), "ci_hi": float(hi), "p": float(p), "n": n}


def zscore(df):
    return (df - df.mean()) / df.std(ddof=1)


def ols_z(df, outcome, terms):
    """OLS of z(outcome) on z(terms), complete cases, HC3 standard errors. Returns the fit or None."""
    d = df[[outcome] + terms].replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < max(MIN_N, len(terms) + 3) or (d.std(ddof=1) == 0).any():
        return None
    z = zscore(d)
    return sm.OLS(z[outcome], sm.add_constant(z[terms])).fit(cov_type="HC3")


def spearman_table(df):
    rows = [{"face": f, "predictor": p, **spearman_boot(df[p], df[f])} for f in OUTCOMES for p in PREDICTORS]
    return pd.DataFrame(rows)


def ols_table(df):
    terms = PREDICTORS + CONTROLS
    rows = []
    for f in OUTCOMES:
        fit = ols_z(df, f, terms)
        for t in terms:
            if fit is None:
                rows.append({"face": f, "term": t, "beta": np.nan, "se": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                             "p": np.nan, "n": int(df[[f] + terms].dropna().shape[0]), "r2": np.nan})
                continue
            lo, hi = fit.conf_int().loc[t]
            rows.append({"face": f, "term": t, "beta": fit.params[t], "se": fit.bse[t], "ci_lo": lo, "ci_hi": hi,
                         "p": fit.pvalues[t], "n": int(fit.nobs), "r2": fit.rsquared})
    return pd.DataFrame(rows)


def r2_table(df):
    rows = []
    for f in OUTCOMES:
        d = df[[f] + PREDICTORS].dropna()  # same countries for all three models
        row = {"face": f, "n": len(d)}
        for label, terms in [("hill1_only", ["hill1"]), ("resp_div_only", ["resp_div"]), ("both", PREDICTORS)]:
            fit = ols_z(d, f, terms)
            row[f"r2_{label}"] = fit.rsquared if fit is not None else np.nan
            row[f"adj_r2_{label}"] = fit.rsquared_adj if fit is not None else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def summary_table(df):
    rows = [("n_countries", len(df)), ("n_events", int(df["n_events"].sum())),
            ("n_countries_with_events", int((df["n_events"] > 0).sum())),
            ("n_countries_with_resp_div", int(df["resp_div"].notna().sum()))]
    rows += [(f"median_{c}", df[c].median()) for c in PREDICTORS + OUTCOMES + ["resp_divergence", "phi_sync", "mean_crop_cv"]]
    return pd.DataFrame(rows, columns=["quantity", "value"])


def main():
    df = analysis_table()
    tables = {"spearman": spearman_table(df), "ols": ols_table(df), "r2_comparison": r2_table(df),
              "summary": summary_table(df)}
    sp = tables["spearman"]
    assert not sp.duplicated(["face", "predictor"]).any() and len(sp) == len(OUTCOMES) * len(PREDICTORS)
    assert sp["rho"].dropna().between(-1, 1).all()
    assert (sp["ci_lo"].dropna() <= sp["rho"].dropna()).all() and (sp["rho"].dropna() <= sp["ci_hi"].dropna()).all()
    assert tables["ols"]["r2"].dropna().between(0, 1).all()
    paths.STATS_DIR.mkdir(parents=True, exist_ok=True)
    for name, t in tables.items():
        t.to_csv(paths.STATS_DIR / f"{name}.csv", index=False)

    pd.set_option("display.width", 200)
    print(f"countries: {len(df)}, events: {int(df['n_events'].sum())}")
    print("\nSpearman rho [bootstrap 95% CI]:")
    for r in sp.itertuples():
        print(f"  {r.face:<13} vs {r.predictor:<8} rho={r.rho:+.2f} [{r.ci_lo:+.2f}, {r.ci_hi:+.2f}]  p={r.p:.3f}  n={r.n}")
    print("\nOLS standardised betas (HC3):")
    print(tables["ols"].round(3).to_string(index=False))
    print("\nR2 comparison:")
    print(tables["r2_comparison"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
