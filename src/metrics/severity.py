"""Does the crop-diversity effect on the national value anomaly change with drought severity, and do crops fail
together more in severe years? Country-year panel, window 1993-2023, included countries.

Outputs (outputs/stats/):
  severity_panel.csv       model, term, beta, se, ci_lo, ci_hi, p, n_obs, n_countries, r2_within
  severity_counts.csv      kind, bin, hill1_tercile, n_country_years, n_countries
  severity_robustness.csv  check, model, term, statistic, value, label, n_obs, n_countries
  severity_breadth.csv     model, term, beta, se, ci_lo, ci_hi, p, n_obs, n_countries, r2_within
  figures/results/12_severity.png

Severity bins on spei = spei12_w (CLAUDE.md classes, exclusive 0/1 dummies; reference = every other year):
  moderate  -1.5 < spei <= -1.0      severe  -2.0 < spei <= -1.5      extreme  spei <= -2.0
  sev_plus  spei <= -1.5 (pooled severe or worse; model bins_pooled_hill1, with moderate as above)
Heat bins on heat = heat_z_w (data/clean/heat.parquet; hot year = heat >= 1.0):
  hot_mod  1.0 <= heat < 1.5        hot_sev  1.5 <= heat < 2.0        hot_ext  heat >= 2.0

Estimator: src.metrics.panel.design + panel.fe_fit (country and year fixed effects by FWL, CR1 covariance
clustered by country, t with G-1 df). z(v) = z-score of the country-level window value of v across the
countries of each model's estimation sample (panel.zc). Interaction terms are named `<bin>_x_<var>`.

Models in severity_panel.csv (outcome anom = relative anomaly of national crop value unless stated):
  bins               anom ~ moderate + severe + extreme
  bins_hill1         bins + each bin x z(hill1)
  curve              dry = min(spei, 0), dry_sq = dry^2: anom ~ dry + dry_sq + dry x z(hill1) + dry_sq x z(hill1)
  bins_controls      bins_hill1 + each bin x z(log_gdp_pc) + each bin x z(irrig_share)
                     (window means of the controls; countries without them are dropped)
  loss10             linear probability model, outcome loss10 = 1[index < 0.9]: bins + each bin x z(hill1)
  heat_bins          anom ~ hot_mod + hot_sev + hot_ext + each heat bin x z(hill1)
  bins_pooled_hill1  anom ~ moderate + sev_plus + each x z(hill1)
Derived rows (linear combinations, SE by the delta method on the clustered covariance, src.metrics.heat.linear_combo):
  <b>_minus_<b0>_x_hill1   (<b>_x_hill1) - (<b0>_x_hill1), b0 = the mildest bin of the model
  <b>_at_p10, <b>_at_p90   <b> + <b>_x_hill1 * z(h), h = 10th / 90th percentile of hill1 across the sample
                           countries (other interacted variables at z = 0, i.e. their sample mean). For anom it
                           is the anomaly in a bin-year relative to the country's other years (fraction of trend);
                           for loss10 the change in the probability of a >10% loss.

severity_robustness.csv (model bins_hill1 unless stated):
  permutation       hill1 permuted across countries (each country keeps its whole series; only the country ->
                    hill1 mapping is shuffled), model refitted, N_PERM permutations, seed SEED. Two-sided
                    p = (1 + #{|s_perm| >= |s_obs|}) / (1 + N_PERM), for s = the estimate (p_perm) and for
                    s = its clustered t-statistic (p_perm_t, studentised). Also for bins_pooled_hill1.
  loco / loyo       leave one country / one year out: refit, z(hill1) recomputed on the remaining countries;
                    statistics min, max, and the country / year whose removal moves extreme_x_hill1 most.
  wild_bootstrap    wild cluster bootstrap (Rademacher weights per country, N_BOOT draws, seed SEED) on the
                    FWL-residualised regression; bootstrap y* is re-residualised on both fixed effects before
                    the CR1 t-statistic is recomputed. p_wcr: null b = 0 imposed (restricted residuals);
                    ci_lo / ci_hi: unrestricted bootstrap-t interval b - q(t*) se.
  pooled_cutoff     sev_plus_x_hill1 from bins_pooled_hill1 (beta, se, p).

Crop breadth and co-movement (severity_breadth.csv), from outputs/crop_anomalies.parquet (crops that pass the
resp_beta rules; yield_anom = LOWESS relative yield anomaly, non-imputed years only). Per country-year, crops
with a finite yield_anom; country-years with fewer than MIN_CROPS crops are dropped; w_i = value_share_i
renormalised over the crops present that year.
  breadth_t = sum_i w_i 1[yield_anom_it < 0]                        (value-weighted share of crops below trend)
  z_it      = yield_anom_it / SD_i, SD_i = SD (ddof=1) of crop i's yield_anom over the window in that country
  comove_t  = [(sum_i w_i z_it)^2 - sum_i w_i^2 z_it^2] / [1 - sum_i w_i^2]
            = sum_{i!=j} w_i w_j z_it z_jt / sum_{i!=j} w_i w_j     (weighted mean pairwise product)
  Models: breadth_bins, breadth_bins_hill1, comove_bins, comove_bins_hill1 (outcome on the drought bins, and
  bins + bins x z(hill1)), country and year fixed effects, clustered by country.

Needs outputs/metrics_national.parquet, outputs/metrics.parquet, outputs/crop_anomalies.parquet,
data/clean/heat.parquet.

Run: python -m src.metrics.severity      (real data only)
"""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..common import paths
from ..common.contract import load
from . import panel
from .events import DROUGHT_THR
from .figures import DPI, FIGSIZE, INK, MUTED
from .heat import HOT_THR, linear_combo

SEVERE_THR, EXTREME_THR = -1.5, -2.0
HOT_SEV_THR, HOT_EXT_THR = 1.5, 2.0
DROUGHT_BINS = ["moderate", "severe", "extreme"]
HEAT_BINS = ["hot_mod", "hot_sev", "hot_ext"]
POOLED_BINS = ["moderate", "sev_plus"]
BIN_LABELS = {"moderate": "Moderate\n−1.5 < SPEI ≤ −1.0", "severe": "Severe\n−2.0 < SPEI ≤ −1.5",
              "extreme": "Extreme\nSPEI ≤ −2.0"}
LOSS_THR = 0.9
MIN_CROPS = 5
PCTS = (10, 90)
N_PERM, N_BOOT, SEED = 2000, 9999, 20240611
FOCUS = "extreme_x_hill1"
COLS = ["model", "term", "beta", "se", "ci_lo", "ci_hi", "p", "n_obs", "n_countries", "r2_within"]
ROB_COLS = ["check", "model", "term", "statistic", "value", "label", "n_obs", "n_countries"]
PANEL_FILE, COUNTS_FILE = "severity_panel.csv", "severity_counts.csv"
ROB_FILE, BREADTH_FILE = "severity_robustness.csv", "severity_breadth.csv"
FIG_NAME = "12_severity.png"
N_COUNTRIES = 107


# ---------------------------------------------------------------- bins

def drought_bins(spei):
    """Exclusive 0/1 dummies moderate, severe, extreme, sev_plus (= severe or extreme) from SPEI-12.
    NaN spei -> NaN dummies. Returns DataFrame aligned to spei."""
    s = pd.Series(spei, dtype=float)
    ok = s.notna()
    out = pd.DataFrame({
        "moderate": (s <= DROUGHT_THR) & (s > SEVERE_THR),
        "severe": (s <= SEVERE_THR) & (s > EXTREME_THR),
        "extreme": s <= EXTREME_THR,
        "sev_plus": s <= SEVERE_THR,
    }, index=s.index)
    return out.astype(float).where(ok, np.nan)


def heat_bins(heat):
    """Exclusive 0/1 dummies hot_mod, hot_sev, hot_ext from heat_z_w. NaN heat -> NaN dummies."""
    h = pd.Series(heat, dtype=float)
    out = pd.DataFrame({
        "hot_mod": (h >= HOT_THR) & (h < HOT_SEV_THR),
        "hot_sev": (h >= HOT_SEV_THR) & (h < HOT_EXT_THR),
        "hot_ext": h >= HOT_EXT_THR,
    }, index=h.index)
    return out.astype(float).where(h.notna(), np.nan)


# ---------------------------------------------------------------- crop breadth

def crop_breadth(ca, min_crops=MIN_CROPS):
    """Per country-year breadth and comove (module docstring). Returns iso3, year, n_crops, breadth, comove."""
    c = ca[np.isfinite(ca["yield_anom"])].copy()
    c["sd"] = c.groupby(["iso3", "item"])["yield_anom"].transform(lambda x: x.std(ddof=1))
    c = c[np.isfinite(c["sd"]) & (c["sd"] > 0)]
    c["n_crops"] = c.groupby(["iso3", "year"])["item"].transform("size")
    c = c[c["n_crops"] >= min_crops].copy()
    c["w"] = c["value_share"] / c.groupby(["iso3", "year"])["value_share"].transform("sum")
    c["z"] = c["yield_anom"] / c["sd"]
    c["wz"], c["w2z2"], c["w2"] = c["w"] * c["z"], (c["w"] * c["z"]) ** 2, c["w"] ** 2
    c["wneg"] = c["w"] * (c["yield_anom"] < 0)
    g = c.groupby(["iso3", "year"])
    s = g[["wz", "w2z2", "w2", "wneg"]].sum()
    out = pd.DataFrame({"n_crops": g.size(), "breadth": s["wneg"],
                        "comove": (s["wz"] ** 2 - s["w2z2"]) / (1 - s["w2"])}).reset_index()
    return out


def comove_bruteforce(w, z):
    """Weighted mean of z_i z_j over ordered pairs i != j with weights w_i w_j (w renormalised to sum 1)."""
    w = np.asarray(w, dtype=float) / np.sum(w)
    z = np.asarray(z, dtype=float)
    num = den = 0.0
    for i in range(len(w)):
        for j in range(len(w)):
            if i != j:
                num += w[i] * w[j] * z[i] * z[j]
                den += w[i] * w[j]
    return num / den


# ---------------------------------------------------------------- estimation

def fit_outcome(sample, y, slopes, inter):
    """panel.fit_model for an arbitrary outcome column y: complete cases, z-scores across the sample countries
    (panel.design), two-way FE fit with clustered SE (panel.fe_fit). Returns (fit, design df, scale) or None."""
    need = ["iso3", "year", y] + list(slopes) + list(dict.fromkeys(list(inter) + ["hill1"]))
    d = sample[need].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    if d["iso3"].nunique() < panel.MIN_COUNTRIES:
        return None
    d, terms, scale = panel.design(d, slopes, inter)
    return panel.fe_fit(d, y, terms), d, scale


def hill1_pcts(d):
    """10th and 90th percentile of the country-level hill1 across the countries of a design frame."""
    return np.percentile(d.groupby("iso3")["hill1"].first(), PCTS)


def derived_rows(fit, d, scale, bins):
    """Derived linear combinations for a bins x z(hill1) model (module docstring). Returns list of dicts."""
    rows = []
    base = bins[0]
    for b in bins[1:]:
        rows.append({"term": f"{b}_minus_{base}_x_hill1",
                     **linear_combo(fit, {f"{b}_x_hill1": 1.0, f"{base}_x_hill1": -1.0})})
    mu, sd = scale["hill1"]
    for q, pct in zip(hill1_pcts(d), PCTS):
        zq = (q - mu) / sd
        for b in bins:
            rows.append({"term": f"{b}_at_p{pct}", **linear_combo(fit, {b: 1.0, f"{b}_x_hill1": zq})})
    return rows


def coef_table(name, fit, extra=()):
    meta = {"n_obs": fit["nobs"], "n_countries": fit["n_clusters"], "r2_within": fit["r2_within"]}
    return [{"model": name, **r, **meta} for r in list(panel.coef_rows(fit)) + list(extra)]


def panel_specs(df):
    """name -> (outcome, slopes, interacted vars, bins for derived rows or None)."""
    return {
        "bins": ("anom", DROUGHT_BINS, [], None),
        "bins_hill1": ("anom", DROUGHT_BINS, ["hill1"], DROUGHT_BINS),
        "curve": ("anom", ["dry", "dry_sq"], ["hill1"], None),
        "bins_controls": ("anom", DROUGHT_BINS, ["hill1", "log_gdp_pc", "irrig_share"], DROUGHT_BINS),
        "loss10": ("loss10", DROUGHT_BINS, ["hill1"], DROUGHT_BINS),
        "heat_bins": ("anom", HEAT_BINS, ["hill1"], HEAT_BINS),
        "bins_pooled_hill1": ("anom", POOLED_BINS, ["hill1"], POOLED_BINS),
    }


def run_models(df, specs):
    """Fit every spec on df. Returns (coefficient table, dict name -> (fit, design df, scale))."""
    rows, fits = [], {}
    for name, (y, slopes, inter, bins) in specs.items():
        res = fit_outcome(df, y, slopes, inter)
        assert res is not None, f"{name}: fewer than {panel.MIN_COUNTRIES} countries"
        fit, d, scale = res
        fits[name] = res
        rows += coef_table(name, fit, derived_rows(fit, d, scale, bins) if bins else ())
    return pd.DataFrame(rows)[COLS], fits


# ---------------------------------------------------------------- robustness

def permute_country(df, col, rng):
    """Copy of df with the country-level column `col` shuffled across countries: each country's rows keep
    every other column; all rows of a country get the same permuted value."""
    per = df.groupby("iso3")[col].first()
    shuffled = pd.Series(rng.permutation(per.to_numpy()), index=per.index)
    out = df.copy()
    out[col] = out["iso3"].map(shuffled)
    return out


def interaction_stats(fit, bins):
    """For b_x_hill1 of each bin and (b - b0)_x_hill1 of the non-base bins: the estimate (key) and its
    clustered t-statistic (key + '|t'), as a dict."""
    combos = {f"{b}_x_hill1": {f"{b}_x_hill1": 1.0} for b in bins}
    combos.update({f"{b}_minus_{bins[0]}_x_hill1": {f"{b}_x_hill1": 1.0, f"{bins[0]}_x_hill1": -1.0}
                   for b in bins[1:]})
    out = {}
    for k, w in combos.items():
        r = linear_combo(fit, w)
        out[k], out[k + "|t"] = r["beta"], r["beta"] / r["se"]
    return out


def permutation_test(df, slopes, n_perm=N_PERM, seed=SEED, y="anom"):
    """Permutation p-values of the hill1 interaction statistics (interaction_stats), for the estimate and for
    its clustered t (studentised). Returns (observed dict, DataFrame of permuted statistics, p dict)."""
    obs = interaction_stats(fit_outcome(df, y, slopes, ["hill1"])[0], slopes)
    rng = np.random.default_rng(seed)
    draws = [interaction_stats(fit_outcome(permute_country(df, "hill1", rng), y, slopes, ["hill1"])[0], slopes)
             for _ in range(n_perm)]
    draws = pd.DataFrame(draws)
    p = {k: (1 + int((draws[k].abs() >= abs(v)).sum())) / (1 + n_perm) for k, v in obs.items()}
    return obs, draws, p


def leave_one_out(df, by, slopes=DROUGHT_BINS, term=FOCUS, y="anom"):
    """Refit bins_hill1 leaving out each value of `by` in turn. Returns Series value -> term estimate."""
    est = {}
    for v in sorted(df[by].unique()):
        fit = fit_outcome(df[df[by] != v], y, slopes, ["hill1"])[0]
        est[v] = fit["params"][term]
    return pd.Series(est)


def _fe_basis(d, fe=("iso3", "year")):
    """Orthonormal basis of the fixed-effect dummies used by panel.residualise."""
    dummies = [pd.get_dummies(d[f], drop_first=i > 0, dtype=float) for i, f in enumerate(fe)]
    q, _ = np.linalg.qr(pd.concat(dummies, axis=1).to_numpy())
    return q


def wild_cluster_bootstrap(d, y, terms, test, n_boot=N_BOOT, seed=SEED, chunk=500):
    """Wild cluster bootstrap for coefficient `test` (module docstring). Returns dict beta, se, p_wcr,
    ci_lo, ci_hi, n_boot. se is the CR1 SE of panel.fe_fit (asserted)."""
    terms = list(terms)
    r = panel.residualise(d, [y] + terms)
    yt, x = r[:, 0], r[:, 1:]
    n, k = x.shape
    j = terms.index(test)
    q = _fe_basis(d)
    g_codes, g_idx = np.unique(d["iso3"].to_numpy(), return_inverse=True)
    n_g = len(g_codes)
    agg = np.zeros((n_g, n))
    agg[g_idx, np.arange(n)] = 1.0
    c = n_g / (n_g - 1) * (n - 1) / (n - (k + d["year"].nunique() - 1))
    bread = np.linalg.inv(x.T @ x)
    a = x @ bread[j]                     # beta_j = a' y

    def t_se(ystar):                     # ystar: n x B, already in the residualised space
        bj = a @ ystar
        u = ystar - x @ (bread @ (x.T @ ystar))
        se = np.sqrt(c * ((agg @ (a[:, None] * u)) ** 2).sum(axis=0))
        return bj, se

    b_obs, se_obs = (v[0] for v in t_se(yt[:, None]))
    ref = panel.fe_fit(d, y, terms)
    assert np.isclose(b_obs, ref["params"][test]) and np.isclose(se_obs, np.sqrt(ref["cov"].loc[test, test])), \
        "bootstrap does not reproduce fe_fit"

    keep = [i for i in range(k) if i != j]
    xr = x[:, keep]
    fit_r = xr @ np.linalg.lstsq(xr, yt, rcond=None)[0]
    u_r = yt - fit_r
    fit_u = x @ (bread @ (x.T @ yt))
    u_u = yt - fit_u
    rng = np.random.default_rng(seed)
    t_wcr, t_wcu = [], []
    for start in range(0, n_boot, chunk):
        m = min(chunk, n_boot - start)
        v = rng.choice([-1.0, 1.0], size=(n_g, m))[g_idx]
        for fitted, u, centre, store in ((fit_r, u_r, 0.0, t_wcr), (fit_u, u_u, b_obs, t_wcu)):
            e = u[:, None] * v
            ystar = fitted[:, None] + e - q @ (q.T @ e)     # re-residualise the bootstrap error on both FE
            bj, se = t_se(ystar)
            store.append((bj - centre) / se)
    t_wcr, t_wcu = np.concatenate(t_wcr), np.concatenate(t_wcu)
    t_obs = b_obs / se_obs
    lo_q, hi_q = np.quantile(t_wcu, [0.025, 0.975])
    return {"beta": b_obs, "se": se_obs, "p_wcr": float(np.mean(np.abs(t_wcr) >= abs(t_obs))),
            "ci_lo": b_obs - hi_q * se_obs, "ci_hi": b_obs - lo_q * se_obs, "n_boot": n_boot}


def robustness(df, coef, fits):
    """severity_robustness.csv rows (module docstring)."""
    rows = []
    fit, d, _ = fits["bins_hill1"]
    meta = {"model": "bins_hill1", "n_obs": fit["nobs"], "n_countries": fit["n_clusters"]}

    t0 = time.time()
    for model, slopes in (("bins_hill1", DROUGHT_BINS), ("bins_pooled_hill1", POOLED_BINS)):
        obs, draws, p = permutation_test(df, slopes)
        assert len(draws) == N_PERM and draws.notna().all().all()
        f = fits[model][0]
        lab = f"{N_PERM} permutations of hill1 across countries, seed {SEED}"
        for k in (k for k in obs if not k.endswith("|t")):
            for stat, val, note in (("p_perm", p[k], lab + "; statistic = estimate"),
                                    ("p_perm_t", p[k + "|t"], lab + "; statistic = clustered t"),
                                    ("observed", obs[k], "estimate"), ("observed_t", obs[k + "|t"], "clustered t")):
                rows.append({"check": "permutation", "model": model, "term": k, "statistic": stat, "value": val,
                             "label": note, "n_obs": f["nobs"], "n_countries": f["n_clusters"]})
    print(f"permutations done in {time.time() - t0:.0f} s")

    full = fit["params"][FOCUS]
    for check, by in (("loco", "iso3"), ("loyo", "year")):
        est = leave_one_out(df, by)
        most = (est - full).abs().idxmax()
        for stat, val, lab in (("min", est.min(), est.idxmin()), ("max", est.max(), est.idxmax()),
                               ("most_influential", est[most], most), ("full_sample", full, "")):
            rows.append({"check": check, **meta, "term": FOCUS, "statistic": stat, "value": val, "label": str(lab)})

    wb = wild_cluster_bootstrap(d, "anom", fit["params"].index, FOCUS)
    for stat in ("p_wcr", "ci_lo", "ci_hi"):
        rows.append({"check": "wild_bootstrap", **meta, "term": FOCUS, "statistic": stat, "value": wb[stat],
                     "label": f"{N_BOOT} Rademacher draws by country, seed {SEED}"})

    pooled = coef[(coef["model"] == "bins_pooled_hill1") & (coef["term"] == "sev_plus_x_hill1")].iloc[0]
    for stat in ("beta", "se", "p"):
        rows.append({"check": "pooled_cutoff", "model": "bins_pooled_hill1", "term": "sev_plus_x_hill1",
                     "statistic": stat, "value": pooled[stat], "label": "sev_plus = spei12_w <= -1.5",
                     "n_obs": pooled["n_obs"], "n_countries": pooled["n_countries"]})
    out = pd.DataFrame(rows)[ROB_COLS]
    return out.astype({"n_obs": int, "n_countries": int})


# ---------------------------------------------------------------- counts

def severity_counts(df):
    """Country-years and distinct countries per bin (drought on the anom sample, heat on rows with heat),
    overall and by hill1 tercile of the countries (tercile cut points across all countries in df)."""
    per = df.groupby("iso3")["hill1"].first()
    terc = pd.qcut(per, 3, labels=["low", "mid", "high"]).astype(str)
    d = df.assign(hill1_tercile=df["iso3"].map(terc))
    rows = []
    for kind, bins in (("drought", DROUGHT_BINS + ["sev_plus"]), ("heat", HEAT_BINS)):
        for b in bins:
            sel = d[d[b] == 1]
            rows.append({"kind": kind, "bin": b, "hill1_tercile": "all",
                         "n_country_years": len(sel), "n_countries": sel["iso3"].nunique()})
            for t in ("low", "mid", "high"):
                s = sel[sel["hill1_tercile"] == t]
                rows.append({"kind": kind, "bin": b, "hill1_tercile": t,
                             "n_country_years": len(s), "n_countries": s["iso3"].nunique()})
    return pd.DataFrame(rows), terc


# ---------------------------------------------------------------- data

def load_frame():
    """panel.load_panel() plus index, loss10, heat, drought and heat bins, dry_sq."""
    df = panel.load_panel()
    nat = pd.read_parquet(paths.NATIONAL)[["iso3", "year", "index"]]
    heat = load("heat")[["iso3", "year", "heat_z_w"]].rename(columns={"heat_z_w": "heat"})
    assert not heat.duplicated(["iso3", "year"]).any()
    df = df.merge(nat, on=["iso3", "year"], how="left").merge(heat, on=["iso3", "year"], how="left")
    assert df["index"].notna().all()
    df["loss10"] = (df["index"] < LOSS_THR).astype(float)
    df = pd.concat([df, drought_bins(df["spei"]), heat_bins(df["heat"])], axis=1)
    df["dry_sq"] = df["dry"] ** 2
    return df


# ---------------------------------------------------------------- figure

def fig_severity(coef, breadth_coef, counts, bcounts, n_countries, pcts):
    """Left: implied anomaly by drought bin at hill1 p10 / p90 (bins_hill1). Right: breadth by bin (breadth_bins).
    bcounts: bin -> (country-years, countries) in the breadth sample."""
    pal = plt.get_cmap("tab10").colors
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE)
    c = coef[coef["model"] == "bins_hill1"].set_index("term")
    n = counts[(counts["kind"] == "drought") & (counts["hill1_tercile"] == "all")].set_index("bin")
    ticks = [f"{BIN_LABELS[b]}\n(n = {int(n.loc[b, 'n_country_years'])} c-y,\n{int(n.loc[b, 'n_countries'])} countries)"
             for b in DROUGHT_BINS]
    x = np.arange(len(DROUGHT_BINS))
    for dx, pct, col, q in ((-0.12, 10, pal[1], pcts[0]), (0.12, 90, pal[0], pcts[1])):
        r = c.loc[[f"{b}_at_p{pct}" for b in DROUGHT_BINS]]
        ax1.errorbar(x + dx, 100 * r["beta"], yerr=[100 * (r["beta"] - r["ci_lo"]), 100 * (r["ci_hi"] - r["beta"])],
                     fmt="o", color=col, markersize=5, capsize=3, linewidth=1,
                     label=f"hill1 at {pct}th percentile ({q:.1f} effective crops)")
    ax1.axhline(0, color=MUTED, linewidth=0.8)
    ax1.set_xticks(x, ticks, fontsize=6.5)
    ax1.grid(axis="x", visible=False)
    ax1.set_xlabel("Drought bin (SPEI-12, December, cropland-weighted)", fontsize=7)
    ax1.set_ylabel("National crop value anomaly vs. the country's other years\n(% of trend), 95% CI", fontsize=7)
    ax1.set_title("National value anomaly by drought severity\nbins x z(hill1), country and year FE, clustered SE",
                  fontsize=7.5, color=INK, loc="left")
    ax1.legend(frameon=False, fontsize=6.5, loc="lower left")

    b = breadth_coef[breadth_coef["model"] == "breadth_bins"].set_index("term").loc[DROUGHT_BINS]
    ax2.bar(x, 100 * b["beta"], width=0.55, color=pal[2])
    ax2.errorbar(x, 100 * b["beta"], yerr=[100 * (b["beta"] - b["ci_lo"]), 100 * (b["ci_hi"] - b["beta"])],
                 fmt="none", ecolor=INK, capsize=3, linewidth=0.8)
    ax2.axhline(0, color=MUTED, linewidth=0.8)
    ax2.set_xticks(x, [f"{BIN_LABELS[k]}\n(n = {bcounts[k][0]} c-y,\n{bcounts[k][1]} countries)" for k in DROUGHT_BINS],
                   fontsize=6.5)
    ax2.grid(axis="x", visible=False)
    ax2.set_xlabel("Drought bin (SPEI-12, December, cropland-weighted)", fontsize=7)
    ax2.set_ylabel("Change in value-weighted share of crops with yield\nbelow trend vs. other years "
                   "(percentage points), 95% CI", fontsize=7)
    ax2.set_title(f"Breadth (share of crop value with yield below trend)\n"
                  f"country and year FE, clustered SE; ≥ {MIN_CROPS} crops per country-year",
                  fontsize=7.5, color=INK, loc="left")
    fig.suptitle("Crop diversity and drought severity", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.text(0.01, 0.01, f"Source: FAOSTAT (QCL, QV), SPEIbase v2.10, MapSPAM 2020. 1993–2023, n = {n_countries} countries. "
             f"c-y = country-years.", fontsize=6, color=MUTED, ha="left", va="bottom")
    out = paths.fig_dir("results") / FIG_NAME
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out.relative_to(paths.ROOT)}")


# ---------------------------------------------------------------- main

def check_coef(coef):
    assert not coef.duplicated(["model", "term"]).any(), "duplicate model/term rows"
    assert np.isfinite(coef[["beta", "se"]]).all().all() and (coef["se"] > 0).all(), "non-finite beta or se"
    assert (coef["ci_lo"] <= coef["beta"]).all() and (coef["beta"] <= coef["ci_hi"]).all()
    assert coef["p"].between(0, 1).all() and coef["r2_within"].between(0, 1).all()
    assert (coef["n_countries"] >= panel.MIN_COUNTRIES).all() and (coef["n_countries"] <= N_COUNTRIES).all()


def main():
    assert not paths.USE_DUMMY, "severity has no dummy data: run without USE_DUMMY"
    df = load_frame()
    assert len(df) == df["iso3"].nunique() * df["year"].nunique(), "panel is not balanced"
    assert df["year"].between(1993, 2023).all()
    assert (df[DROUGHT_BINS].sum(axis=1) <= 1).all() and (df[HEAT_BINS].sum(axis=1).dropna() <= 1).all()
    assert (df["sev_plus"] == df["severe"] + df["extreme"]).all()
    assert int(df[DROUGHT_BINS].sum().sum()) == int((df["spei"] <= DROUGHT_THR).sum())

    coef, fits = run_models(df, panel_specs(df))
    check_coef(coef)
    ref = panel.fit_model(df, DROUGHT_BINS, ["hill1"], (), ())[0]
    assert np.allclose(ref["params"], fits["bins_hill1"][0]["params"]), "fit_outcome differs from panel.fit_model"
    counts, terc = severity_counts(df)
    assert (counts["n_country_years"] >= 0).all()
    tot = counts[counts["hill1_tercile"] == "all"].set_index("bin")["n_country_years"]
    split = counts[counts["hill1_tercile"] != "all"].groupby("bin")["n_country_years"].sum()
    assert (tot == split.reindex(tot.index)).all(), "tercile counts do not add up"

    ca = pd.read_parquet(paths.CROP_ANOMALIES)
    assert not ca.duplicated(["iso3", "year", "item"]).any()
    br = crop_breadth(ca)
    assert not br.duplicated(["iso3", "year"]).any() and (br["n_crops"] >= MIN_CROPS).all()
    assert br["breadth"].between(0, 1).all() and np.isfinite(br["comove"]).all()
    bdf = df.merge(br, on=["iso3", "year"], how="inner")
    bspecs = {f"{y}_{s}": (y, DROUGHT_BINS, inter, None)
              for y in ("breadth", "comove") for s, inter in (("bins", []), ("bins_hill1", ["hill1"]))}
    bcoef, _ = run_models(bdf, bspecs)
    check_coef(bcoef)

    rob = robustness(df, coef, fits)
    assert not rob.duplicated(["check", "model", "term", "statistic"]).any()
    assert rob.loc[rob["statistic"].isin(["p_perm", "p_perm_t", "p_wcr"]), "value"].between(0, 1).all()
    assert np.isfinite(rob["value"]).all()

    paths.STATS_DIR.mkdir(parents=True, exist_ok=True)
    for frame, name in ((coef, PANEL_FILE), (counts, COUNTS_FILE), (rob, ROB_FILE), (bcoef, BREADTH_FILE)):
        frame.to_csv(paths.STATS_DIR / name, index=False)
        print(f"wrote {(paths.STATS_DIR / name).relative_to(paths.ROOT)} ({len(frame)} rows)")

    d = fits["bins_hill1"][1]
    pcts = hill1_pcts(d)
    bcounts = {b: (int((bdf[b] == 1).sum()), bdf.loc[bdf[b] == 1, "iso3"].nunique()) for b in DROUGHT_BINS}
    fig_severity(coef, bcoef, counts, bcounts, df["iso3"].nunique(), pcts)

    pd.set_option("display.width", 220)
    print(f"\npanel: {len(df)} country-years, {df['iso3'].nunique()} countries, {df['year'].min()}-{df['year'].max()}; "
          f"hill1 p10 = {pcts[0]:.2f}, p90 = {pcts[1]:.2f}; hill1 tercile cut points = "
          f"{np.quantile(df.groupby('iso3')['hill1'].first(), [1 / 3, 2 / 3]).round(2).tolist()}, "
          f"countries per tercile = {terc.value_counts().reindex(['low', 'mid', 'high']).tolist()}")
    print(f"breadth/comove: {len(br)} country-years with >= {MIN_CROPS} crops in {br['iso3'].nunique()} countries; "
          f"median crops per country-year {br['n_crops'].median():.0f}")
    print("\ncounts per bin:")
    print(counts.pivot_table(index=["kind", "bin"], columns="hill1_tercile",
                             values=["n_country_years", "n_countries"], sort=False).to_string())
    print("\nseverity_panel (anom in fractions of trend; loss10 in probability; hill1 etc. as z-scores):")
    print(coef.round(4).to_string(index=False))
    print("\nseverity_robustness:")
    print(rob.assign(value=rob["value"].round(4)).to_string(index=False))
    print("\nseverity_breadth (breadth in share of crop value; comove in SD^2 units):")
    print(bcoef.round(4).to_string(index=False))
    print("\nraw means by drought bin (no fixed effects):")
    raw = bdf.assign(bin=np.select([bdf[b] == 1 for b in DROUGHT_BINS], DROUGHT_BINS, "other"))
    print(raw.groupby("bin")[["breadth", "comove", "anom"]].agg(["mean", "count"]).round(4).to_string())


if __name__ == "__main__":
    main()
