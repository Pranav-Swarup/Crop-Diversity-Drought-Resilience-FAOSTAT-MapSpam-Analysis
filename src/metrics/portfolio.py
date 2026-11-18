"""Crop-portfolio engine: expected value and risk of a country's crop mix when harvested area
is reallocated between the crops it already grows (mean-variance frontier within bounds).

Outputs
  outputs/portfolio_crops.parquet     one row per iso3 x crop (all crops of included countries)
  outputs/portfolio_frontier.parquet  one row per iso3 x point x portfolio crop (weights long)
  outputs/portfolio_cov.json          per country: the arrays evaluate() needs
  outputs/stats/portfolio_summary.csv one row per country
  figures/results/06_portfolio_frontier.png

Run: python -m src.metrics.portfolio   (USE_DUMMY=1 for dummy inputs; reads
     outputs/crop_anomalies.parquet for resp_beta, so run after src.metrics.run)

Model (per country; crops i = 1..n in the portfolio, years t in 1993-2023)
  r_it   = value_const_it / area_ha_it                     return per ha, I$/ha (constant 2014-16 prices)
  tau_it = LOWESS trend of r_i (frac 0.5) fitted on non-imputed years only
  a_it   = (r_it - tau_it) / tau_it                        relative anomaly; NaN in imputed years
  mu_i   = sum_{t in RECENT, non-imp} A_it tau_it / sum_{same t} A_it      expected I$/ha, RECENT = 2019-2023
  w0_i   = sum_{t in RECENT} A_it / sum_j sum_{t in RECENT} A_jt           current area share (all rows)
  Sigma  = covariance of a (shrunk, see shrink_cov)
  beta_i = resp_beta of crop i (outputs/crop_anomalies.parquet, joined on item)

  For area shares w (sum w = 1, lo <= w <= hi), with value shares s_i = w_i mu_i / (w . mu):
  exp_value_ha(w)   = w . mu                                              I$/ha
  risk_sd(w)        = sqrt(s' Sigma s) = sqrt((w*mu)' Sigma (w*mu)) / (w . mu)
                      = SD over years of the relative shortfall (V_t - E) / E of total value,
                        V_t = sum_i w_i mu_i (1 + a_it), E = w . mu
  drought_change(w) = SPEI_DROUGHT * sum_i s_i beta_i                     relative change of total value
                      in a year with SPEI-12 = -1.5 vs a year with SPEI-12 = 0;
                      NaN unless every crop with w_i > 0 has a beta
  drought_coverage  = sum of s_i over crops with a beta

Choices
  RECENT 2019-2023: last five years of the window; mean of the trend (not the observations) so one
    bad year does not set the level; area-weighted so that w0 . mu is the trend value per ha of the
    current mix. Needs >= MIN_RECENT_OBS non-imputed recent years.
  Crop inclusion: a row with value and area > 0 in all 31 years (complete), >= MIN_YEARS (20)
    non-imputed years and >= MIN_SHARE (1%) of the country's window value (the resp_beta rules,
    src.metrics.response), detrended SD >= MIN_SD (0.01, the flat-series CV of src.crops.quality).
  Imputed years (flags E, I) are dropped before detrending and from the covariance, so carried-forward
    values cannot lower a crop's variance. Covariances use the pairwise overlap of non-imputed years.
  Shrinkage (shrink_cov): correlations are shrunk towards the constant-correlation target (Ledoit &
    Wolf 2004), variances kept; analytic intensity
      lambda = sum_{i<j} Var^(r_ij) / sum_{i<j} (r_ij - rbar)^2,  clipped to [0, 1]
    (Schaefer & Strimmer 2005 estimator), computed pairwise on overlapping years; raised further if
    needed so that the shrunk correlation matrix has smallest eigenvalue >= PSD_EPS. A target of 0
    correlation was not used: crop anomalies are on average positively correlated (common weather),
    and shrinking towards 0 lowered risk_sd at current shares by a median 8% on the real data, i.e.
    it would credit diversification that the data do not show. Pairs with < MIN_OVERLAP common
    years get rbar.
  Bounds: lo_i = max(w0_i (1 - REL_BAND), w0_i - ABS_BAND), hi_i = min(w0_i (1 + REL_BAND), w0_i + ABS_BAND):
    no crop is more than halved or grown by half, and no crop gains or loses more than 10 percentage
    points of the portfolio area. No new crops. Area outside the portfolio crops is not modelled.
  Frontier: point 0 = minimum risk_sd within bounds (no return constraint); points 0..N_POINTS-1 minimise
    risk_sd subject to w . mu >= E_k, E_k evenly spaced from E(point 0) to the maximum feasible w . mu.
    Solved with SLSQP from several feasible starts; a backward pass makes risk non-decreasing in E_k.
    "min_risk_at_current" = minimum risk_sd subject to w . mu >= w0 . mu.

Checks (asserted)
  identity: at w0, evaluate() with the sample covariance of the years where every portfolio crop is
    non-imputed equals the SD of the directly built series (V_t - E) / E on those years (1e-9).
  national figure: exp_value_ha(w0) vs the directly computed value per ha of the portfolio crops over
    RECENT, sum_{i,t} V_it / sum_{i,t} A_it (all rows, no model): |exp_gap| <= EXP_GAP_TOL (0.15) per country and
    median |exp_gap| <= EXP_GAP_MEDIAN_TOL (0.02).
    The gap is not 0 by construction: it is minus the area-weighted mean relative residual of the
    portfolio crops from their trends in RECENT, plus the effect of imputed recent years (left out of mu).
    risk_sd(w0) vs the SD of the relative anomalies of the yearly series sum_i V_it / sum_i A_it is
    reported (risk_gap), not asserted: it differs through shrinkage, the masking of imputed years and
    the changing historical crop mix (the portfolio holds shares at w0).
  Coverage: value_coverage = RECENT value of portfolio crops / RECENT value of all crops.

portfolio_crops.parquet (key iso3, item_code)
  iso3, item_code, item, in_portfolio (bool), exclude_reason ("" if in), value_share (window value
  share, all crops), area_share_all (RECENT area share of all crops), area_share (w0; NaN if out),
  share_lo, share_hi (bounds; NaN if out), value_ha (mu, I$/ha), sd_anom (SD of a_i), n_years (non-imputed
  years used), resp_beta (NaN if none), has_beta (bool)
portfolio_frontier.parquet (key iso3, point, item_code)
  iso3, point (0..N_POINTS-1 frontier, N_POINTS = current, N_POINTS + 1 = min_risk_at_current),
  kind ("frontier" | "current" | "min_risk_at_current"), target_value_ha (E_k; NaN for current),
  exp_value_ha, risk_sd, drought_change, drought_coverage (per point, repeated on each crop row),
  item_code, item, weight (area share of that crop at that point)
portfolio_cov.json
  {"meta": {...constants and units...},
   "countries": {iso3: {"items": [...], "item_codes": [...], "w0": [...], "lo": [...], "hi": [...],
                        "mu": [...], "cov": [[...]], "beta": [... null if none], "shrinkage": lambda, "mean_corr": rbar,
                        "frontier": [{"exp_value_ha", "risk_sd"}...]}}}
  Arrays share the order of "items"; evaluate(w, mu, cov, beta) on them reproduces the parquet values.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..common import paths
from ..common.contract import YEAR_MAX, YEAR_MIN, load
from . import response
from .detrend import FRAC, detrend
from .diversity import hill1

YEARS = np.arange(YEAR_MIN, YEAR_MAX + 1)
RECENT = (2019, 2023)
MIN_RECENT_OBS = 3
MIN_YEARS = response.MIN_YEARS
MIN_SHARE = response.MIN_SHARE
MIN_SD = 0.01
MIN_OVERLAP = 10
MIN_CROPS = 2
REL_BAND = 0.5
ABS_BAND = 0.10
SPEI_DROUGHT = -1.5
N_POINTS = 20
PSD_EPS = 1e-6
EXP_GAP_TOL = 0.15
EXP_GAP_MEDIAN_TOL = 0.02
IDENTITY_TOL = 1e-9
FEAS_TOL = 1e-8
FIG_COUNTRIES = ("IND", "COL", "FIN")
FIG_MIN_CROPS = 3
PORTFOLIO_COV = paths.OUT / "portfolio_cov.json"
SUMMARY = paths.STATS_DIR / "portfolio_summary.csv"


# ---------------------------------------------------------------- portfolio maths (pure numpy)

def evaluate(weights, mu, cov, beta, spei=SPEI_DROUGHT):
    """Expected value per ha, risk and drought change of area shares `weights` (see module docstring).

    weights, mu, beta: length-n arrays (beta may hold NaN); cov: n x n covariance of relative anomalies.
    Returns dict(exp_value_ha, risk_sd, drought_change, drought_coverage).
    """
    w, mu, beta = (np.asarray(x, dtype=float) for x in (weights, mu, beta))
    cov = np.asarray(cov, dtype=float)
    e = float(w @ mu)
    s = w * mu / e
    risk = float(np.sqrt(max(s @ cov @ s, 0.0)))
    known = np.isfinite(beta)
    coverage = float(s[known].sum())
    complete = bool(np.all(known | (w <= 0)))
    drought = float(spei * (s[known] @ beta[known])) if complete else np.nan
    return {"exp_value_ha": e, "risk_sd": risk, "drought_change": drought, "drought_coverage": coverage}


def shrink_cov(anom, min_overlap=MIN_OVERLAP, eps=PSD_EPS):
    """Shrunk covariance of anomalies `anom` (years x crops, NaN = not observed).

    Correlations r_ij on the pairwise overlap of observed years. Target T: 1 on the diagonal and
    rbar = max(0, mean of the r_ij) off it (constant correlation, Ledoit & Wolf 2004). Shrunk
    R* = (1 - lambda) R + lambda T with
        lambda = sum_{i<j} Var^(r_ij) / sum_{i<j} (r_ij - rbar)^2,  clipped to [0, 1],
        Var^(r_ij) = k/(k-1)^3 sum_t (x_ti x_tj - mean_t(x_ti x_tj))^2,  x standardised on the k overlap years
    (Schaefer & Strimmer 2005 estimator; the covariance of r_ij with rbar is ignored). lambda is then
    raised (bisection) until min eigenvalue of R* >= eps, if needed. Variances come from each crop's own
    observed years (ddof=1) and are not shrunk. Pairs with < min_overlap common years get rbar.
    Returns dict(cov, lam, lam_analytic, rbar, n_pairs_target).
    """
    a = np.asarray(anom, dtype=float)
    n = a.shape[1]
    sd = np.array([np.nanstd(a[:, i], ddof=1) for i in range(n)])
    r, var_r, have = np.eye(n), np.zeros((n, n)), np.zeros((n, n), bool)
    for i in range(n):
        for j in range(i + 1, n):
            ok = np.isfinite(a[:, i]) & np.isfinite(a[:, j])
            k = int(ok.sum())
            if k < min_overlap:
                continue
            xi, xj = a[ok, i], a[ok, j]
            wk = (xi - xi.mean()) / xi.std(ddof=1) * (xj - xj.mean()) / xj.std(ddof=1)
            r[i, j] = r[j, i] = wk.sum() / (k - 1)
            var_r[i, j] = k / (k - 1) ** 3 * ((wk - wk.mean()) ** 2).sum()
            have[i, j] = True
    n_target = n * (n - 1) // 2 - int(have.sum())
    rbar = max(0.0, float(r[have].mean())) if have.any() else 0.0
    for i, j in zip(*np.triu_indices(n, 1)):
        if not have[i, j]:
            r[i, j] = r[j, i] = rbar
    t = np.full((n, n), rbar)
    np.fill_diagonal(t, 1.0)
    den = ((r[have] - rbar) ** 2).sum()
    lam_a = float(np.clip(var_r[have].sum() / den, 0, 1)) if den > 0 else 1.0

    def min_eig(lam):
        return float(np.linalg.eigvalsh((1 - lam) * r + lam * t).min())

    lam = lam_a
    if n > 1 and min_eig(lam) < eps and min_eig(1.0) >= eps:
        lo, hi = lam, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if min_eig(mid) < eps else (lo, mid)
        lam = hi
    rs = (1 - lam) * r + lam * t
    return {"cov": rs * np.outer(sd, sd), "lam": float(lam), "lam_analytic": lam_a, "rbar": rbar,
            "n_pairs_target": n_target}


def share_bounds(w0, rel=REL_BAND, absolute=ABS_BAND):
    """Per-crop (lo, hi) area-share bounds around current shares w0."""
    w0 = np.asarray(w0, dtype=float)
    lo = np.maximum(w0 * (1 - rel), w0 - absolute)
    hi = np.minimum(w0 * (1 + rel), w0 + absolute)
    return np.maximum(lo, 0.0), hi


def max_value_weights(mu, lo, hi):
    """Shares maximising w . mu within bounds and sum 1 (greedy: start at lo, fill highest mu first)."""
    w = np.asarray(lo, dtype=float).copy()
    left = 1.0 - w.sum()
    for i in np.argsort(-np.asarray(mu)):
        add = min(hi[i] - w[i], left)
        w[i] += add
        left -= add
    return w


def _feasible(w, mu, lo, hi, target):
    ok = np.all(w >= lo - FEAS_TOL) and np.all(w <= hi + FEAS_TOL) and abs(w.sum() - 1) < FEAS_TOL
    return bool(ok and (target is None or w @ mu >= target * (1 - FEAS_TOL)))


def _cv2(w, c, mu):
    return (w @ c @ w) / (w @ mu) ** 2


def min_risk(mu, cov, lo, hi, target=None, starts=()):
    """Shares minimising risk_sd s.t. sum w = 1, lo <= w <= hi and (if target) w . mu >= target.

    Minimises risk_sd^2 = w'Cw / (w . mu)^2, C = diag(mu) cov diag(mu), with SLSQP from each start;
    returns the best feasible result (starts included as candidates).
    """
    mu = np.asarray(mu, dtype=float)
    scale = mu.mean()
    m = mu / scale                                   # risk_sd is invariant to rescaling mu
    c = np.outer(m, m) * np.asarray(cov, dtype=float)
    t = None if target is None else target / scale

    def fun(w):
        v, e = w @ c @ w, w @ m
        return v / e ** 2, 2 * (c @ w) / e ** 2 - 2 * v * m / e ** 3

    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones_like(w)}]
    if t is not None:
        cons.append({"type": "ineq", "fun": lambda w: w @ m - t, "jac": lambda w: m})
    best, best_f = None, np.inf
    for w_start in starts:
        w_start = np.asarray(w_start, dtype=float)
        cands = [w_start]
        res = minimize(fun, w_start, jac=True, method="SLSQP", bounds=list(zip(lo, hi)), constraints=cons,
                       options={"ftol": 1e-15, "maxiter": 1000})
        cands.append(np.clip(res.x, lo, hi))
        for w in cands:
            if _feasible(w, m, lo, hi, t) and _cv2(w, c, m) < best_f:
                best, best_f = w, _cv2(w, c, m)
    assert best is not None, "min_risk: no feasible start"
    return best


def _blend_start(w_a, w_b, mu, target):
    """Convex combination of w_a and w_b (w_b . mu >= target) with value >= target."""
    ea, eb = w_a @ mu, w_b @ mu
    if ea >= target:
        return w_a
    theta = min(1.0, (target - ea) / (eb - ea)) if eb > ea else 1.0
    return (1 - theta) * w_a + theta * w_b


def frontier(mu, cov, w0, lo, hi, n_points=N_POINTS):
    """Frontier weights and targets. Returns dict(weights [n_points x n], targets, w_opt) where
    w_opt = minimum risk_sd at w . mu >= w0 . mu. Risk is non-decreasing along the points."""
    mu, w0 = np.asarray(mu, dtype=float), np.asarray(w0, dtype=float)
    w_max = max_value_weights(mu, lo, hi)
    room = (hi - lo).sum()
    centre = lo + (hi - lo) * (1 - lo.sum()) / room if room > 0 else w0   # same position in every band
    w_gmv = min_risk(mu, cov, lo, hi, None, starts=(w0, centre, w_max))
    targets = np.linspace(w_gmv @ mu, w_max @ mu, n_points)
    ws, prev = [], w_gmv
    for k, tk in enumerate(targets):
        if k == 0:
            ws.append(w_gmv)
            continue
        starts = [_blend_start(prev, w_max, mu, tk), _blend_start(w0, w_max, mu, tk),
                  _blend_start(w_gmv, w_max, mu, tk), w_max]
        prev = min_risk(mu, cov, lo, hi, tk, starts=starts)
        ws.append(prev)
    ws = np.array(ws)
    c = np.outer(mu, mu) * cov
    for k in range(n_points - 2, -1, -1):           # w_{k+1} is feasible for target k
        if _cv2(ws[k + 1], c, mu) < _cv2(ws[k], c, mu):
            ws[k] = ws[k + 1]
    e0 = w0 @ mu
    cands = [w0, min_risk(mu, cov, lo, hi, e0, starts=[w0] + [w for w in ws if w @ mu >= e0])]
    cands += [w for w in ws if w @ mu >= e0]
    w_opt = min(cands, key=lambda w: _cv2(w, c, mu))
    return {"weights": ws, "targets": targets, "w_opt": w_opt}


def identity_gap(w0, mu, anom, beta):
    """|evaluate() risk - SD of directly built (V_t - E)/E| on years where every crop is observed,
    using the unshrunk sample covariance of those years. NaN if fewer than 3 such years."""
    a = np.asarray(anom, dtype=float)
    bal = a[np.isfinite(a).all(axis=1)]
    if len(bal) < 3:
        return np.nan
    e = w0 @ mu
    v = (w0 * mu * (1 + bal)).sum(axis=1)
    direct = np.std((v - e) / e, ddof=1)
    return float(abs(evaluate(w0, mu, np.cov(bal, rowvar=False, ddof=1), beta)["risk_sd"] - direct))


# ---------------------------------------------------------------- data

def crop_table(g, betas):
    """Per-crop inputs for one country. g: that country's crops.parquet rows with a value.
    betas: Series item -> resp_beta. Returns (table, anomaly DataFrame years x item_code of portfolio crops)."""
    total = g["value_const"].sum()
    rec = g[g["year"].between(*RECENT)]
    rec_area = rec.groupby("item_code")["area_ha"].sum()
    rows, anoms = [], {}
    for code, c in g.groupby("item_code"):
        c = c.sort_values("year")
        item = c["item"].iloc[0]
        ok = ~c["is_imputed"] & (c["area_ha"] > 0) & (c["value_const"] > 0)
        ni = c[ok]
        d = detrend((ni["value_const"] / ni["area_ha"]).to_numpy(), ni["year"].to_numpy(), FRAC)
        d.index = ni["year"].to_numpy()
        anom = d["anom"].reindex(YEARS)
        sd = float(anom.std(ddof=1)) if anom.notna().sum() > 1 else np.nan
        recent = d.loc[(d.index >= RECENT[0]) & (d.index <= RECENT[1]) & d["trend"].notna()]
        rec_a = ni.set_index("year")["area_ha"].reindex(recent.index)
        mu = float((rec_a * recent["trend"]).sum() / rec_a.sum()) if len(recent) else np.nan
        share = c["value_const"].sum() / total
        complete = c.loc[c["area_ha"] > 0, "year"].nunique() == len(YEARS)
        reason = ("incomplete series" if not complete else
                  "value share < 1%" if share < MIN_SHARE else
                  f"< {MIN_YEARS} non-imputed years" if anom.notna().sum() < MIN_YEARS else
                  f"< {MIN_RECENT_OBS} non-imputed recent years" if len(recent) < MIN_RECENT_OBS else
                  "flat series" if not sd >= MIN_SD else "")
        beta = betas.get(item, np.nan)
        rows.append({"item_code": int(code), "item": item, "in_portfolio": reason == "", "exclude_reason": reason,
                     "value_share": float(share), "rec_area": float(rec_area.get(code, 0.0)),
                     "value_ha": mu, "sd_anom": sd, "n_years": int(anom.notna().sum()),
                     "resp_beta": float(beta), "has_beta": bool(np.isfinite(beta))})
        if reason == "":
            anoms[int(code)] = anom.to_numpy()
    t = pd.DataFrame(rows)
    t["area_share_all"] = t["rec_area"] / t["rec_area"].sum()
    port = t["in_portfolio"]
    t["area_share"] = np.where(port, t["rec_area"] / t.loc[port, "rec_area"].sum(), np.nan)
    return t, pd.DataFrame(anoms, index=YEARS)


def national_direct(g, codes):
    """Directly computed value per ha of crops `codes` (all rows). Returns (sum V / sum A over RECENT,
    SD of the LOWESS relative anomalies of the yearly series sum_i V_it / sum_i A_it)."""
    p = g[g["item_code"].isin(codes)].groupby("year")[["value_const", "area_ha"]].sum().reindex(YEARS)
    d = detrend((p["value_const"] / p["area_ha"]).to_numpy(), YEARS, FRAC)
    rec = p.loc[RECENT[0]:RECENT[1]]
    return float(rec["value_const"].sum() / rec["area_ha"].sum()), float(d["anom"].std(ddof=1))


def country_portfolio(iso3, g, betas):
    """Everything for one country. Returns dict or None (fewer than MIN_CROPS portfolio crops)."""
    t, anom = crop_table(g, betas)
    t.insert(0, "iso3", iso3)
    port = t[t["in_portfolio"]].reset_index(drop=True)
    if len(port) < MIN_CROPS:
        return {"crops": t, "port": None}
    a = anom[port["item_code"]].to_numpy()
    w0, mu, beta = port["area_share"].to_numpy(), port["value_ha"].to_numpy(), port["resp_beta"].to_numpy()
    sh = shrink_cov(a)
    cov = sh["cov"]
    lo, hi = share_bounds(w0)
    fr = frontier(mu, cov, w0, lo, hi)
    gap = identity_gap(w0, mu, a, beta)
    exp_direct, risk_direct = national_direct(g, port["item_code"])
    t["share_lo"] = t["item_code"].map(dict(zip(port["item_code"], lo)))
    t["share_hi"] = t["item_code"].map(dict(zip(port["item_code"], hi)))
    return {"crops": t, "port": port, "w0": w0, "mu": mu, "beta": beta, "cov": cov, "lo": lo, "hi": hi,
            "shrink": sh, "fr": fr, "identity_gap": gap, "exp_direct": exp_direct, "risk_direct": risk_direct}


def frontier_rows(iso3, r):
    """Long frontier table for one country."""
    port, fr = r["port"], r["fr"]
    pts = [(k, "frontier", tk, w) for k, (tk, w) in enumerate(zip(fr["targets"], fr["weights"]))]
    pts += [(N_POINTS, "current", np.nan, r["w0"]),
            (N_POINTS + 1, "min_risk_at_current", float(r["w0"] @ r["mu"]), fr["w_opt"])]
    out = []
    for k, kind, tk, w in pts:
        ev = evaluate(w, r["mu"], r["cov"], r["beta"])
        out.append(pd.DataFrame({"iso3": iso3, "point": k, "kind": kind, "target_value_ha": tk, **ev,
                                 "item_code": port["item_code"].to_numpy(), "item": port["item"].to_numpy(),
                                 "weight": w}))
    return pd.concat(out, ignore_index=True)


def summary_row(iso3, r, total_rec_value):
    """One portfolio_summary.csv row."""
    port, w0, w1 = r["port"], r["w0"], r["fr"]["w_opt"]
    cur = evaluate(w0, r["mu"], r["cov"], r["beta"])
    opt = evaluate(w1, r["mu"], r["cov"], r["beta"])
    d = pd.Series(w1 - w0, index=port["item"].to_numpy())
    gain, lose = d[d > 1e-9].nlargest(3), d[d < -1e-9].nsmallest(3)
    row = {"iso3": iso3, "n_crops": len(port),
           "value_coverage": r["rec_value"] / total_rec_value,
           "area_coverage": float(r["crops"].loc[r["crops"]["in_portfolio"], "area_share_all"].sum()),
           "exp_value_ha_current": cur["exp_value_ha"], "exp_value_ha_direct": r["exp_direct"],
           "exp_gap": cur["exp_value_ha"] / r["exp_direct"] - 1,
           "risk_current": cur["risk_sd"], "risk_direct": r["risk_direct"],
           "risk_gap": cur["risk_sd"] / r["risk_direct"] - 1,
           "exp_value_ha_opt": opt["exp_value_ha"], "risk_opt": opt["risk_sd"],
           "risk_reduction_pct": 100 * (1 - opt["risk_sd"] / cur["risk_sd"]),
           "drought_change_current": cur["drought_change"], "drought_change_opt": opt["drought_change"],
           "drought_coverage": cur["drought_coverage"],
           "hill1_current": hill1(w0), "hill1_opt": hill1(w1),
           "shrinkage": r["shrink"]["lam"], "mean_corr": r["shrink"]["rbar"], "n_pairs_target": r["shrink"]["n_pairs_target"],
           "identity_gap": r["identity_gap"]}
    for kind, s in (("gain", gain), ("lose", lose)):
        for i in range(3):
            row[f"{kind}_{i + 1}"] = s.index[i] if i < len(s) else ""
            row[f"{kind}_{i + 1}_pp"] = 100 * float(s.iloc[i]) if i < len(s) else np.nan
    return row


def _num(x):
    return [None if not np.isfinite(v) else float(v) for v in np.asarray(x, dtype=float).ravel()]


def cov_json(results):
    """portfolio_cov.json content."""
    meta = {"recent_years": list(RECENT), "rel_band": REL_BAND, "abs_band": ABS_BAND,
            "spei_drought": SPEI_DROUGHT, "n_points": N_POINTS,
            "units": {"mu": "I$ per ha (constant 2014-2016)", "cov": "relative anomalies (fraction)^2",
                      "w0": "area share of portfolio crops", "beta": "relative yield anomaly per unit SPEI-12"},
            "formulas": {"exp_value_ha": "sum(w*mu)", "s": "w*mu/sum(w*mu)", "risk_sd": "sqrt(s' cov s)",
                         "drought_change": "spei_drought*sum(s*beta); null unless every crop with w>0 has beta"}}
    out = {}
    for iso3, r in results.items():
        f = r["fr"]["weights"]
        out[iso3] = {"items": r["port"]["item"].tolist(), "item_codes": [int(c) for c in r["port"]["item_code"]],
                     "w0": _num(r["w0"]), "lo": _num(r["lo"]), "hi": _num(r["hi"]), "mu": _num(r["mu"]),
                     "cov": [_num(row) for row in r["cov"]], "beta": _num(r["beta"]),
                     "shrinkage": r["shrink"]["lam"], "mean_corr": r["shrink"]["rbar"],
                     "frontier": [{k: evaluate(w, r["mu"], r["cov"], r["beta"])[k] for k in ("exp_value_ha", "risk_sd")}
                                  for w in f]}
    return {"meta": meta, "countries": out}


# ---------------------------------------------------------------- figure

def figure_countries(summary):
    """FIG_COUNTRIES with >= FIG_MIN_CROPS portfolio crops; filled up to 3 with the countries
    having the most portfolio crops (ties alphabetical)."""
    ok = summary[summary["n_crops"] >= FIG_MIN_CROPS]
    pick = [c for c in FIG_COUNTRIES if c in set(ok["iso3"])]
    rest = ok[~ok["iso3"].isin(pick)].sort_values(["n_crops", "iso3"], ascending=[False, True])["iso3"]
    return pick + rest.tolist()[: 3 - len(pick)]


def fig_frontier(frontier_df, summary):
    from . import figures as F                     # rcParams and save()
    isos = figure_countries(summary)
    pts = frontier_df.drop_duplicates(["iso3", "point"])
    colors = plt.get_cmap("tab10").colors
    fig, axes = plt.subplots(1, len(isos), figsize=F.FIGSIZE, squeeze=False)
    for ax, iso3 in zip(axes[0], isos):
        p = pts[pts["iso3"] == iso3]
        f = p[p["kind"] == "frontier"].sort_values("point")
        cur = p[p["kind"] == "current"].iloc[0]
        opt = p[p["kind"] == "min_risk_at_current"].iloc[0]
        n = int(summary.set_index("iso3").loc[iso3, "n_crops"])
        ax.plot(100 * f["risk_sd"], f["exp_value_ha"], color=colors[0], lw=1.5, marker=".", ms=4,
                label="Frontier within share bounds")
        ax.scatter(100 * cur["risk_sd"], cur["exp_value_ha"], color=colors[1], s=36, zorder=3, label="Current shares")
        ax.scatter(100 * opt["risk_sd"], opt["exp_value_ha"], color=colors[2], marker="D", s=30, zorder=3,
                   label="Min risk at ≥ current value")
        ax.set_title(f"{iso3} ({n} crops)", fontsize=8.5, color=F.INK, loc="left")
        ax.set_xlabel("Risk: SD of relative value shortfall (%)")
        ax.set_ylabel("Expected value (2014–16 I$ per ha)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, 0.04))
    fig.suptitle(f"Expected value vs risk when area is reallocated between existing crops "
                 f"(each share within {1 - REL_BAND:g}–{1 + REL_BAND:g}× and ±{100 * ABS_BAND:g} pp of current)",
                 fontsize=9, color=F.INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    F.save(fig, "06_portfolio_frontier.png", len(isos))
    return isos


# ---------------------------------------------------------------- main

@threadpool_limits.wrap(limits=1, user_api="blas")
def main():
    crops, quality = load("crops"), load("quality")
    included = quality.loc[quality["included"], "iso3"].tolist()
    anoms = pd.read_parquet(paths.CROP_ANOMALIES)
    betas = anoms.drop_duplicates(["iso3", "item"]).set_index(["iso3", "item"])["resp_beta"]
    crops = crops[crops["iso3"].isin(included) & crops["value_const"].notna()]
    assert crops.groupby(["iso3", "item"])["item_code"].nunique().max() == 1, "item names not unique per country"

    crop_rows, frontier_parts, summary, results = [], [], [], {}
    for iso3, g in crops.groupby("iso3", sort=True):
        b = betas.loc[iso3] if iso3 in betas.index.get_level_values(0) else pd.Series(dtype=float)
        r = country_portfolio(iso3, g, b)
        crop_rows.append(r["crops"])
        if r["port"] is None:
            continue
        rec = g[g["year"].between(*RECENT)]
        r["rec_value"] = rec.loc[rec["item_code"].isin(r["port"]["item_code"]), "value_const"].sum()
        results[iso3] = r
        frontier_parts.append(frontier_rows(iso3, r))
        summary.append(summary_row(iso3, r, rec["value_const"].sum()))

    cols = ["iso3", "item_code", "item", "in_portfolio", "exclude_reason", "value_share", "area_share_all",
            "area_share", "share_lo", "share_hi", "value_ha", "sd_anom", "n_years", "resp_beta", "has_beta"]
    pc = pd.concat(crop_rows, ignore_index=True)
    for c in ("share_lo", "share_hi"):
        pc[c] = pc[c] if c in pc else np.nan
    pc = pc[cols].astype({"iso3": str, "item_code": int, "item": str, "in_portfolio": bool,
                          "exclude_reason": str, "n_years": int, "has_beta": bool})
    fr = pd.concat(frontier_parts, ignore_index=True)
    fr = fr.astype({"iso3": str, "point": int, "kind": str, "item_code": int, "item": str})
    sm = pd.DataFrame(summary)

    # Assertions.
    assert not pc.duplicated(["iso3", "item_code"]).any()
    assert not fr.duplicated(["iso3", "point", "item_code"]).any()
    assert np.allclose(pc[pc["in_portfolio"]].groupby("iso3")["area_share"].sum().loc[sm["iso3"]], 1, atol=1e-9)
    assert np.allclose(fr.groupby(["iso3", "point"])["weight"].sum(), 1, atol=FEAS_TOL)
    bnd = fr.merge(pc[["iso3", "item_code", "share_lo", "share_hi"]], on=["iso3", "item_code"])
    assert (bnd["weight"] >= bnd["share_lo"] - FEAS_TOL).all() and (bnd["weight"] <= bnd["share_hi"] + FEAS_TOL).all()
    assert (fr["risk_sd"] >= 0).all() and (fr["exp_value_ha"] > 0).all()
    assert (sm["risk_opt"] <= sm["risk_current"] + 1e-10).all()
    assert (sm["exp_value_ha_opt"] >= sm["exp_value_ha_current"] * (1 - FEAS_TOL)).all()
    fpts = fr[fr["kind"] == "frontier"].drop_duplicates(["iso3", "point"]).sort_values(["iso3", "point"])
    assert (fpts.groupby("iso3")["risk_sd"].diff().dropna() >= -1e-10).all(), "frontier risk decreases"
    ig = sm["identity_gap"].dropna()
    assert (ig < IDENTITY_TOL).all(), f"identity violated: max {ig.max()}"
    assert (sm["exp_gap"].abs() <= EXP_GAP_TOL).all(), \
        f"exp_value_ha off the direct figure: {sm.loc[sm['exp_gap'].abs() > EXP_GAP_TOL, ['iso3', 'exp_gap']]}"
    assert sm["exp_gap"].abs().median() <= EXP_GAP_MEDIAN_TOL
    assert (sm["value_coverage"] <= 1 + 1e-9).all() and (sm["hill1_opt"] >= 1).all()

    paths.OUT.mkdir(parents=True, exist_ok=True)
    paths.STATS_DIR.mkdir(parents=True, exist_ok=True)
    pc.to_parquet(paths.PORTFOLIO_CROPS, index=False)
    fr.to_parquet(paths.PORTFOLIO_FRONTIER, index=False)
    sm.to_csv(SUMMARY, index=False)
    with open(PORTFOLIO_COV, "w") as f:
        json.dump(cov_json(results), f, allow_nan=False)
    for p in (paths.PORTFOLIO_CROPS, paths.PORTFOLIO_FRONTIER, PORTFOLIO_COV, SUMMARY):
        print(f"wrote {p.relative_to(paths.ROOT)}")
    isos = fig_frontier(fr, sm)

    print(f"\ncountries: {len(sm)} of {len(included)} included (fewer than {MIN_CROPS} portfolio crops: "
          f"{len(included) - len(sm)})")
    print(f"portfolio crops: {int(pc['in_portfolio'].sum())} of {len(pc)}; median per country {sm['n_crops'].median():.0f}")
    print("excluded crops by reason: " + ", ".join(f"{k}={v}" for k, v in pc.loc[~pc['in_portfolio'], 'exclude_reason'].value_counts().items()))
    print(f"median value coverage {sm['value_coverage'].median():.3f} (p10 {sm['value_coverage'].quantile(0.1):.3f}); "
          f"median area coverage {sm['area_coverage'].median():.3f}")
    print(f"median |exp_gap| {sm['exp_gap'].abs().median():.4f} (max {sm['exp_gap'].abs().max():.4f}); "
          f"|exp_gap| > 0.05: {sm.loc[sm['exp_gap'].abs() > 0.05, 'iso3'].tolist()}; "
          f"median risk_gap {sm['risk_gap'].median():.3f}; identity checked in {len(ig)} countries, max gap {ig.max():.1e}")
    print(f"median shrinkage {sm['shrinkage'].median():.3f}, median mean correlation {sm['mean_corr'].median():.3f}; pairs at target {int(sm['n_pairs_target'].sum())}")
    print(f"median risk current {sm['risk_current'].median():.4f}, optimised {sm['risk_opt'].median():.4f}; "
          f"median % risk reduction {sm['risk_reduction_pct'].median():.1f} (p10 {sm['risk_reduction_pct'].quantile(0.1):.1f}, "
          f"p90 {sm['risk_reduction_pct'].quantile(0.9):.1f})")
    dc = sm["drought_change_current"]
    print(f"drought_change available for {int(dc.notna().sum())} countries; median current {dc.median():.4f}, "
          f"optimised {sm['drought_change_opt'].median():.4f}")
    print(f"median hill1 current {sm['hill1_current'].median():.2f}, optimised {sm['hill1_opt'].median():.2f}")
    print(f"figure countries: {isos}")


if __name__ == "__main__":
    main()
