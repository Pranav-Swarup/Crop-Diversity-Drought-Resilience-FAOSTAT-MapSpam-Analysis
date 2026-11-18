"""District metrics (four faces, response diversity, portfolio decomposition, time-varying diversity) for India.

Pure functions on a crop table with the columns of data/clean/india/crops.parquet
(dist_code, year, crop, area_ha, prod_t, yield_t_ha, value_const) plus a district climate table. Every rule comes from
src.metrics (detrend, faces, diversity, response, decompose, events): nothing is re-implemented here.

Value index = sum of value_const over the crops that have a price and a complete series in that district ("balanced" crops).
A crop reported only in some years would make the sum jump when it enters or leaves; such crops are left out of the value
index (and reported as `value_coverage`), never filled. Diversity (hill1, hill1_tv) uses crops with reported area throughout the corresponding full or lagged window.
"""
import numpy as np
import pandas as pd

from ..metrics.decompose import decompose
from ..metrics.detrend import detrend, lowess_trend
from ..metrics.diversity import hill1
from ..metrics.events import find_events
from ..metrics.faces import n_usable_events, recovery, resistance, stability, vulnerability
from ..metrics.response import MIN_SHARE, resp_beta, resp_div, resp_divergence

MIN_COVERAGE = 0.7       # balanced crops must hold this share of the district's priced value, else the district is dropped
MIN_CROPS_VALUE = 3      # at least this many balanced crops


def area_matrix(g, years):
    return g.pivot_table(index="crop", columns="year", values="area_ha", aggfunc="sum").reindex(columns=years)


def hill1_tv(A, years, window):
    """Hill-1 of the area pooled over the `window` years before t (t-window..t-1); NaN if the window leaves the data."""
    out = {}
    for t in years:
        cols = [y for y in range(t - window, t)]
        out[t] = hill1(A[cols].sum(axis=1, min_count=window).dropna().to_numpy()) if cols[0] >= years[0] else np.nan
    return pd.Series(out)


def district_value(g, years):
    """(balanced value matrix crops x years, coverage). NaN rows dropped: complete series only."""
    v = g[g.value_const.notna()].pivot_table(index="crop", columns="year", values="value_const", aggfunc="sum").reindex(columns=years)
    bal = v.dropna(how="any")
    cov = bal.mean(axis=1).sum() / v.fillna(0).mean(axis=1).sum() if len(v) else np.nan
    return bal, cov


def one_district(g, clim, years, events):
    """Returns (metrics dict or None, panel DataFrame or None)."""
    A = area_matrix(g, years)
    bal, cov = district_value(g, years)
    if len(bal) < MIN_CROPS_VALUE or not np.isfinite(cov) or cov < MIN_COVERAGE:
        return None, None
    yrs = np.asarray(years)
    total = bal.sum(axis=0).to_numpy()
    d = detrend(total, x=yrs)
    if not np.isfinite(d["index"]).all():
        return None, None
    index = pd.Series(d["index"].to_numpy(), index=years)
    ev = events.year.tolist()
    spei = clim.set_index("year").spei12_w.reindex(years)
    drought_years = spei.index[spei <= -1.0]

    # per-crop yield anomalies and response slopes (balanced crops with >= MIN_SHARE of value)
    share = bal.mean(axis=1) / bal.mean(axis=1).sum()
    use = share[share >= MIN_SHARE].index
    yld = g.pivot_table(index="crop", columns="year", values="yield_t_ha", aggfunc="mean").reindex(index=use, columns=years)
    betas, anoms = {}, {}
    for crop in use:
        a = detrend(yld.loc[crop].to_numpy(), x=yrs)["anom"].to_numpy()
        anoms[crop] = a
        betas[crop] = resp_beta(a, spei.to_numpy())
    b = np.array([betas[c] for c in use])
    w = share[use].to_numpy()
    # residual = value - LOWESS trend, without the positivity filter of detrend() (only ratios need it)
    resid = np.column_stack([bal.loc[c].to_numpy() - lowess_trend(bal.loc[c].to_numpy(), x=yrs) for c in bal.index])
    dec = decompose(bal.T.to_numpy(), resid)

    m = {"dist_code": g.dist_code.iloc[0], "hill1": hill1(A.dropna().mean(axis=1).to_numpy()), "stability": stability(total, d["resid"].to_numpy()),
         "resistance": resistance(index, ev), "recovery": recovery(index, ev), "vulnerability": vulnerability(index),
         "vuln_drought": vulnerability(index, drought_years), "n_events": n_usable_events(index, ev),
         "n_drought_years": int(len(drought_years)), "resp_div": resp_div(b, w), "resp_divergence": resp_divergence(b),
         "phi_sync": dec["phi_sync"], "mean_crop_cv": dec["mean_crop_cv"], "n_crops_value": len(bal), "value_coverage": float(cov),
         "n_crops_beta": int(np.isfinite(b).sum())}
    pn = pd.DataFrame({"dist_code": m["dist_code"], "year": years, "anom": d["anom"].to_numpy(), "index": d["index"].to_numpy(),
                       "spei": spei.to_numpy(), "hill1_tv": hill1_tv(A, years, 5).to_numpy(), "hill1_tv3": hill1_tv(A, years, 3).to_numpy()})
    ca = pd.DataFrame([{"dist_code": m["dist_code"], "year": y, "crop": c, "yield_anom": anoms[c][i], "value_share": share[c], "resp_beta": betas[c]}
                       for c in use for i, y in enumerate(years)])
    return m, pn, ca


def run(crops, climate, years, spei_col):
    """crops: standard table; climate: dist_code, year, <spei_col>. Returns (metrics, panel, crop_anomalies, dropped)."""
    clim = climate[["dist_code", "year", spei_col]].rename(columns={spei_col: "spei12_w"})
    clim = clim[clim.year.between(years[0], years[-1])]
    ev = find_events(clim.rename(columns={"dist_code": "iso3"}), year_min=years[0], year_max=years[-1])
    ev_by = {k: v for k, v in ev.groupby("iso3")}
    cg = {k: v for k, v in clim.groupby("dist_code")}
    ms, ps, cs, dropped = [], [], [], []
    for code, g in crops.groupby("dist_code"):
        if code not in cg or cg[code].spei12_w.notna().sum() < len(years):
            dropped.append((code, "no complete district SPEI"))
            continue
        r = one_district(g, cg[code], years, ev_by.get(code, pd.DataFrame({"year": []})))
        if r[0] is None:
            dropped.append((code, "value index: too few balanced crops or coverage < %.0f%%" % (MIN_COVERAGE * 100)))
            continue
        ms.append(r[0]); ps.append(r[1]); cs.append(r[2])
    return pd.DataFrame(ms), pd.concat(ps, ignore_index=True), pd.concat(cs, ignore_index=True), pd.DataFrame(dropped, columns=["dist_code", "reason"])
