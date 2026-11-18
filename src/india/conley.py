"""Conley spatial-HAC standard errors for the phase 4 district panel -> outputs/stats/india_conley.csv.

Same model and estimation sample as the core panel (`src.india.panel`): anom ~ spei + spei*z + z, district and state-year
fixed effects. Only the covariance changes. District clustering assumes independent districts; neighbouring districts share
0.5 degree SPEI cells and monsoons, so errors are spatially correlated. Conley (1999) with the Hsiang (2010) form:

    V = (X'X)^-1 [ sum_t sum_i sum_j K_s(d_ij) s_it s_jt'  +  sum_i sum_t sum_{l=1..L} K_t(l) (s_it s_i,t-l' + s_i,t-l s_it') ] (X'X)^-1
    s_it = x_it * u_it  (x, u residualised on the fixed effects)

K_s(d) = max(0, 1 - d/cutoff) (Bartlett, distance between district centroids in km), K_t(l) = 1 - l/(L+1) (Bartlett, Newey-West),
L = 5 years. The spatial sum includes the own district (i = j), so cutoff -> 0 and L = 0 gives the heteroskedasticity-robust
variance. Small-sample factor (n-1)/(n-k_fe-k) as in fe_fit without the cluster term. p-values and CIs use a t distribution
with G-1 degrees of freedom, as in fe_fit. Cutoffs 100, 200, 400 km; the cluster-by-district and cluster-by-state rows of
the core model are repeated for comparison.

Run: python -m src.india.conley
"""
import geopandas as gpd
import numpy as np
import pandas as pd
import scipy.stats as st

from ..common import paths
from ..metrics.panel import residualise
from . import panel as P

CUTOFFS = (100, 200, 400)
LAG = 5
TERMS = ["spei", "spei_z", "z"]


def centroids():
    """District centroid (lon, lat) from districts.geojson, computed in an equal-area-friendly projected CRS (EPSG:7755, India)."""
    g = gpd.read_file(P.IN / "districts.geojson")
    c = g.to_crs(7755).geometry.centroid.to_crs(4326)
    return pd.DataFrame({"dist_code": g.dist_code.astype(str).to_numpy(), "lon": c.x.to_numpy(), "lat": c.y.to_numpy()}).set_index("dist_code")


def haversine_km(lon, lat):
    lo, la = np.deg2rad(lon), np.deg2rad(lat)
    dlat, dlon = la[:, None] - la[None, :], lo[:, None] - lo[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(la)[:, None] * np.cos(la)[None, :] * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def conley_cov(xt, u, dist_idx, year_idx, D, cutoff, lag, bread):
    """xt (n,k) and u (n,) residualised; dist_idx/year_idx integer codes; D (G,G) distances in km. Returns the (k,k) covariance
    before the small-sample factor."""
    n, k = xt.shape
    G, T = D.shape[0], year_idx.max() + 1
    S = np.zeros((G, T, k))
    S[dist_idx, year_idx] = xt * u[:, None]            # balanced or not: missing cells stay zero
    K = np.clip(1 - D / cutoff, 0, None)               # includes the diagonal (= 1)
    meat = np.zeros((k, k))
    for t in range(T):                                  # spatial part, same year
        s = S[:, t, :]
        meat += s.T @ K @ s
    for l in range(1, lag + 1):                         # serial part, same district, both directions
        w = 1 - l / (lag + 1)
        a, b = S[:, l:, :].reshape(-1, k), S[:, :-l, :].reshape(-1, k)
        m = a.T @ b
        meat += w * (m + m.T)
    return bread @ meat @ bread


def fit_conley(d, cent, cutoffs=CUTOFFS, lag=LAG):
    d = d.sort_values(["dist_code", "year"]).reset_index(drop=True)
    r = residualise(d, ["anom"] + TERMS, P.FE)
    yt, xt = r[:, 0], r[:, 1:]
    n, k = xt.shape
    bread = np.linalg.inv(xt.T @ xt)
    beta = bread @ xt.T @ yt
    u = yt - xt @ beta
    codes, dist_idx = np.unique(d.dist_code.to_numpy(), return_inverse=True)
    year_idx = (d.year - d.year.min()).to_numpy()
    c = cent.reindex(codes)
    assert c.notna().all().all(), "district without a centroid"
    D = haversine_km(c.lon.to_numpy(), c.lat.to_numpy())
    k_fe = len(codes) + d.sy.nunique() - 1
    corr = (n - 1) / (n - k - k_fe + 1)
    out = {}
    for cut in cutoffs:
        out[cut] = corr * conley_cov(xt, u, dist_idx, year_idx, D, cut, lag, bread)
    return pd.Series(beta, index=TERMS), out, len(codes), n, D


def rows_from(model, beta, cov, G, n, note, n_districts=None):
    se = np.sqrt(np.diag(cov))
    tcrit = st.t.ppf(0.975, G - 1)
    return [{"model": model, "term": t, "beta": b, "se": s, "ci_lo": b - tcrit * s, "ci_hi": b + tcrit * s,
             "p": 2 * st.t.sf(abs(b / s), G - 1), "n_obs": n, "n_districts": G if n_districts is None else n_districts, "n_clusters": G if n_districts is not None else None, "note": note} for t, b, s in zip(TERMS, beta, se)]


def main():
    cent = centroids()
    rows = []
    for window in ("core", "ext5"):
        data, col, years, _ = P.build(window)
        d = P.prep(data["core"]["panel"])
        # reference: the same model with district and state clustering (src.metrics.panel.fe_fit)
        for cl in ("dist_code", "state_name"):
            f = P.fe_fit(d, "anom", TERMS, cluster=cl, fe=P.FE)
            rows += rows_from(f"{window}:cluster_{cl.split('_')[0]}", f["params"].to_numpy(), f["cov"].to_numpy(),
                              f["n_clusters"], f["nobs"], f"CR1 clustered by {cl}", d.dist_code.nunique())
        beta, covs, G, n, D = fit_conley(d, cent)
        f = P.fe_fit(d, "anom", TERMS, cluster="dist_code", fe=P.FE)
        assert np.allclose(beta.to_numpy(), f["params"].to_numpy(), atol=1e-8), "Conley and fe_fit disagree on the coefficients"
        for cut, cov in covs.items():
            rows += rows_from(f"{window}:conley_{cut}km", beta.to_numpy(), cov, G, n, f"Bartlett {cut} km, serial lag {LAG}, t(G-1)")
        near = (D < 200).sum(axis=1) - 1
        print(f"[{window}] districts {G}, obs {n}; neighbours within 200 km: median {np.median(near):.0f}, max {near.max()}; "
              f"variable {col}")
    out = pd.DataFrame(rows).assign(spei_variable=col)
    assert out.se.gt(0).all() and np.isfinite(out.p).all()
    out.to_csv(paths.STATS_DIR / "india_conley.csv", index=False)
    show = out[out.term == "spei_z"][["model", "beta", "se", "p"]].round(4)
    print("\nspei x diversity, by covariance estimator:\n" + show.to_string(index=False))


if __name__ == "__main__":
    main()
