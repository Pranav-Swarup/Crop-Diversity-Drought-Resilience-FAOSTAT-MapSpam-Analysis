"""Unit tests for src/india (synthetic data only; no downloads needed). Run: python -m pytest tests/test_india.py"""
import numpy as np
import pandas as pd
import pytest

from src.india import clean, conley, geo, metrics, panel
from src.metrics.panel import fe_fit

YEARS = list(range(1998, 2023))


def crop_table(code="D1", crops=("a", "b", "c", "d"), years=YEARS, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for c_i, c in enumerate(crops):
        for y in years:
            area = 100.0 * (c_i + 1)
            yld = 2.0 + 0.02 * (y - years[0]) + rng.normal(0, 0.1)
            rows.append(dict(dist_code=code, year=y, crop=c, area_ha=area, prod_t=area * yld, yield_t_ha=yld,
                             price_usd_t=100.0, value_const=area * yld * 100.0))
    return pd.DataFrame(rows)


# ---- diversity -----------------------------------------------------------
def test_hill1_tv_uses_only_past_years():
    g = crop_table()
    A = metrics.area_matrix(g, YEARS)
    A.loc["a", 2003] = 1e9                                   # a huge shock in 2003 must not enter t = 2003
    h = metrics.hill1_tv(A, YEARS, 5)
    assert h[1998:2002].isna().all() and h[2003] == pytest.approx(h[2003])
    clean_h = metrics.hill1_tv(metrics.area_matrix(g, YEARS), YEARS, 5)
    assert h[2003] == pytest.approx(clean_h[2003])           # unaffected by its own year
    assert h[2004] < clean_h[2004]                           # but the shock enters the next year


def test_hill1_tv_equal_shares_is_number_of_crops():
    g = crop_table()
    g["area_ha"] = 100.0
    h = metrics.hill1_tv(metrics.area_matrix(g, YEARS), YEARS, 5)
    assert np.allclose(h.dropna().to_numpy(), 4.0)


def test_area_matrix_preserves_unreported_area():
    g = crop_table()
    g = g[~((g.crop == "a") & (g.year == 2005))]
    A = metrics.area_matrix(g, YEARS)
    assert pd.isna(A.loc["a", 2005]) and A.shape == (4, len(YEARS))


# ---- value index rules ---------------------------------------------------
def test_district_value_drops_unbalanced_crops_and_reports_coverage():
    g = crop_table()
    g.loc[(g.crop == "d") & (g.year == 2010), "value_const"] = np.nan        # crop d has a gap: not balanced
    bal, cov = metrics.district_value(g, YEARS)
    assert list(bal.index) == ["a", "b", "c"]
    assert 0.5 < cov < 1.0


def both_districts(bad):
    """Good district G plus a bad one B; returns crops and a SPEI table."""
    g = pd.concat([crop_table("G", seed=1), bad])
    spei = np.sin(np.arange(len(YEARS)) * 1.3) * 1.5
    return g, spei


def test_district_with_too_few_crops_is_dropped():
    g, spei = both_districts(crop_table("B", crops=("a", "b")))
    clim = pd.concat([pd.DataFrame({"dist_code": c, "year": YEARS, "spei12_w": spei}) for c in ("G", "B")])
    m, p, ca, dropped = metrics.run(g, clim, YEARS, "spei12_w")
    assert list(m.dist_code) == ["G"] and list(dropped.dist_code) == ["B"] and "too few" in dropped.reason.iloc[0]


def test_run_on_synthetic_district_gives_finite_faces():
    g = crop_table()
    spei = np.sin(np.arange(len(YEARS)) * 1.3) * 1.5
    clim = pd.DataFrame({"dist_code": "D1", "year": YEARS, "spei12_w": spei})
    m, p, ca, dropped = metrics.run(g, clim, YEARS, "spei12_w")
    assert len(m) == 1 and dropped.empty
    assert np.isfinite(m[["hill1", "stability", "vulnerability", "phi_sync"]].to_numpy()).all()
    assert (p.groupby("dist_code").size() == len(YEARS)).all()
    assert p.spei.round(6).tolist() == pytest.approx(np.round(spei, 6).tolist())


def test_run_drops_district_without_complete_spei():
    g, spei = both_districts(crop_table("B", seed=2))
    clim = pd.concat([pd.DataFrame({"dist_code": "G", "year": YEARS, "spei12_w": spei}),
                      pd.DataFrame({"dist_code": "B", "year": YEARS[:-1], "spei12_w": spei[:-1]})])
    m, p, ca, dropped = metrics.run(g, clim, YEARS, "spei12_w")
    assert list(m.dist_code) == ["G"] and dropped.reason.iloc[0] == "no complete district SPEI"


# ---- cleaning rules ------------------------------------------------------
def test_season_rule_prefers_total_else_sums():
    df = pd.DataFrame({"id": range(5), "y": 2000, "district_code": ["1", "1", "1", "2", "2"], "crop_name": "Rice",
                       "season": ["Kharif", "Rabi", "Total", "Kharif", "Rabi"], "area": [10, 5, 15, 7, 3], "production": [20, 10, 30, 14, 6]})
    r = clean.season_rule(df).set_index("district_code")
    assert r.loc["1", "area_ha"] == 15 and r.loc["1", "rule"] == "total"        # not 30: no double count
    assert r.loc["2", "area_ha"] == 10 and r.loc["2", "rule"] == "sum"


def test_split_flags_detects_parent_that_lost_territory():
    years = list(range(1998, 2010))
    rows = []
    for d in ["A", "B", "C", "D", "E"]:
        for y in years:
            area = 100.0
            if d == "A" and y >= 2004:
                area = 40.0                                   # parent lost territory to the new district E
            if d == "E" and y < 2004:
                continue
            rows.append(dict(district_code=d, y=y, area_ha=area))
    a = pd.DataFrame(rows)
    state_of = pd.Series("S", index=list("ABCDE"))
    first = pd.Series({"A": 1998, "B": 1998, "C": 1998, "D": 1998, "E": 2004})
    flags = clean.split_flags(a, state_of, first, years)
    assert set(flags) == {"A"}


# ---- matching ------------------------------------------------------------
def test_name_normalisation():
    assert geo.norm("Wardha") == geo.norm("Vardha")
    assert geo.norm("Kolhapur District") == geo.norm("Kolhapur")
    assert geo.norm_state("Jammu & Kashmir") == geo.norm_state("Jammu and Kashmir")
    assert geo.norm("Pune") != geo.norm("Nashik")


# ---- panel machinery -----------------------------------------------------
def fake_panel(G=40, T=12, seed=1):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame([(f"d{g}", f"S{g % 4}", 1998 + t) for g in range(G) for t in range(T)], columns=["dist_code", "state_name", "year"])
    d["sy"] = d.state_name + "_" + d.year.astype(str)
    d["spei"] = rng.normal(size=len(d))
    d["z"] = rng.normal(size=len(d))
    d["spei_z"] = d.spei * d.z
    d["anom"] = 0.05 * d.spei + 0.2 * d.spei_z + rng.normal(0, 0.1, len(d))
    return d


def test_fast_t_matches_fe_fit():
    d = fake_panel().sort_values(["dist_code", "year"]).reset_index(drop=True)
    terms = ["spei", "spei_z", "z"]
    f = fe_fit(d, "anom", terms, cluster="dist_code", fe=panel.FE)
    q = panel.fe_basis(d)
    k_fe = d.dist_code.nunique() + d.sy.nunique() - 1
    t, b = panel.fast_t(q, d.anom.to_numpy(), d[terms].to_numpy(), d.dist_code.to_numpy(), k_fe, 1)
    assert b == pytest.approx(f["params"]["spei_z"])
    se = np.sqrt(f["cov"].loc["spei_z", "spei_z"])
    n, k = len(d), len(terms)
    c_fast = (n - 1) / (n - k - k_fe + 1)
    c_fe = (n - 1) / (n - k - (d.sy.nunique() - 1))               # fe_fit: counts the state-year dummies only
    assert t == pytest.approx(f["params"]["spei_z"] / se * np.sqrt(c_fe / c_fast), rel=1e-6)


def test_conley_tiny_cutoff_no_lag_is_hc0():
    d = fake_panel().sort_values(["dist_code", "year"]).reset_index(drop=True)
    cent = pd.DataFrame({"lon": np.linspace(70, 90, 40), "lat": np.linspace(10, 30, 40)}, index=[f"d{g}" for g in range(40)])
    beta, covs, G, n, D = conley.fit_conley(d, cent, cutoffs=(1,), lag=0)
    from src.metrics.panel import residualise
    r = residualise(d, ["anom"] + conley.TERMS, panel.FE)
    xt, yt = r[:, 1:], r[:, 0]
    bread = np.linalg.inv(xt.T @ xt)
    u = yt - xt @ (bread @ xt.T @ yt)
    hc0 = bread @ (xt.T * u ** 2) @ xt @ bread
    k_fe = G + d.sy.nunique() - 1
    corr = (n - 1) / (n - xt.shape[1] - k_fe + 1)
    assert np.allclose(covs[1], corr * hc0, rtol=1e-8)


def test_conley_se_grows_with_spatial_dependence():
    rng = np.random.default_rng(3)
    G, T = 60, 15
    d = pd.DataFrame([(f"d{g}", f"S{g % 3}", 1998 + t) for g in range(G) for t in range(T)], columns=["dist_code", "state_name", "year"])
    d["sy"] = d.state_name + "_" + d.year.astype(str)
    sh = rng.normal(size=(3, T))                             # one shock per state-year, shared by all its districts (absorbed by sy FE)
    shock = sh[d.dist_code.str[1:].astype(int) % 3, d.year - 1998]
    dshock = rng.normal(size=(G // 6, T))                    # plus a shock shared by clusters of 6 nearby districts, not absorbed
    nb = dshock[d.dist_code.str[1:].astype(int) // 6 % (G // 6), d.year - 1998]
    d["z"] = rng.normal(size=len(d)); d["spei"] = nb + rng.normal(0, .3, len(d)); d["spei_z"] = d.spei * d.z
    d["anom"] = shock + 0.3 * nb + rng.normal(0, .1, len(d))         # the outcome moves with the same local shock
    cluster_id = np.arange(G) // 6 % (G // 6)
    cent = pd.DataFrame({"lon": 80.0 + cluster_id * 5 + rng.normal(0, .05, G), "lat": np.full(G, 20.0)}, index=[f"d{g}" for g in range(G)])
    _, covs, *_ = conley.fit_conley(d, cent, cutoffs=(1, 50), lag=0)
    assert covs[50][0, 0] > covs[1][0, 0]                   # allowing spatial correlation raises the variance of the spei slope


def test_haversine_known_distance():
    D = conley.haversine_km(np.array([77.2090, 72.8777]), np.array([28.6139, 19.0760]))   # Delhi, Mumbai
    assert D[0, 1] == pytest.approx(1148, abs=15) and D[0, 0] == 0 and np.allclose(D, D.T)


# ---- validation table ----------------------------------------------------
def test_validation_table_documented_years_have_sources():
    clim = pd.DataFrame({"dist_code": ["x"] * len(YEARS), "year": YEARS, "spei12_w": -1.5})
    t = panel.validation_table(clim, YEARS, "spei12_w")
    doc = t[t.documented_drought_year]
    assert set(doc.year) == {2002, 2004, 2009, 2014, 2015}
    assert doc.url.str.startswith("https://").all() and doc.monsoon_departure_lpa_pct.lt(0).all()
    assert t[t.documented_drought_year.isna()].url.isna().all() and (t.share_drought == 1).all()
