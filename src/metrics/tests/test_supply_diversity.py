"""Unit tests for the supply-source diversity. Run: python -m pytest src/metrics/tests/test_supply_diversity.py"""
import numpy as np
import pandas as pd

from src.crops.supply_diversity import complete_spei, country_supply, domestic_share, partner_weights

YEARS = np.arange(1993, 2024)
T = np.arange(len(YEARS))


def wave(k):
    """Zero-mean cosine with k cycles over the 31 years; different k are exactly orthogonal."""
    return 1.5 * np.cos(2 * np.pi * k * T / len(T))


def table(**series):
    return pd.DataFrame(series, index=pd.Index(YEARS, name="year"))


def test_domestic_share_and_shares_sum_to_one():
    trade = pd.DataFrame({"prod_t": [60.0, 40.0, np.nan], "import_t": [40.0, 60.0, 10.0]})
    assert np.isclose(domestic_share(trade), 100 / 200)           # the year with missing production is skipped
    spei = table(AAA=wave(1), BBB=wave(2), CCC=wave(3))
    row, _ = country_supply("AAA", 0.5, pd.Series({"BBB": 30.0, "CCC": 10.0}), spei)
    assert np.isclose(row["s_dom"], 0.5)
    w, cover = partner_weights(pd.Series({"BBB": 30.0, "CCC": 10.0}), spei)
    assert np.isclose(w.sum(), 1) and np.isclose(w["BBB"], 0.75) and cover == 1.0
    assert np.isnan(domestic_share(pd.DataFrame({"prod_t": [np.nan], "import_t": [1.0]})))


def test_partners_with_the_same_series_are_one_bet():
    spei = table(AAA=wave(1), BBB=wave(1), CCC=wave(1))
    row, _ = country_supply("AAA", 0.3, pd.Series({"BBB": 5.0, "CCC": 5.0}), spei)
    assert np.isclose(row["supply_div_exp"], 1.0) and np.isclose(row["supply_ratio"] * row["supply_div"], 1.0)
    assert row["supply_div"] > 1.5 and np.isclose(row["partner_corr"], 1.0)
    assert np.isclose(row["import_div_exp"], 1.0)


def test_independent_partners_keep_their_count():
    spei = table(AAA=wave(1), BBB=wave(2), CCC=wave(3))
    row, _ = country_supply("AAA", 0.4, pd.Series({"BBB": 5.0, "CCC": 5.0}), spei)
    assert np.isclose(row["supply_div_exp"], row["supply_div"]) and np.isclose(row["supply_ratio"], 1.0)
    assert np.isclose(row["partner_corr"], 0.0, atol=1e-12)
    assert np.isclose(row["supply_div"], np.exp(-(0.4 * np.log(0.4) + 2 * 0.3 * np.log(0.3))))


def test_partner_dry_share_hand_made():
    own = np.zeros(len(T)); own[[3, 4]] = -1.2                    # own drought years 1996, 1997
    b = np.zeros(len(T)); b[[3, 10]] = -1.0                       # exactly -1.0 counts as dry
    c = np.zeros(len(T)); c[[3, 4, 10]] = -2.0
    spei = table(AAA=own, BBB=b, CCC=c)
    row, expo = country_supply("AAA", 0.2, pd.Series({"BBB": 1.0, "CCC": 3.0}), spei)
    e = expo.set_index("year")
    assert np.isclose(e.loc[1996, "partner_dry_share"], 1.0)      # both partners dry
    assert np.isclose(e.loc[1997, "partner_dry_share"], 0.75)     # only CCC (weight 0.75)
    assert np.isclose(e.loc[2003, "partner_dry_share"], 1.0)
    assert np.isclose(e.loc[2000, "partner_dry_share"], 0.0)
    assert np.isclose(e.loc[1997, "partner_spei"], 0.25 * 0.0 + 0.75 * -2.0)
    assert np.isclose(row["partner_dry_own_dry"], (1.0 + 0.75) / 2)
    other = e.drop(index=[1996, 1997])["partner_dry_share"].mean()
    assert np.isclose(row["partner_dry_other"], other)


def test_partner_without_climate_series_is_dropped():
    clim = pd.DataFrame({"iso3": np.repeat(["AAA", "BBB", "CCC"], len(T)), "year": np.tile(YEARS, 3),
                         "spei12_w": np.concatenate([wave(1), wave(2), wave(3)])})
    clim.loc[(clim["iso3"] == "CCC") & (clim["year"] == 2000), "spei12_w"] = np.nan   # incomplete -> dropped
    spei = complete_spei(clim)
    assert list(spei.columns) == ["AAA", "BBB"]
    row, _ = country_supply("AAA", 0.5, pd.Series({"BBB": 60.0, "CCC": 20.0, "ZZZ": 20.0}), spei)
    assert row["n_partners"] == 1 and np.isclose(row["partner_cover"], 0.6)
    assert np.isclose(row["supply_div"], 2.0)                      # s = [0.5, 0.5] after renormalising
    assert country_supply("AAA", 0.5, pd.Series({"CCC": 5.0}), spei) is None


def test_no_negative_similarity_gain():
    spei = table(AAA=wave(1), BBB=-wave(1))                        # anti-correlated: counts as fully distinct
    row, _ = country_supply("AAA", 0.5, pd.Series({"BBB": 1.0}), spei)
    assert np.isclose(row["supply_div_exp"], 2.0) and np.isclose(row["partner_corr"], -1.0)
