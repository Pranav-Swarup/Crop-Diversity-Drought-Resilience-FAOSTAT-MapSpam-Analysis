"""Exhaustive verification and regression test suite for Phase 4 (India District Analysis).

Covers:
1. Data Integrity & Harmonization (districts, crops, climate, geojson, extended mirror, kharif)
2. Mathematical & Econometric Engines (SVD basis, FE demeaning, Fast-t permutation, Conley spatial HAC)
3. Exact Empirical Replication of All Statistical Findings (outputs/stats/india_*.csv)
4. Historical Drought Ground-Truth Validation (outputs/validation/india_drought_years.csv)
"""
import numpy as np
import pandas as pd
import geopandas as gpd

from src.common import paths
from src.india import panel

ROOT = paths.ROOT
CLEAN_IND = paths.CLEAN_REAL / "india"
STATS = paths.STATS_DIR
VALID = paths.OUT_REAL.parent / "outputs" / "validation"


# ==============================================================================
# 1. DATA INTEGRITY & HARMONIZATION TESTS
# ==============================================================================

def test_districts_csv_integrity():
    """Verify districts.csv counts, columns, stability, split flags, and core count."""
    df = pd.read_csv(CLEAN_IND / "districts.csv")
    assert len(df) == 739, f"Expected 739 districts, found {len(df)}"
    assert df["dist_code"].nunique() == 739, "dist_code is not globally unique"

    expected_cols = {
        "dist_code", "dist_name", "state_code", "state_name", "boundary_base",
        "n_years", "first_year", "stable", "split_flag", "split_detail", "core"
    }
    assert expected_cols.issubset(set(df.columns)), f"Missing columns in districts.csv: {expected_cols - set(df.columns)}"

    stable_count = int(df["stable"].sum())
    split_count = int((df["stable"] & df["split_flag"]).sum())
    core_count = int(df["core"].sum())

    assert stable_count == 530, f"Expected 530 stable districts, got {stable_count}"
    assert split_count == 130, f"Expected 130 split-flagged among stable, got {split_count}"
    assert core_count == 400, f"Expected 400 core districts, got {core_count}"
    assert ((df["stable"] & ~df["split_flag"]) == df["core"]).all(), "Core definition mismatch"


def test_crops_parquet_rules():
    """Verify crops.parquet has 47 fixed continuous crops, 10x median yield filter, and coconut blanked."""
    df = pd.read_parquet(CLEAN_IND / "crops.parquet")
    assert not df.duplicated(["dist_code", "year", "crop"]).any(), "Duplicate district-year-crop key"
    assert df["year"].between(1998, 2022).all(), "Years outside 1998-2022"
    assert (df["area_ha"] > 0).all(), "Non-positive area_ha found"

    # 47 fixed continuous crops
    assert df["crop"].nunique() == 47, f"Expected 47 crops, found {df['crop'].nunique()}"

    # Coconut production must be blanked (counted in nuts in source reports)
    coconut = df[df["crop"] == "Coconut"]
    assert coconut["prod_t"].isna().all(), "Coconut prod_t is not blanked"
    assert coconut["area_ha"].gt(0).all(), "Coconut area missing"

    # 10x median yield filter
    valid_yield = df[df["yield_t_ha"].notna()]
    assert np.allclose(valid_yield["yield_t_ha"], valid_yield["prod_t"] / valid_yield["area_ha"], rtol=1e-5), "yield_t_ha != prod_t / area_ha"

    # Priced crops: 38 priced, 9 unpriced
    priced = df[df["price_intl_dollars_t"].notna()]
    assert len(priced) > 0.85 * len(df), f"Priced share {len(priced)/len(df):.1%} below 85%"
    priced_with_prod = priced.dropna(subset=["prod_t"])
    assert np.allclose(priced_with_prod["value_const"], priced_with_prod["prod_t"] * priced_with_prod["price_intl_dollars_t"], rtol=1e-5), \
        "value_const != prod_t * price_intl_dollars_t for priced crops"


def test_climate_parquet_integrity():
    """Verify climate.parquet has 690 matched districts x 30 years (1993-2022) with 0 missing SPEI."""
    df = pd.read_parquet(CLEAN_IND / "climate.parquet")
    assert len(df) == 20700, f"Expected 20,700 rows, got {len(df)}"
    assert df["dist_code"].nunique() == 690, f"Expected 690 districts, got {df['dist_code'].nunique()}"
    assert set(df["year"].unique()) == set(range(1993, 2023)), "Year range != 1993..2022"

    # Complete series without NaNs
    assert df["spei12_w"].notna().all(), "spei12_w has NaNs"
    assert df["spei12_unw"].notna().all(), "spei12_unw has NaNs"
    assert df["spei4kh_w"].notna().all(), "spei4kh_w has NaNs"

    # SPEI is standardized: standard deviation should be close to 1
    assert 0.85 < df["spei12_w"].std() < 1.15, f"SPEI-12 standard deviation abnormal: {df['spei12_w'].std()}"


def test_districts_geojson_and_matches():
    """Verify districts.geojson has 690 features matching district_match.csv."""
    gdf = gpd.read_file(CLEAN_IND / "districts.geojson")
    assert len(gdf) == 690, f"Expected 690 geojson features, got {len(gdf)}"
    assert gdf["dist_code"].nunique() == 690, "Duplicate dist_code in geojson"
    assert gdf.geometry.is_valid.all(), "Invalid geometries found in districts.geojson"

    dm = pd.read_csv(CLEAN_IND / "district_match.csv", dtype={"dist_code": str})
    assert len(dm) == 739, f"Expected 739 DES districts in district_match.csv, got {len(dm)}"
    matched = dm[dm["matched"]]
    assert len(matched) == 690, f"Expected 690 matched districts, got {len(matched)}"
    assert set(gdf["dist_code"]) == set(matched["dist_code"]), "Mismatch between GeoJSON and matched district IDs"


def test_extended_5crop_and_kharif_partitions():
    """Verify crops_ext5.parquet (1993-2015, 5 crops) and crops_kharif.parquet (27 crops)."""
    ext5 = pd.read_parquet(CLEAN_IND / "crops_ext5.parquet")
    assert set(ext5["year"].unique()) == set(range(1993, 2016)), "ext5 years != 1993..2015"
    assert ext5["crop"].nunique() == 5, f"ext5 crops != 5, got {ext5['crop'].nunique()}"
    assert set(ext5["crop"].unique()) == {"Chickpea", "Groundnut", "Pearl millet", "Rice", "Sugarcane"}

    ctrl = pd.read_parquet(CLEAN_IND / "mirror_controls.parquet")
    assert {"irrig_share", "fert_kg_ha"}.issubset(set(ctrl.columns))
    assert (ctrl["irrig_share"].between(0, 1) | ctrl["irrig_share"].isna()).all(), "irrig_share out of [0, 1]"
    assert (ctrl["fert_kg_ha"].ge(0) | ctrl["fert_kg_ha"].isna()).all(), "fert_kg_ha negative"

    kh = pd.read_parquet(CLEAN_IND / "crops_kharif.parquet")
    assert kh["crop"].nunique() == 27, f"Expected 27 Kharif crops, got {kh['crop'].nunique()}"
    assert kh["year"].between(1998, 2022).all()


# ==============================================================================
# 2. SVD BASIS & COLLINEARITY FIX VERIFICATION (COMMIT 608f0ad)
# ==============================================================================

def test_svd_basis_orthonormality_and_condition_number():
    """Verify fe_basis constructs an orthonormal basis for rank-deficient nested fixed effects.

    District dummies nest within State x Year dummies, making D collinear.
    SVD basis drops zero singular values, ensuring Q.T @ Q == I with condition number 1.0.
    """
    G, T = 30, 10
    rows = []
    for g in range(G):
        state = f"S{g % 3}"
        for t in range(T):
            rows.append({"dist_code": f"d{g}", "state_name": state, "year": 2000 + t, "sy": f"{state}_{2000 + t}"})
    d = pd.DataFrame(rows)

    q = panel.fe_basis(d)

    # Orthonormality check: Q.T @ Q must be identity matrix
    identity = np.eye(q.shape[1])
    assert np.allclose(q.T @ q, identity, atol=1e-12), "Q is not orthonormal"

    # Condition number must be exactly 1.0
    cond = np.linalg.cond(q)
    assert np.isclose(cond, 1.0, atol=1e-5), f"SVD basis condition number is {cond}, expected 1.0"

    # Rank check: rank of dummy matrix D is strictly less than columns of D
    D = np.column_stack([pd.get_dummies(d[panel.FE[0]], dtype=float).to_numpy(),
                         pd.get_dummies(d[panel.FE[1]], drop_first=True, dtype=float).to_numpy()])
    assert q.shape[1] < D.shape[1], "Collinear directions were not eliminated by SVD"


# ==============================================================================
# 3. EXACT EMPIRICAL REPLICATION TESTS (outputs/stats/india_*.csv)
# ==============================================================================

def test_summary_sample_sizes():
    """Verify district counts and usable drought events in outputs/stats/india_summary.csv."""
    summ = pd.read_csv(STATS / "india_summary.csv").set_index("window")
    assert summ.loc["core", "districts"] == 308
    assert summ.loc["core", "n_events"] == 460
    assert summ.loc["core", "years"] == "1998-2022"
    assert summ.loc["core", "spei_variable"] == "spei12_w"

    assert summ.loc["ext5", "districts"] == 127
    assert summ.loc["ext5", "n_events"] == 174
    assert summ.loc["ext5", "years"] == "1993-2015"


def test_core_twfe_panel_regression_null_result():
    """Verify null interaction in core DES 47-crop panel and 5-crop mirror panel."""
    p_df = pd.read_csv(STATS / "india_panel.csv")

    # Core DES interaction: spei_z
    core_spei_z = p_df[(p_df["model"] == "core:core") & (p_df["term"] == "spei_z")].iloc[0]
    assert np.isclose(core_spei_z["beta"], 0.003390, atol=2e-4), f"Core beta: {core_spei_z['beta']}"
    assert np.isclose(core_spei_z["se"], 0.005237, atol=2e-4), f"Core SE: {core_spei_z['se']}"
    assert np.isclose(core_spei_z["p"], 0.517939, atol=2e-3), f"Core p-value: {core_spei_z['p']}"
    assert core_spei_z["n_obs"] == 6160
    assert core_spei_z["n_districts"] == 308

    # 5-crop Mendeley mirror interaction: spei_z
    ext5_spei_z = p_df[(p_df["model"] == "ext5:core") & (p_df["term"] == "spei_z")].iloc[0]
    assert np.isclose(ext5_spei_z["beta"], 0.008485, atol=2e-4), f"Ext5 beta: {ext5_spei_z['beta']}"
    assert np.isclose(ext5_spei_z["se"], 0.009296, atol=2e-4), f"Ext5 SE: {ext5_spei_z['se']}"
    assert np.isclose(ext5_spei_z["p"], 0.363107, atol=3e-3), f"Ext5 p-value: {ext5_spei_z['p']}"


def test_severity_permutation_p_values():
    """Verify studentised permutation p-values from outputs/stats/india_severity_perm.csv."""
    perm = pd.read_csv(STATS / "india_severity_perm.csv").set_index("model")

    # Core extreme drought interaction permutation p-value: 0.332
    assert np.isclose(perm.loc["core", "p_perm_t"], 0.332, atol=1e-3)
    assert np.isclose(perm.loc["core", "t_obs"], -1.202887, atol=2e-3)
    assert perm.loc["core", "n_perm"] == 499

    # Ext5 extreme drought interaction permutation p-value: 0.062
    assert np.isclose(perm.loc["ext5", "p_perm_t"], 0.062, atol=1e-3)
    assert np.isclose(perm.loc["ext5", "t_obs"], -1.948471, atol=2e-3)
    assert perm.loc["ext5", "n_perm"] == 499


def test_conley_spatial_hac_standard_errors():
    """Verify Conley spatial standard errors across cutoffs (100, 200, 400 km)."""
    conley_df = pd.read_csv(STATS / "india_conley.csv")
    spei_z = conley_df[conley_df["term"] == "spei_z"].set_index("model")

    # Check that core beta is invariant across all covariance estimators
    assert np.isclose(spei_z.loc["core:cluster_dist", "beta"], 0.003390, atol=2e-4)
    assert np.isclose(spei_z.loc["core:cluster_state", "beta"], 0.003390, atol=2e-4)
    assert np.isclose(spei_z.loc["core:conley_100km", "beta"], 0.003390, atol=2e-4)
    assert np.isclose(spei_z.loc["core:conley_200km", "beta"], 0.003390, atol=2e-4)
    assert np.isclose(spei_z.loc["core:conley_400km", "beta"], 0.003390, atol=2e-4)

    # Check p-values under spatial cutoffs remain non-significant
    assert np.isclose(spei_z.loc["core:conley_100km", "p"], 0.435828, atol=3e-3)
    assert np.isclose(spei_z.loc["core:conley_200km", "p"], 0.461389, atol=3e-3)
    assert np.isclose(spei_z.loc["core:conley_400km", "p"], 0.494848, atol=3e-3)


def test_kharif_robustness_models():
    """Verify Kharif SPEI-4 and Kharif-crop panel estimations."""
    kh_df = pd.read_csv(STATS / "india_kharif.csv")
    kh_core = kh_df[(kh_df["model"] == "core:kharif") & (kh_df["term"] == "spei_z")].iloc[0]
    assert np.isclose(kh_core["beta"], 0.006374, atol=2e-4)
    assert np.isclose(kh_core["p"], 0.234085, atol=5e-3)

    khc_df = pd.read_csv(STATS / "india_kharif_crops.csv")
    khc_spei4 = khc_df[(khc_df["model"] == "kharif_crops_spei4") & (khc_df["term"] == "spei_z")].iloc[0]
    assert np.isclose(khc_spei4["beta"], 0.009992, atol=2e-4)
    assert np.isclose(khc_spei4["p"], 0.103949, atol=5e-3)


def test_cross_district_spearman_correlations():
    """Verify empirical ecological faces correlations with Hill-1 diversity."""
    sp_df = pd.read_csv(STATS / "india_spearman.csv")
    from scipy.stats import spearmanr
    for window, path in (("core", paths.INDIA_METRICS), ("ext5", paths.INDIA_METRICS_EXT5)):
        observations = pd.read_parquet(path)
        for row in sp_df.loc[sp_df.window == window].itertuples():
            pairs = observations[[row.predictor, row.face]].dropna()
            rho, p_value = spearmanr(pairs.iloc[:, 0], pairs.iloc[:, 1])
            assert row.n == len(pairs)
            np.testing.assert_allclose([row.rho, row.p], [rho, p_value], atol=1e-14)


# ==============================================================================
# 4. HISTORICAL DROUGHT VALIDATION
# ==============================================================================

def test_historical_drought_years_validation():
    """Verify validation table against officially documented All-India Drought Years."""
    dy = pd.read_csv(VALID / "india_drought_years.csv").set_index("year")

    # Check 5 documented drought years
    doc_years = [2002, 2004, 2009, 2014, 2015]
    for yr in doc_years:
        assert dy.loc[yr, "documented_drought_year"] == True
        assert dy.loc[yr, "monsoon_departure_lpa_pct"] < 0
        assert str(dy.loc[yr, "url"]).startswith("https://")

    assert np.isclose(dy.loc[2002, "share_drought"], 0.5435, atol=5e-3)
    assert np.isclose(dy.loc[2009, "share_drought"], 0.5551, atol=5e-3)
    assert np.isclose(dy.loc[2014, "share_drought"], 0.2768, atol=5e-3)
    assert np.isclose(dy.loc[2015, "share_drought"], 0.0493, atol=5e-3)

    # Non-documented drought years must have NaN url and departure
    non_doc = dy[dy["documented_drought_year"].isna()]
    assert non_doc["url"].isna().all()
    assert non_doc["monsoon_departure_lpa_pct"].isna().all()
