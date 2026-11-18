"""Check the published case selection and the values shared by its charts and explorer table."""
import pandas as pd

from src.common import paths


def test_published_case_slopes_match_observations():
    published = pd.read_csv(paths.OUT_REAL / "validation/case_resp_beta.csv", float_precision="round_trip")
    cases = {"IND", "COL", "FIN"}
    assert set(published.iso3) == cases

    observations = pd.read_parquet(paths.OUT_REAL / "crop_anomalies.parquet")
    observations = observations[observations.iso3.isin(cases)]
    keys = ["iso3", "item"]
    expected = observations.groupby(keys).agg(
        value_share=("value_share", "first"), resp_beta=("resp_beta", "first"), n_years=("yield_anom", "count")
    )
    actual = published.set_index(keys).sort_index()
    pd.testing.assert_frame_equal(actual[expected.columns], expected, check_exact=True)
    for iso in sorted(cases):
        shown = actual.loc[iso]
        assert set(shown.index[shown.in_figure]) == set(shown.value_share.nlargest(6).index)
