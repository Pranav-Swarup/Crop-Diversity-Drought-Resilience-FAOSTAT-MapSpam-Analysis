"""Unit and integration tests for src/explorer and Phase 4 India district explorer integration.

Run: python -m pytest tests/test_explorer.py
"""
import json
import re
import pandas as pd
import pytest

from src.common import paths
from src.explorer import build
from src.explorer.india_data import load_india_data


@pytest.fixture(scope="module")
def india_payload():
    return load_india_data()


def test_india_payload_structure_and_counts(india_payload):
    assert india_payload is not None
    assert set(india_payload.keys()) == {"geojson", "meta", "metrics", "ts", "stats", "extra_metrics"}
    assert len(india_payload["geojson"]["features"]) == 690
    assert len(india_payload["metrics"]["core"]) == len(pd.read_parquet(paths.INDIA_METRICS))
    assert len(india_payload["metrics"]["ext5"]) == len(pd.read_parquet(paths.INDIA_METRICS_EXT5))
    assert len(india_payload["ts"]["core"]) == len(pd.read_parquet(paths.INDIA_METRICS))
    assert len(india_payload["ts"]["ext5"]) == len(pd.read_parquet(paths.INDIA_METRICS_EXT5))


def test_india_metrics_match_parquet(india_payload):
    m_core = pd.read_parquet(paths.INDIA_METRICS).set_index("dist_code")
    m_core.index = m_core.index.astype(str)
    for code, row in m_core.iterrows():
        got = india_payload["metrics"]["core"][code]
        for col in ("hill1", "stability", "vulnerability", "phi_sync"):
            assert got[col] == float(row[col])
        assert got["n_events"] == int(row["n_events"])

    m_ext5 = pd.read_parquet(paths.INDIA_METRICS_EXT5).set_index("dist_code")
    m_ext5.index = m_ext5.index.astype(str)
    for code, row in m_ext5.iterrows():
        got = india_payload["metrics"]["ext5"][code]
        for col in ("hill1", "stability", "vulnerability", "phi_sync"):
            assert got[col] == float(row[col])


def test_india_timeseries_alignment_and_strict_json(india_payload):
    # Strict JSON serialization (no NaN or Infinity allowed)
    dumped = json.dumps(india_payload, allow_nan=False)
    assert len(dumped) > 100_000

    for win, expected_len in (("core", 25), ("ext5", 23)):
        for code, ts in india_payload["ts"][win].items():
            assert len(ts["years"]) == expected_len, (win, code)
            assert len(ts["spei"]) == expected_len
            assert len(ts["spei_kh"]) == expected_len
            assert len(ts["anom"]) == expected_len
            assert len(ts["hill1_tv"]) == expected_len
            # First 5 years of 5-year lagged hill1_tv must be None, rest finite numbers
            assert all(v is None for v in ts["hill1_tv"][:5])
            assert all(isinstance(v, (int, float)) and v >= 1.0 for v in ts["hill1_tv"][5:])
            for crop, vals in ts["crops"].items():
                assert len(vals) == expected_len, (win, code, crop)


def test_india_stats_tables_embedded(india_payload):
    stats = india_payload["stats"]
    assert len(stats["summary"]) == len(pd.read_csv(paths.STATS_DIR / "india_summary.csv")) == 2
    assert len(stats["panel"]) == len(pd.read_csv(paths.STATS_DIR / "india_panel.csv")) == 104
    assert len(stats["severity"]) == len(pd.read_csv(paths.STATS_DIR / "india_severity.csv")) == 28
    assert len(stats["vs_national"]) == len(pd.read_csv(paths.STATS_DIR / "india_vs_national.csv")) == 30
    assert len(stats["conley"]) == len(pd.read_csv(paths.STATS_DIR / "india_conley.csv")) == 30
    assert len(stats["kharif"]) == len(pd.read_csv(paths.STATS_DIR / "india_kharif.csv")) == 36
    assert len(stats["kharif_crops"]) == len(pd.read_csv(paths.STATS_DIR / "india_kharif_crops.csv")) == 54
    assert len(stats["leave_out_state"]) == len(pd.read_csv(paths.STATS_DIR / "india_leave_out_state.csv")) == 80
    doc_years = [r["year"] for r in stats["drought_years"] if r["documented_drought_year"]]
    assert set(doc_years) == {2002, 2004, 2009, 2014, 2015}


def test_built_explorer_html_contains_phase4_elements():
    html_path = paths.EXPLORER / "index.html"
    assert html_path.exists(), "explorer/index.html does not exist"
    size_mb = html_path.stat().st_size / (1024 * 1024)
    assert size_mb < build.MAX_SIZE_MB

    content = html_path.read_text(encoding="utf-8")
    for token in (
        'id="scope-select"',
        'value="india_core"',
        'value="india_ext5"',
        'id="india-state-select"',
        'id="india-dist-select"',
        'id="tab-india"',
        'id="india-view"',
        "function renderIndiaMap()",
        "function renderDistrictCharts(",
        "function renderIndiaResultsTab()",
    ):
        assert token in content, f"Missing {token} in explorer/index.html"

    # Extract embedded indiaData JSON and verify district counts
    m = re.search(r"const indiaData = (\{.*?\});\n", content)
    assert m is not None, "indiaData assignment not found in explorer/index.html"
    parsed = json.loads(m.group(1))
    assert len(parsed["metrics"]["core"]) == len(pd.read_parquet(paths.INDIA_METRICS))
    assert len(parsed["metrics"]["ext5"]) == len(pd.read_parquet(paths.INDIA_METRICS_EXT5))


def test_embedded_result_tables_preserve_source_cells():
    html = (paths.EXPLORER / "index.html").read_text()
    tables = json.loads(re.search(r"^\s*const nationalTables = (.*);$", html, re.M)[1])
    sources = [*paths.STATS_DIR.glob("*.csv"), *paths.VALIDATION_DIR.glob("*.csv"),
               paths.OUT / "exposure_diversity.parquet", paths.OUT / "supply_diversity.parquet"]
    assert set(tables) == {source.stem for source in sources}
    for source in sources:
        frame = (pd.read_csv(source, float_precision="round_trip") if source.suffix == ".csv"
                 else pd.read_parquet(source))
        assert tables[source.stem]["columns"] == frame.columns.tolist()
        assert tables[source.stem]["rows"] == frame.astype(object).where(frame.notna(), None).values.tolist()


def test_national_chart_values_preserve_source_gaps():
    html = (paths.EXPLORER / "index.html").read_text()
    series = json.loads(re.search(r"^\s*const tsData = (.*);$", html, re.M)[1])
    anomalies = pd.read_parquet(paths.CROP_ANOMALIES).set_index(["iso3", "item", "year"]).sort_index()
    for iso, country in series.items():
        for crop, shown in country["crops"].items():
            actual = anomalies.loc[(iso, crop), "yield_anom"].reindex(country["years"])
            assert shown == actual.astype(object).where(actual.notna(), None).tolist(), (iso, crop)


def test_embedded_district_geometry_is_valid_and_clockwise(india_payload):
    from shapely.geometry import shape

    for feature in india_payload["geojson"]["features"]:
        geom = shape(feature["geometry"])
        assert geom.is_valid and not geom.is_empty
        polygons = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
        for polygon in polygons:
            assert not polygon.exterior.is_ccw
            assert all(ring.is_ccw for ring in polygon.interiors)
