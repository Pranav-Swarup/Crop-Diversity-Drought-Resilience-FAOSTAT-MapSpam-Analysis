"""Unit and integration tests for src/explorer and Phase 4 India district explorer integration.

Run: python -m pytest tests/test_explorer.py
"""
import json
import re
import numpy as np
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
    assert len(india_payload["metrics"]["core"]) == 303
    assert len(india_payload["metrics"]["ext5"]) == 127
    assert len(india_payload["ts"]["core"]) == 303
    assert len(india_payload["ts"]["ext5"]) == 127


def test_india_metrics_match_parquet(india_payload):
    m_core = pd.read_parquet(paths.INDIA_METRICS).set_index("dist_code")
    m_core.index = m_core.index.astype(str)
    for code, row in m_core.iterrows():
        got = india_payload["metrics"]["core"][code]
        for col in ("hill1", "stability", "vulnerability", "phi_sync"):
            assert got[col] == pytest.approx(float(row[col]), abs=1e-4)
        assert got["n_events"] == int(row["n_events"])

    m_ext5 = pd.read_parquet(paths.INDIA_METRICS_EXT5).set_index("dist_code")
    m_ext5.index = m_ext5.index.astype(str)
    for code, row in m_ext5.iterrows():
        got = india_payload["metrics"]["ext5"][code]
        for col in ("hill1", "stability", "vulnerability", "phi_sync"):
            assert got[col] == pytest.approx(float(row[col]), abs=1e-4)


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
    assert set(doc_years) == {2002, 2009, 2014, 2015}


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
    assert len(parsed["metrics"]["core"]) == 303
    assert len(parsed["metrics"]["ext5"]) == 127
