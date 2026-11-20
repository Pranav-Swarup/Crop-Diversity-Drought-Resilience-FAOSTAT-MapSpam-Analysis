"""Reconcile stored analytical figures from real input and measurement tables."""

import shutil

from future_work import validation


def test_crop_calendar_figures():
    validation.seasonal()


def test_extreme_drought_benchmark():
    validation.historical()


def test_comparison_plot_is_identical_after_a_different_plot_style(tmp_path, monkeypatch):
    from future_work import figures

    shutil.copy2(figures.RESULT / "history_severity.csv", tmp_path)
    monkeypatch.setattr(figures, "RESULT", tmp_path)
    monkeypatch.setattr(figures, "FIG", tmp_path)
    figures.history_figure()
    path = tmp_path / "04_extreme_drought_power.png"
    original = path.read_bytes()
    with figures.plt.rc_context({"font.size": 15, "axes.grid": True, "axes.facecolor": "grey"}):
        figures.history_figure()
    assert original == path.read_bytes()


def test_satellite_footprints():
    validation.vegetation()


def test_heat_and_vegetation_effects():
    validation.temperature()


def test_district_status_and_comparison():
    validation.district()
