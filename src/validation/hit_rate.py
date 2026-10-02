"""Does our drought catalogue catch well-known droughts?  -> outputs/validation/drought_hits.csv

Reads src/validation/known_droughts.csv (iso3, year, event_name, source_citation; filled in and
checked by hand) and, for every known event, reports

  hit_spei12_w       spei12_w     <= -1.0  (cropland-weighted SPEI-12 at December; our drought-year rule)
  in_event_catalogue the (iso3, year) is a row in events.csv (first year of a run, 3 drought-free years before)
  hit_spei6min_w     spei6min_w   <= -1.5  (worst month of cropland-weighted SPEI-6)
  hit_spei12_unw     spei12_unw   <= -1.0  (same rule, area-weighted instead of cropland-weighted)
  *_pm1              the same test, passed if any of year-1, year, year+1 qualifies (supplementary)

Also writes drought_hits_summary.csv (hit rate per test next to its base rate over ALL country-years)
and figures/validation/01_hit_rate.png.

Uses climate.parquet directly (not the metrics tables), so countries dropped from the analysis
sample (e.g. by the inclusion rules) can still be checked.

Run: python -m src.validation.hit_rate        (USE_DUMMY=1 for dummy inputs)
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..common import paths
from ..common.contract import load
from ..metrics.events import DROUGHT_THR

KNOWN = Path(__file__).with_name("known_droughts.csv")
SPEI6_THR = -1.5
FIGSIZE, DPI = (8, 5), 300
INK, MUTED, GRID = "#222222", "#666666", "#e3e3e3"

# (label, column in the per-event table, how the base rate is computed)
TESTS = [
    ("SPEI-12 <= -1.0, cropland-weighted", "hit_spei12_w", "spei12_w"),
    ("SPEI-12 <= -1.0, unweighted", "hit_spei12_unw", "spei12_unw"),
    ("SPEI-6 min <= -1.5, cropland-weighted", "hit_spei6min_w", "spei6min_w"),
    ("In event catalogue (events.csv)", "in_event_catalogue", "catalogue"),
    ("SPEI-12 <= -1.0 within +-1 year, weighted", "hit_spei12_w_pm1", None),
    ("SPEI-12 <= -1.0 within +-1 year, unweighted", "hit_spei12_unw_pm1", None),
]


def read_known():
    known = pd.read_csv(KNOWN, dtype={"iso3": str, "event_name": str, "source_citation": str})
    assert list(known.columns) == ["iso3", "year", "event_name", "source_citation"], known.columns.tolist()
    assert known["iso3"].str.fullmatch(r"[A-Z]{3}").all(), "known_droughts.csv: iso3 must be 3 upper-case letters"
    assert not known.duplicated(["iso3", "year"]).any(), "known_droughts.csv: duplicate (iso3, year)"
    return known


def indicator(values, thr):
    """True/False where the value exists, <NA> where it does not."""
    return values.le(thr).where(values.notna()).astype("boolean")


def within_one_year(clim, iso3, year, col, thr):
    v = clim.loc[(clim["iso3"] == iso3) & clim["year"].between(year - 1, year + 1), col].dropna()
    return pd.NA if v.empty else bool((v <= thr).any())


def hits_table(known, climate, events):
    cols = ["iso3", "year", "spei12_w", "spei12_unw", "spei6min_w"]
    t = known.merge(climate[cols], on=["iso3", "year"], how="left")
    t["in_climate"] = t["spei12_w"].notna() | t["spei12_unw"].notna()
    t["hit_spei12_w"] = indicator(t["spei12_w"], DROUGHT_THR)
    t["hit_spei12_unw"] = indicator(t["spei12_unw"], DROUGHT_THR)
    t["hit_spei6min_w"] = indicator(t["spei6min_w"], SPEI6_THR)
    cat = set(zip(events["iso3"], events["year"]))
    t["in_event_catalogue"] = pd.Series(
        [(i, y) in cat for i, y in zip(t["iso3"], t["year"])], index=t.index).where(t["spei12_w"].notna()).astype("boolean")
    for col, src in [("hit_spei12_w_pm1", "spei12_w"), ("hit_spei12_unw_pm1", "spei12_unw")]:
        t[col] = pd.Series([within_one_year(climate, i, y, src, DROUGHT_THR) for i, y in zip(t["iso3"], t["year"])],
                           index=t.index, dtype="boolean")
    return t


def base_rate(climate, events, source):
    """Share of all country-years in climate.parquet that the test would flag."""
    if source is None:
        return np.nan
    if source == "catalogue":
        n = climate["spei12_w"].notna().sum()
        return len(events) / n if n else np.nan
    thr = SPEI6_THR if source == "spei6min_w" else DROUGHT_THR
    v = climate[source].dropna()
    return float((v <= thr).mean()) if len(v) else np.nan


def summary_table(t, climate, events):
    rows = []
    for label, col, source in TESTS:
        ok = t[col].dropna()
        rows.append({"test": label, "n_events_testable": len(ok), "n_hits": int(ok.sum()),
                     "hit_rate": float(ok.mean()) if len(ok) else np.nan,
                     "base_rate_all_country_years": base_rate(climate, events, source)})
    return pd.DataFrame(rows)


def fig_hit_rate(t, s):
    c = plt.get_cmap("tab10").colors
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE, gridspec_kw={"width_ratios": [1, 1.7]})

    # (a) hit rate, weighted vs unweighted, next to the share of all country-years flagged
    s2 = s.iloc[:2]
    x = np.arange(2)
    ax1.bar(x, s2["hit_rate"], width=0.6, color=[c[0], c[1]])
    ax1.hlines(s2["base_rate_all_country_years"], x - 0.3, x + 0.3, color=INK, linestyle="--", linewidth=1,
               label="Share of all country-years flagged")
    for xi, r in zip(x, s2.itertuples()):
        ax1.text(xi, r.hit_rate + 0.02, f"{r.n_hits}/{r.n_events_testable}", ha="center", fontsize=8, color=INK)
    ax1.set_xticks(x, ["Cropland-\nweighted", "Unweighted"])
    ax1.set_ylim(0, 1.1)
    ax1.set_ylabel("Share of known events with SPEI-12 <= -1.0 (fraction)")
    ax1.grid(axis="x", visible=False)
    ax1.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), fontsize=7)
    ax1.set_title("(a) Hit rate", fontsize=9, color=INK, loc="left")

    # (b) SPEI-12 at each known event, weighted vs unweighted
    labels = [f"{r.iso3} {r.year}" for r in t.itertuples()]
    y = np.arange(len(t))[::-1]
    ax2.axvline(DROUGHT_THR, color=MUTED, linestyle="--", linewidth=0.9)
    ax2.scatter(t["spei12_w"], y + 0.12, s=28, color=c[0], label="Cropland-weighted", zorder=3)
    ax2.scatter(t["spei12_unw"], y - 0.12, s=28, color=c[1], marker="s", label="Unweighted", zorder=3)
    for yi, r in zip(y, t.itertuples()):
        if not r.in_climate:
            ax2.text(0.02, yi, "no climate data", fontsize=6.5, color=MUTED, va="center", transform=ax2.get_yaxis_transform())
    ax2.set_yticks(y, labels, fontsize=7)
    ax2.set_xlabel("SPEI-12 at December (standard deviations; drought if <= -1.0)")
    ax2.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=7)
    ax2.set_title("(b) SPEI-12 in each known drought year", fontsize=9, color=INK, loc="left")
    fig.suptitle("Does our drought index flag well-known droughts?", fontsize=10, color=INK, x=0.01, ha="left")

    src = ("SYNTHETIC DUMMY DATA, not results. " if paths.USE_DUMMY else "") + \
        "Source: SPEIbase v2.10, MapSPAM 2020; known events from src/validation/known_droughts.csv."
    fig.text(0.01, 0.01, src, fontsize=6, color=MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    out = paths.fig_dir("validation") / "01_hit_rate.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out.relative_to(paths.ROOT)}")


def main():
    known, climate, events = read_known(), load("climate"), load("events")
    blank = int(known["source_citation"].isna().sum())
    if blank:
        print(f"NOTE: {blank} of {len(known)} known events have no source_citation yet; check them before this goes on a slide.")
    t = hits_table(known, climate, events)
    s = summary_table(t, climate, events)

    out = paths.VALIDATION_DIR
    out.mkdir(parents=True, exist_ok=True)
    keep = ["iso3", "year", "event_name", "source_citation", "in_climate", "spei12_w", "spei12_unw", "spei6min_w"] + [c for _, c, _ in TESTS]
    t = t[keep]
    assert not t.duplicated(["iso3", "year"]).any() and len(t) == len(known)
    assert s["hit_rate"].dropna().between(0, 1).all()
    t.to_csv(out / "drought_hits.csv", index=False)
    s.to_csv(out / "drought_hits_summary.csv", index=False)

    missing = t.loc[~t["in_climate"], "iso3"].unique().tolist()
    if missing:
        print(f"WARNING: no climate data for {missing}; those events are not counted in any hit rate.")
    pd.set_option("display.width", 220)
    print("\nPer event:")
    print(t.drop(columns=["source_citation"]).round(2).to_string(index=False))
    print("\nSummary:")
    print(s.round(3).to_string(index=False))
    fig_hit_rate(t, s)


if __name__ == "__main__":
    main()