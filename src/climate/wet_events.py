"""Wet events from climate.parquet -> data/clean/wet_events.csv.

Mirror of the drought event rule (src/metrics/events.py) on the sign-flipped series:
wet year = spei12_w >= +1.0; event year = first year of a run of wet years, with 3 years
before and 3 after inside 1993-2023 and the 3 years before wet-free.
Severity: moderate >= 1.0, severe >= 1.5, extreme >= 2.0.

Output: data/clean/wet_events.csv (iso3, year, spei12_w, severity)

Run: python -m src.climate.wet_events
"""
from ..common import paths
from ..common.contract import load, summary, validate
from ..metrics.events import DROUGHT_THR, PAD, find_events

WET_THR = -DROUGHT_THR


def wet_events(climate):
    """Wet events of a DataFrame with iso3, year, spei12_w, via find_events on the negated series."""
    flipped = climate.dropna(subset=["spei12_w"]).copy()
    flipped["spei12_w"] = -flipped["spei12_w"]
    ev = find_events(flipped, thr=-WET_THR)
    ev["spei12_w"] = -ev["spei12_w"]
    return ev


def main():
    climate = load("climate")
    ev = wet_events(climate)
    validate(ev, "wet_events")

    spei = climate.set_index(["iso3", "year"])["spei12_w"]
    for r in ev.itertuples():
        assert r.spei12_w >= WET_THR
        prior = spei.reindex([(r.iso3, y) for y in range(r.year - PAD, r.year)])
        assert prior.notna().all() and (prior < WET_THR).all(), f"{r.iso3} {r.year}: wet year in prior {PAD} years"
    assert (ev.loc[ev["severity"] == "extreme", "spei12_w"] >= 2.0).all()
    assert (ev.loc[ev["severity"] == "moderate", "spei12_w"] < 1.5).all()

    paths.WET_EVENTS.parent.mkdir(parents=True, exist_ok=True)
    ev.to_csv(paths.WET_EVENTS, index=False)
    print(summary(ev, "wet_events"))
    print(f"wet years (spei12_w >= {WET_THR}): {(climate['spei12_w'] >= WET_THR).sum()}; "
          f"events: {len(ev)} in {ev['iso3'].nunique()} countries; {ev['severity'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
