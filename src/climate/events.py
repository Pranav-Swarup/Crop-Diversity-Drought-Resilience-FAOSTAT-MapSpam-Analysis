"""Drought events from climate.parquet -> data/clean/events.csv (rule in src/metrics/events.py).

First-iteration pipeline written from src/metrics' side; Prathmesh owns this module.

Run: python -m src.climate.events
"""
from ..common import paths
from ..common.contract import load, summary, validate
from ..metrics.events import DROUGHT_THR, find_events


def main():
    climate = load("climate")
    events = find_events(climate.dropna(subset=["spei12_w"]))
    validate(events, "events")
    events.to_csv(paths.EVENTS, index=False)
    print(summary(events, "events"))
    print(f"drought years (spei12_w <= {DROUGHT_THR}): {(climate['spei12_w'] <= DROUGHT_THR).sum()}; "
          f"events: {len(events)} in {events['iso3'].nunique()} countries; {events['severity'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
