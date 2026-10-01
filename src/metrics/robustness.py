"""Robustness of the headline Spearman correlations -> outputs/stats/robustness.csv.

Each variant recomputes the metrics (or the sample) and reports rho of every face vs hill1
and vs resp_div, next to the baseline.

Run: python -m src.metrics.robustness
"""
import pandas as pd

from ..common import paths
from .data import analysis_table
from .events import find_events
from .run import compute, load_inputs
from .stats import spearman_table

MAX_IMPUTED_SHARE = 0.10


def main():
    crops, climate, events, included = load_inputs()
    has_climate = len(climate) > 0

    variants = {"baseline": {}}
    variants["lowess_frac_0.3"] = {"frac": 0.3}
    variants["lowess_frac_0.7"] = {"frac": 0.7}
    if has_climate:
        variants["severe_drought_-1.5"] = {"thr": -1.5, "events": find_events(climate.dropna(subset=["spei12_w"]), thr=-1.5)}

    rows = []
    base = None
    for name, kw in variants.items():
        ev = kw.pop("events", events)
        table = analysis_table(compute(crops, climate, ev, included, **kw)["metrics"])
        if name == "baseline":
            base = table
        rows.append(spearman_table(table).assign(variant=name, n_countries=len(table)))

    low_imp = base[base["share_imputed"] <= MAX_IMPUTED_SHARE]
    rows.append(spearman_table(low_imp).assign(variant=f"imputed_value_share<={MAX_IMPUTED_SHARE}", n_countries=len(low_imp)))
    for region in sorted(base["region"].unique()):
        sub = base[base["region"] != region]
        rows.append(spearman_table(sub).assign(variant=f"without {region}", n_countries=len(sub)))

    out = pd.concat(rows, ignore_index=True)[["variant", "n_countries", "face", "predictor", "rho", "ci_lo", "ci_hi", "p", "n"]]
    assert not out.duplicated(["variant", "face", "predictor"]).any()
    assert out["rho"].dropna().between(-1, 1).all()
    paths.STATS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(paths.STATS_DIR / "robustness.csv", index=False)

    wide = out.assign(pair=out["face"] + "~" + out["predictor"]).pivot(index="variant", columns="pair", values="rho")
    wide = wide.loc[out["variant"].drop_duplicates()]
    pd.set_option("display.width", 250)
    print(f"variants: {len(wide)}; rows: {len(out)}")
    print("Spearman rho by variant:")
    print(wide.round(2).to_string())
    b = wide.loc["baseline"]
    flips = (wide.mul(b).lt(0) & wide.notna()).sum()
    print("\nvariants where the sign differs from baseline: " + ", ".join(f"{k}={v}" for k, v in flips.items()))


if __name__ == "__main__":
    main()
