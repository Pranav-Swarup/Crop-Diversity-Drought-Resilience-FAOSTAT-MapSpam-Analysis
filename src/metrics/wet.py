"""Resilience faces against wet extremes (cropland-weighted SPEI-12 >= +1.0), for included countries.

Outputs:
  outputs/metrics_wet.parquet       iso3, resistance_wet, recovery_wet, vuln_wet, n_wet_events
  outputs/stats/wet_spearman.csv    Spearman rho (bootstrap 95% CI) of each wet face vs hill1 and resp_div
  outputs/stats/wet_vs_dry.csv      per-country dry vs wet face pairs: rho, CI, n, medians
  figures/results/07_wet_vs_dry.png hill1 vs resistance (dry events) and vs resistance_wet

Needs outputs/metrics_national.parquet, outputs/metrics.parquet and data/clean/wet_events.csv.

Run: python -m src.metrics.wet        (USE_DUMMY=1 for dummy inputs)
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..common import paths
from ..common.contract import load, summary, validate
from . import faces
from .data import PREDICTORS, analysis_table
from .figures import FIGSIZE, INK, MUTED, region_colors, save
from .stats import spearman_boot

WET_THR = 1.0
WET_FACES = ["resistance_wet", "recovery_wet", "vuln_wet"]
PAIRS = [("resistance", "resistance_wet"), ("recovery", "recovery_wet"), ("vuln_drought", "vuln_wet")]


def compute(national, climate, wet_events):
    """One row per country in `national`."""
    wet_by = wet_events.groupby("iso3")["year"].apply(list).to_dict()
    wet_years = climate[climate["spei12_w"] >= WET_THR].groupby("iso3")["year"].apply(list).to_dict()
    rows = []
    for iso3, g in national.groupby("iso3", sort=True):
        index = g.set_index("year")["index"]
        ev = wet_by.get(iso3, [])
        rows.append({"iso3": iso3,
                     "resistance_wet": faces.resistance(index, ev),
                     "recovery_wet": faces.recovery(index, ev),
                     "vuln_wet": faces.vulnerability(index, wet_years.get(iso3, [])),
                     "n_wet_events": faces.n_usable_events(index, ev)})
    out = pd.DataFrame(rows)
    out["n_wet_events"] = out["n_wet_events"].astype(int)
    return out


def spearman_wet_table(df):
    rows = [{"face": f, "predictor": p, **spearman_boot(df[p], df[f])} for f in WET_FACES for p in PREDICTORS]
    return pd.DataFrame(rows)[["face", "predictor", "rho", "ci_lo", "ci_hi", "p", "n"]]


def wet_vs_dry_table(df):
    rows = []
    for dry, wet in PAIRS:
        s = spearman_boot(df[dry], df[wet])
        ok = df[[dry, wet]].dropna()
        rows.append({"dry": dry, "wet": wet, "rho": s["rho"], "ci_lo": s["ci_lo"], "ci_hi": s["ci_hi"], "n": s["n"],
                     "median_dry": ok[dry].median(), "median_wet": ok[wet].median()})
    return pd.DataFrame(rows)


def fig_wet_vs_dry(df, spearman):
    """spearman: rows (face, predictor, rho, ci_lo, ci_hi, n) covering hill1 vs resistance and resistance_wet."""
    colors = region_colors(df)
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, sharey=True)
    panels = [("resistance", "Dry events (SPEI-12 ≤ −1)", "Resistance to drought: $I_t$ / mean $I$ of 3 prior years"),
              ("resistance_wet", "Wet events (SPEI-12 ≥ +1)", "Resistance to wet events: $I_t$ / mean $I$ of 3 prior years")]
    for ax, (face, title, ylabel) in zip(axes, panels):
        d = df.dropna(subset=["hill1", face])
        for region, g in d.groupby("region"):
            ax.scatter(g["hill1"], g[face], s=16, color=colors[region], edgecolor="white", linewidth=0.4)
        s = spearman[(spearman["face"] == face) & (spearman["predictor"] == "hill1")].iloc[0]
        text = f"ρ = {s.rho:+.2f} [{s.ci_lo:+.2f}, {s.ci_hi:+.2f}], n = {s.n}" if np.isfinite(s.rho) else f"no data (n = {s.n})"
        ax.set_title(f"{title}\nSpearman {text}", fontsize=7.5, color=INK, loc="left")
        ax.set_xlabel("Crop diversity, hill1 (effective number of crops)")
        ax.set_ylabel(ylabel, fontsize=7)
        ax.axhline(1.0, color=MUTED, linewidth=0.6)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, markersize=4, label=r) for r, c in colors.items()]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, 0.035))
    fig.suptitle("Crop diversity and resistance to dry and wet extremes, by country", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.11, 1, 0.95))
    save(fig, "07_wet_vs_dry.png", len(df))


def main():
    national = pd.read_parquet(paths.NATIONAL)
    climate, wet_events = load("climate"), load("wet_events")
    included = load("quality").query("included")["iso3"]
    national = national[national["iso3"].isin(included)]

    mw = compute(national, climate, wet_events)
    validate(mw, "metrics_wet")
    assert set(mw["iso3"]) <= set(included)
    assert mw["n_wet_events"].sum() <= len(wet_events)

    df = analysis_table().merge(mw, on="iso3", how="inner")
    sp, wd = spearman_wet_table(df), wet_vs_dry_table(df)
    assert not sp.duplicated(["face", "predictor"]).any() and len(sp) == len(WET_FACES) * len(PREDICTORS)
    for t in (sp, wd):
        assert t["rho"].dropna().between(-1, 1).all()
        assert (t["ci_lo"].dropna() <= t["rho"].dropna()).all() and (t["rho"].dropna() <= t["ci_hi"].dropna()).all()

    paths.OUT.mkdir(parents=True, exist_ok=True)
    paths.STATS_DIR.mkdir(parents=True, exist_ok=True)
    mw.to_parquet(paths.METRICS_WET, index=False)
    sp.to_csv(paths.STATS_DIR / "wet_spearman.csv", index=False)
    wd.to_csv(paths.STATS_DIR / "wet_vs_dry.csv", index=False)

    print(summary(mw, "metrics_wet"))
    print(f"countries with >=1 usable wet event: {(mw['n_wet_events'] > 0).sum()}; wet events used: {int(mw['n_wet_events'].sum())}")
    print("medians: " + ", ".join(f"{c}={mw[c].median():.3f}" for c in WET_FACES))
    print("\nSpearman rho [bootstrap 95% CI]:")
    for r in sp.itertuples():
        print(f"  {r.face:<15} vs {r.predictor:<8} rho={r.rho:+.2f} [{r.ci_lo:+.2f}, {r.ci_hi:+.2f}]  p={r.p:.3f}  n={r.n}")
    print("\nDry vs wet, across countries:")
    for r in wd.itertuples():
        print(f"  {r.dry:<13} vs {r.wet:<15} rho={r.rho:+.2f} [{r.ci_lo:+.2f}, {r.ci_hi:+.2f}]  n={r.n}  "
              f"median dry={r.median_dry:.3f} wet={r.median_wet:.3f}")

    dry = pd.DataFrame([{"face": "resistance", "predictor": "hill1", **spearman_boot(df["hill1"], df["resistance"])}])
    fig_wet_vs_dry(df, pd.concat([sp, dry], ignore_index=True))


if __name__ == "__main__":
    main()
