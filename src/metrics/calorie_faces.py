"""Resilience faces on national crop calorie production, for included countries.

The national series is the sum over crops of kcal (data/clean/calories.parquet, src/crops/calories.py) per
year, 1993-2023, built and detrended exactly as the value index in src.metrics.run.country_metrics:
LOWESS (detrend.FRAC) on the yearly sum reindexed to 1993-2023, I_t = y_t / trend_t; the faces are
src.metrics.faces.* with the drought events of data/clean/events.csv and drought years spei12_w <= -1.0
from data/clean/climate.parquet. test_calories.py asserts that national_faces() gives the same numbers as
country_metrics() for the same series.

kcal_value_coverage is the share of the country's window-total crop value (value_const) held by crop items
that have a calorie factor (src.crops.calories.value_coverage). Faces are computed only where
kcal_value_coverage >= MIN_COVERAGE (0.5: below that the calorie series describes less than half of the
country's crop production by value); other countries keep a row with NaN faces and n_events_kcal = 0.

Outputs:
  outputs/metrics_kcal.parquet          iso3, stability_kcal, resistance_kcal, recovery_kcal, vulnerability_kcal,
                                        vuln_drought_kcal, n_events_kcal, kcal_value_coverage
  outputs/stats/kcal_spearman.csv       Spearman rho (bootstrap 95% CI) of each kcal face vs hill1 and resp_div
  outputs/stats/kcal_vs_value.csv       per face: rho (CI) between the value-based and kcal-based face, medians
  figures/results/09_kcal_vs_value.png  hill1 vs stability (value) and vs stability_kcal

Needs data/clean/calories.parquet (no dummy version), crops.parquet, quality.csv, climate.parquet, events.csv
and outputs/metrics.parquet.

Run: python -m src.metrics.calorie_faces
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker  # noqa: F401
import numpy as np
import pandas as pd

from ..common import paths
from ..common.contract import load
from ..crops.calories import value_coverage
from . import faces
from .data import PREDICTORS, analysis_table
from .detrend import FRAC, detrend
from .events import DROUGHT_THR
from .figures import FIGSIZE, INK, MUTED, region_colors, save
from .run import YEARS
from .stats import spearman_boot

MIN_COVERAGE = 0.5
FACES = ["stability", "resistance", "recovery", "vulnerability", "vuln_drought"]
KCAL_FACES = [f"{f}_kcal" for f in FACES]
COLUMNS = ["iso3"] + KCAL_FACES + ["n_events_kcal", "kcal_value_coverage"]
SPEARMAN_CSV = paths.STATS_DIR / "kcal_spearman.csv"
VS_VALUE_CSV = paths.STATS_DIR / "kcal_vs_value.csv"


def national_faces(yearly, spei, event_years, frac=FRAC, thr=DROUGHT_THR):
    """Faces of one national series. yearly: Series year -> national total; spei: Series year -> spei12_w."""
    spei = spei.reindex(YEARS)
    nat = detrend(yearly.reindex(YEARS).to_numpy(), YEARS, frac)
    index = pd.Series(nat["index"].to_numpy(), index=YEARS)
    drought_years = spei.index[spei <= thr]
    return {
        "stability": faces.stability(nat["y"], nat["resid"]),
        "resistance": faces.resistance(index, event_years),
        "recovery": faces.recovery(index, event_years),
        "vulnerability": faces.vulnerability(index),
        "vuln_drought": faces.vulnerability(index, drought_years),
        "n_events": faces.n_usable_events(index, event_years),
    }


def compute(calories, coverage, climate, events, included, min_coverage=MIN_COVERAGE):
    """One row per included country with a calorie series."""
    spei_by = {k: v.set_index("year")["spei12_w"] for k, v in climate.groupby("iso3")}
    events_by = events.groupby("iso3")["year"].apply(list).to_dict()
    rows = []
    for iso3, g in calories[calories["iso3"].isin(included)].groupby("iso3", sort=True):
        cov = float(coverage.get(iso3, np.nan))
        if np.isfinite(cov) and cov >= min_coverage:
            m = national_faces(g.groupby("year")["kcal"].sum(), spei_by.get(iso3, pd.Series(dtype=float)),
                               events_by.get(iso3, []))
        else:
            m = {f: np.nan for f in FACES} | {"n_events": 0}
        rows.append({"iso3": iso3, **{f"{k}_kcal": v for k, v in m.items()}, "kcal_value_coverage": cov})
    out = pd.DataFrame(rows, columns=COLUMNS)
    return out.astype({"n_events_kcal": int})


def spearman_table(df):
    rows = [{"face": f, "predictor": p, **spearman_boot(df[p], df[f])} for f in KCAL_FACES for p in PREDICTORS]
    return pd.DataFrame(rows)[["face", "predictor", "rho", "ci_lo", "ci_hi", "p", "n"]]


def vs_value_table(df):
    rows = []
    for f in FACES:
        s = spearman_boot(df[f], df[f"{f}_kcal"])
        ok = df[[f, f"{f}_kcal"]].dropna()
        rows.append({"face": f, "face_kcal": f"{f}_kcal", "rho": s["rho"], "ci_lo": s["ci_lo"], "ci_hi": s["ci_hi"],
                     "p": s["p"], "n": s["n"], "median_value": ok[f].median(), "median_kcal": ok[f"{f}_kcal"].median()})
    return pd.DataFrame(rows)


def fig_kcal_vs_value(df):
    colors = region_colors(df)
    d = df.dropna(subset=["hill1", "stability", "stability_kcal"])
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, sharey=True)
    panels = [("stability", "Crop value (constant 2014–2016 I$)", "Stability: 1 / CV of detrended national series (log scale)"),
              ("stability_kcal", "Crop calories (kcal, FAOSTAT SUA factors)", "")]
    for ax, (face, title, ylabel) in zip(axes, panels):
        for region, g in d.groupby("region"):
            ax.scatter(g["hill1"], g[face], s=16, color=colors[region], edgecolor="white", linewidth=0.4)
        s = spearman_boot(d["hill1"], d[face])
        text = f"ρ = {s['rho']:+.2f} [{s['ci_lo']:+.2f}, {s['ci_hi']:+.2f}], n = {s['n']}" if np.isfinite(s["rho"]) else f"no data (n = {s['n']})"
        ax.set_title(f"{title}\nSpearman {text}", fontsize=7.5, color=INK, loc="left")
        ax.set_xlabel("Crop diversity, hill1 (effective number of crops)")
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=7)
    if len(d):
        lo, hi = d[["stability", "stability_kcal"]].min().min(), d[["stability", "stability_kcal"]].max().max()
        axes[0].set_yscale("log")
        axes[0].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        axes[0].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        axes[0].set_yticks([t for t in (2, 5, 10, 20, 50, 100, 200) if lo / 1.5 <= t <= hi * 1.5])
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, markersize=4, label=r) for r, c in colors.items()]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, 0.05))
    fig.suptitle("Crop diversity and stability of national crop value and crop calories, by country",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.035, "Calorie factors: FAOSTAT Supply Utilization Accounts 2010–2023, one factor per crop item; "
             "same countries in both panels.", fontsize=6, color=MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.13, 1, 0.95))
    save(fig, "09_kcal_vs_value.png", len(d))


def check(mk, included):
    assert list(mk.columns) == COLUMNS, f"metrics_kcal: columns {list(mk.columns)}"
    assert not mk["iso3"].duplicated().any() and mk["iso3"].str.fullmatch(r"[A-Z]{3}").all()
    assert set(mk["iso3"]) <= set(included)
    for c in KCAL_FACES + ["kcal_value_coverage"]:
        assert pd.api.types.is_float_dtype(mk[c]), f"metrics_kcal.{c}: not float"
    assert pd.api.types.is_integer_dtype(mk["n_events_kcal"]) and (mk["n_events_kcal"] >= 0).all()
    for c, (lo, hi) in {"stability_kcal": (0, np.inf), "resistance_kcal": (0, np.inf), "recovery_kcal": (0, np.inf),
                        "vulnerability_kcal": (0, 1), "vuln_drought_kcal": (0, 1), "kcal_value_coverage": (0, 1)}.items():
        v = mk[c].dropna()
        assert v.between(lo, hi).all(), f"metrics_kcal.{c}: outside [{lo}, {hi}]"
    low = mk["kcal_value_coverage"] < MIN_COVERAGE
    assert mk.loc[low, KCAL_FACES].isna().all().all() and (mk.loc[low, "n_events_kcal"] == 0).all()


def main():
    needed = [paths.CALORIES, paths.CROPS, paths.QUALITY, paths.CLIMATE, paths.EVENTS, paths.METRICS]
    missing = [str(p.relative_to(paths.ROOT)) for p in needed if not p.exists()]
    if missing:
        hint = " calories.parquet has no dummy version; run without USE_DUMMY=1." if paths.USE_DUMMY else \
            " Run python -m src.crops.calories first." if not paths.CALORIES.exists() else ""
        raise SystemExit(f"src.metrics.calorie_faces: missing input {missing}.{hint}")

    calories = pd.read_parquet(paths.CALORIES)
    crops, quality, climate, events = load("crops"), load("quality"), load("climate"), load("events")
    included = quality.loc[quality["included"], "iso3"].tolist()
    in_scope = set(calories["item_code"])
    coverage = value_coverage(crops, in_scope)

    mk = compute(calories, coverage, climate, events, included)
    check(mk, included)
    no_kcal = sorted(set(included) - set(mk["iso3"]))
    assert not no_kcal, f"included countries without any calorie row: {no_kcal}"

    df = analysis_table(load("metrics")).merge(mk, on="iso3", how="inner")
    sp, vv = spearman_table(df), vs_value_table(df)
    assert not sp.duplicated(["face", "predictor"]).any() and len(sp) == len(KCAL_FACES) * len(PREDICTORS)
    for t in (sp, vv):
        r = t.dropna(subset=["rho"])
        assert r["rho"].between(-1, 1).all() and (r["ci_lo"] <= r["rho"]).all() and (r["rho"] <= r["ci_hi"]).all()

    paths.OUT.mkdir(parents=True, exist_ok=True)
    paths.STATS_DIR.mkdir(parents=True, exist_ok=True)
    mk.to_parquet(paths.METRICS_KCAL, index=False)
    sp.to_csv(SPEARMAN_CSV, index=False)
    vv.to_csv(VS_VALUE_CSV, index=False)

    cov = mk["kcal_value_coverage"]
    print(f"metrics_kcal    rows={len(mk)}  countries={mk['iso3'].nunique()}  "
          f"missing={100 * mk[KCAL_FACES].isna().to_numpy().mean():.1f}% (faces)")
    print(f"kcal_value_coverage: median {cov.median():.3f}, 10th pct {cov.quantile(0.1):.3f}, min {cov.min():.3f} "
          f"({mk.loc[cov.idxmin(), 'iso3']}); below {MIN_COVERAGE}: {int((cov < MIN_COVERAGE).sum())}; "
          f"below 0.9: {sorted(mk.loc[cov < 0.9, 'iso3'])}")
    print(f"usable events: kcal {int(mk['n_events_kcal'].sum())}, value {int(df['n_events'].sum())} (same countries)")
    print("% missing per face: " + ", ".join(f"{c}={100 * mk[c].isna().mean():.0f}" for c in KCAL_FACES))
    print("medians: " + ", ".join(f"{c}={mk[c].median():.3f}" for c in KCAL_FACES))
    print("\nSpearman rho [bootstrap 95% CI], kcal faces:")
    for r in sp.itertuples():
        print(f"  {r.face:<19} vs {r.predictor:<8} rho={r.rho:+.2f} [{r.ci_lo:+.2f}, {r.ci_hi:+.2f}]  p={r.p:.3f}  n={r.n}")
    print("\nValue face vs kcal face, across countries:")
    for r in vv.itertuples():
        print(f"  {r.face:<13} rho={r.rho:+.2f} [{r.ci_lo:+.2f}, {r.ci_hi:+.2f}]  n={r.n}  "
              f"median value={r.median_value:.3f} kcal={r.median_kcal:.3f}")
    fig_kcal_vs_value(df)


if __name__ == "__main__":
    main()
