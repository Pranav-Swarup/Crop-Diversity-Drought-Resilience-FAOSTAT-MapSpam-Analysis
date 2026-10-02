"""Crop-level drought panels for India plus one high-hill1 and one low-hill1 contrast country.

  python -m src.validation.case_studies                      # prints contrast candidates, makes IND only
  python -m src.validation.case_studies --high ISO3 --low ISO3   # makes IND + the two chosen countries

Inputs (all from src/metrics/run.py): metrics.parquet, crop_anomalies.parquet, metrics_national.parquet.
Outputs:
  outputs/validation/case_candidates.csv   countries with >= 3 usable drought events and >= 6 eligible crops
  outputs/validation/case_resp_beta.csv    resp_beta of every eligible crop in each case-study country
  figures/validation/02_case_<ISO3>.png    top-6 crops by value share: yield anomaly lines, drought years
                                           shaded, national value index overlaid

Run with USE_DUMMY=1 until the real files exist (needs `USE_DUMMY=1 python -m src.metrics.run` first).
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..common import paths
from ..common.contract import YEAR_MAX, YEAR_MIN, load
from ..metrics.events import DROUGHT_THR

FOCAL = "IND"
TOP_N = 6            # crops per figure
MIN_EVENTS = 3       # usable drought events (metrics.n_events) for a contrast country
N_PER_SIDE = 5       # candidates printed per event count and side
FIGSIZE, DPI = (8, 5), 300
INK, MUTED, SHADE = "#222222", "#666666", "#e6e6e6"


def country_names():
    try:
        c = load("countries")
    except FileNotFoundError:
        return {}
    return dict(zip(c["iso3"], c["name"]))


def load_inputs():
    metrics, anomalies = load("metrics"), load("crop_anomalies")
    national = pd.read_parquet(paths.NATIONAL)
    need = {"iso3", "year", "index", "is_drought", "is_event"}
    assert need <= set(national.columns), f"metrics_national.parquet is missing {need - set(national.columns)}"
    return metrics, anomalies, national


def candidates(metrics, anomalies, national, names):
    per_crop = anomalies.drop_duplicates(["iso3", "item"])
    df = metrics[["iso3", "hill1", "n_events", "resp_div"]].copy()
    df["name"] = df["iso3"].map(names).fillna("")
    df["n_drought_years"] = df["iso3"].map(national.groupby("iso3")["is_drought"].sum()).fillna(0).astype(int)
    df["n_crops_response"] = df["iso3"].map(per_crop.groupby("iso3").size()).fillna(0).astype(int)
    median = metrics["hill1"].median()
    df["hill1_group"] = np.where(df["hill1"] >= median, "high", "low")
    out = df[(df["n_events"] >= MIN_EVENTS) & (df["n_crops_response"] >= TOP_N)]
    return out.sort_values(["n_events", "hill1"], ascending=[True, False]).reset_index(drop=True), median


def print_candidates(cands, median, metrics, national):
    pd.set_option("display.width", 200)
    cols = ["iso3", "name", "hill1", "n_events", "n_drought_years", "n_crops_response", "resp_div"]
    print(f"\nContrast candidates: >= {MIN_EVENTS} usable events and >= {TOP_N} eligible crops "
          f"(sample median hill1 = {median:.2f}). Pick one high and one low with the SAME n_events.")
    for n, g in cands.groupby("n_events"):
        hi, lo = g[g["hill1_group"] == "high"].head(N_PER_SIDE), g[g["hill1_group"] == "low"].tail(N_PER_SIDE)
        if hi.empty or lo.empty:
            continue
        print(f"\n--- n_events = {n}: highest hill1 ---")
        print(hi[cols].round(3).to_string(index=False))
        print(f"--- n_events = {n}: lowest hill1 ---")
        print(lo[cols].round(3).to_string(index=False))
    if FOCAL in set(metrics["iso3"]):
        r = metrics.set_index("iso3").loc[FOCAL]
        rank = int((metrics["hill1"] > r["hill1"]).sum()) + 1
        dy = national.loc[(national["iso3"] == FOCAL) & national["is_drought"], "year"].tolist()
        print(f"\n{FOCAL}: hill1 = {r['hill1']:.2f} (rank {rank} of {len(metrics)}), n_events = {int(r['n_events'])}, "
              f"drought years {dy}")
        if r["n_events"] < MIN_EVENTS:
            print(f"  NOTE: {FOCAL} has fewer than {MIN_EVENTS} usable events, so its event count will not match the contrast countries.")


def region_series(g, col):
    return g.set_index("year")[col].reindex(range(YEAR_MIN, YEAR_MAX + 1))


def plot_case(iso3, metrics, anomalies, national, names):
    nat = national[national["iso3"] == iso3].sort_values("year")
    an = anomalies[anomalies["iso3"] == iso3]
    m = metrics.set_index("iso3").loc[iso3]
    per_crop = an.drop_duplicates("item").sort_values("value_share", ascending=False)
    top = per_crop.head(TOP_N)
    c = plt.get_cmap("tab10").colors

    fig, ax = plt.subplots(figsize=FIGSIZE)
    drought_years = nat.loc[nat["is_drought"], "year"]
    for y in drought_years:
        ax.axvspan(y - 0.5, y + 0.5, color=SHADE, zorder=0, linewidth=0)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    for i, r in enumerate(top.itertuples()):
        s = region_series(an[an["item"] == r.item], "yield_anom")
        ax.plot(s.index, s.to_numpy(), color=c[i], linewidth=1.1, marker="o", markersize=2.5,
                label=f"{r.item} (value share {r.value_share:.0%}, resp_beta {r.resp_beta:+.3f})")
    ax.plot(nat["year"], nat["index"] - 1, color="black", linewidth=2.4, label="National value index, $I_t$ − 1")
    ev = nat.loc[nat["is_event"], "year"]
    if len(ev):
        ax.scatter(ev, np.full(len(ev), 0.97), transform=ax.get_xaxis_transform(), marker="v", s=40, color="black",
                   clip_on=False, zorder=5, label="Drought event (first year of a run)")
    ax.plot([], [], color=SHADE, linewidth=8, label=f"Drought year (SPEI-12 <= {DROUGHT_THR:g})")

    ax.set_xlim(YEAR_MIN - 0.5, YEAR_MAX + 0.5)
    ax.set_xlabel("Year")
    ax.set_ylabel("Relative anomaly (fraction of LOWESS trend)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=6.5)
    label = names.get(iso3)
    label = f"{label} ({iso3})" if label and label != iso3 else iso3
    ax.set_title(f"{label}: yield anomalies of the {len(top)} largest crops by value share\n"
                 f"hill1 = {m['hill1']:.1f} effective crops, {int(m['n_events'])} usable drought events, "
                 f"resp_div = {m['resp_div']:.3f}", fontsize=9, color=INK, loc="left")
    src = ("SYNTHETIC DUMMY DATA, not results. " if paths.USE_DUMMY else "") + \
        "Source: FAOSTAT (QCL, QV), SPEIbase v2.10, MapSPAM 2020. 1993–2023. Crop lines use non-imputed years only."
    fig.text(0.01, 0.01, src, fontsize=6, color=MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = paths.fig_dir("validation") / f"02_case_{iso3}.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out.relative_to(paths.ROOT)}")
    return top["item"].tolist()


def resp_beta_table(iso3, role, anomalies, in_figure):
    an = anomalies[anomalies["iso3"] == iso3]
    t = an.groupby("item").agg(value_share=("value_share", "first"), resp_beta=("resp_beta", "first"),
                               n_years=("yield_anom", "count")).reset_index()
    t.insert(0, "role", role)
    t.insert(0, "iso3", iso3)
    t["in_figure"] = t["item"].isin(in_figure)
    return t.sort_values("value_share", ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--high", help="ISO3 of the high-hill1 contrast country")
    ap.add_argument("--low", help="ISO3 of the low-hill1 contrast country")
    args = ap.parse_args()

    metrics, anomalies, national = load_inputs()
    names = country_names()
    cands, median = candidates(metrics, anomalies, national, names)
    out = paths.VALIDATION_DIR
    out.mkdir(parents=True, exist_ok=True)
    cands.to_csv(out / "case_candidates.csv", index=False)
    print_candidates(cands, median, metrics, national)

    chosen = [(FOCAL, "focal")]
    if args.high:
        chosen.append((args.high.upper(), "high hill1"))
    if args.low:
        chosen.append((args.low.upper(), "low hill1"))
    if not (args.high and args.low):
        print("\nNo contrast pair chosen yet: re-run with --high ISO3 --low ISO3 (figures made for IND only).")
    for iso3, _ in chosen:
        assert iso3 in set(metrics["iso3"]), f"{iso3} is not in metrics.parquet"
        assert iso3 in set(anomalies["iso3"]), f"{iso3} has no crop_anomalies rows"
    if args.high and args.low:
        h, l = metrics.set_index("iso3").loc[[args.high.upper(), args.low.upper()]].itertuples()
        if not h.hill1 > l.hill1:
            print(f"WARNING: --high {args.high} has hill1 {h.hill1:.2f}, not above --low {args.low} ({l.hill1:.2f}).")
        if h.n_events != l.n_events:
            print(f"WARNING: event counts differ ({args.high}: {h.n_events}, {args.low}: {l.n_events}).")
        for r in (h, l):
            if r.n_events < MIN_EVENTS:
                print(f"WARNING: {r.Index} has only {r.n_events} usable events (< {MIN_EVENTS}).")

    tables = []
    for iso3, role in chosen:
        top = plot_case(iso3, metrics, anomalies, national, names)
        tables.append(resp_beta_table(iso3, role, anomalies, top))
    betas = pd.concat(tables, ignore_index=True)
    assert not betas.duplicated(["iso3", "item"]).any()
    betas.to_csv(out / "case_resp_beta.csv", index=False)
    print("\nresp_beta per crop (OLS slope of yield anomaly on SPEI-12; positive = yield falls in drought):")
    print(betas.round(3).to_string(index=False))


if __name__ == "__main__":
    main()