"""Scientific figures for the extension. Every plotted value is exported with units and sample counts."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .storage import FIG, RESULT

COLORS = ["#495057", "#8A817C", "#007C83", "#AD5520", "#7057A3"]
LABEL = {
    "spei12_w": "Presented December, all crops",
    "december12": "December, matched crops",
    "harvest6": "Six months to harvest",
    "season": "Planting to harvest",
}
WINDOW = {
    "presented_1993_2023": "Presented\n1993–2023",
    "common_1993_2023": "Common countries\n1993–2023",
    "common_1961_2023": "Common countries\n1961–2023",
    "eligible_1961_2023": "Eligible countries\n1961–2023",
}


def finish(fig, name, caption):
    fig.text(
        0.075, 0.025, caption + "   •   Analysis by Pmjarvis", size=8, color="#42484B", va="bottom"
    )
    for ax in fig.axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=9)
    fig.savefig(FIG / f"{name}.png", dpi=210, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# The presented model imports plotting code that changes global Matplotlib settings.
# Each comparison uses the same style whether run alone or after fitting a model.
@plt.style.context("default")
def season_figures():
    faces = pd.read_csv(RESULT / "season_faces.csv")
    cy = pd.read_csv(RESULT / "season_national_year.csv")
    effects = pd.read_csv(RESULT / "season_loss_panel.csv").query("term=='dry'").set_index("measure")
    records = []
    for m, label in LABEL.items():
        mask = cy[m] <= -1
        old = cy.spei12_w <= -1
        effect = effects.loc[m]
        records.append(
            dict(
                measure=m,
                label=label,
                n_country_years=len(cy),
                n_countries=cy.iso3.nunique(),
                n_drought=int(mask.sum()),
                n_events=int(faces.loc[faces.measure == m, "n_events"].sum()),
                newly_dry=int((mask & ~old).sum()),
                no_longer_dry=int((~mask & old).sum()),
                output_difference_pct_trend=100 * effect.beta,
                lo_pct=100 * effect.ci_lo,
                hi_pct=100 * effect.ci_hi,
            )
        )
    t = pd.DataFrame(records)
    t.to_csv(RESULT / "figure_01_drought.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1, 1.05]})
    y = np.arange(len(t))
    axes[0].barh(y, t.n_drought, color=COLORS[:4], height=0.56)
    for i, n in enumerate(t.n_drought):
        axes[0].text(n + 8, i, str(n), va="center", size=9)
    axes[0].set(
        yticks=y,
        yticklabels=t.label,
        xlim=(0, t.n_drought.max() * 1.14),
        xlabel="Drought country-years",
        title="Which years qualify?",
    )
    axes[0].invert_yaxis()
    for i, row in t.iterrows():
        a, b, c = row[["output_difference_pct_trend", "lo_pct", "hi_pct"]]
        axes[1].errorbar(a, i, xerr=[[a - b], [c - a]], fmt="o", capsize=4, color=COLORS[i])
    axes[1].axvline(0, color="#858585", lw=1)
    axes[1].set(
        yticks=y,
        yticklabels=[],
        xlabel="Output difference (% of trend)",
        title="How large is the estimated loss?",
    )
    axes[1].invert_yaxis()
    fig.subplots_adjust(left=0.22, right=0.98, bottom=0.24, top=0.84, wspace=0.2)
    finish(
        fig,
        "01_calendar_drought",
        "1993–2023 • SPEI ≤ −1 • country and year fixed effects at mean crop diversity • 95% country-clustered intervals",
    )

    div = pd.read_csv(RESULT / "season_response_diversity.csv").set_index("iso3")
    crop = pd.read_csv(RESULT / "season_crop_slopes.csv")
    rows = []
    for x in crop.itertuples():
        rows.append(
            dict(
                metric="Crop yield response",
                iso3=x.iso3,
                item=x.item,
                n=x.n_years,
                december=x.december12,
                season=x.season,
                unit="fraction of trend per SPEI unit",
            )
        )
    for iso, x in div.iterrows():
        rows.append(
            dict(
                metric="Response diversity",
                iso3=iso,
                item="",
                n=x.n_crops,
                december=x.december12,
                season=x.season,
                unit="weighted SD of crop slopes",
            )
        )
    for col, title in (("resistance", "Resistance"), ("recovery", "Recovery")):
        d = faces.pivot(index="iso3", columns="measure", values=col)[["december12", "season"]].dropna()
        for iso, x in d.iterrows():
            rows.append(
                dict(
                    metric=title,
                    iso3=iso,
                    item="",
                    n=np.nan,
                    december=x.december12,
                    season=x.season,
                    unit="output-index ratio",
                )
            )
    s = pd.DataFrame(rows).dropna(subset=["december", "season"])
    s.to_csv(RESULT / "figure_02_responses.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(9, 7.7))
    units = {
        "Crop yield response": "Yield slope (fraction of trend per SPEI unit)",
        "Response diversity": "Weighted SD of crop slopes",
        "Resistance": "Event / pre-event output index",
        "Recovery": "Post-event / event output index",
    }
    for ax, (name, g) in zip(axes.ravel(), s.groupby("metric", sort=False)):
        ax.scatter(g.december, g.season, s=15, alpha=0.56, color="#007C83", edgecolors="none")
        lo = min(g.december.min(), g.season.min())
        hi = max(g.december.max(), g.season.max())
        ax.plot([lo, hi], [lo, hi], "--", lw=0.9, color="#495057")
        ax.set(
            title=f"{name}  (n = {len(g)})",
            xlabel=units[name] + "\nDecember exposure",
            ylabel=units[name] + "\nGrowing-season exposure",
            xlim=(lo - 0.04 * (hi - lo), hi + 0.04 * (hi - lo)),
            ylim=(lo - 0.04 * (hi - lo), hi + 0.04 * (hi - lo)),
        )
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.13, top=0.94, wspace=0.33, hspace=0.48)
    finish(
        fig,
        "02_calendar_responses",
        "Paired estimates for the same crops or countries • dashed line marks equal estimates • FAOSTAT, GGCMI, MapSPAM, SPEIbase",
    )


@plt.style.context("default")
def history_figure():
    t = pd.read_csv(RESULT / "history_severity.csv").query("term=='extreme_x_hill1'").copy()
    t["label"] = t.model.map(WINDOW).str.replace("\n", " ", regex=False)
    t["effect_pp"] = 100 * t.beta
    t["lo_pp"] = 100 * t.ci_lo
    t["hi_pp"] = 100 * t.ci_hi
    t["mde_pp"] = 100 * t.mde_80
    t.to_csv(RESULT / "figure_04_extremes.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.8))
    y = np.arange(len(t))
    ax[0].barh(y, t.n_extreme, color=COLORS[: len(t)], height=0.52)
    for i, row in enumerate(t.itertuples()):
        ax[0].text(
            row.n_extreme + 1, i, f"{row.n_extreme} in {row.n_exposed} countries", va="center", size=9
        )
        ax[1].errorbar(
            row.effect_pp,
            i,
            xerr=[[row.effect_pp - row.lo_pp], [row.hi_pp - row.effect_pp]],
            fmt="o",
            color=COLORS[i],
            capsize=4,
        )
    ax[0].set(
        yticks=y,
        yticklabels=[WINDOW[x] for x in t.model],
        xlim=(0, 92),
        xlabel="Extreme country-years (SPEI ≤ −2)",
        title="Observed extreme droughts",
    )
    ax[1].axvline(0, color="#777777", lw=1)
    ax[1].set(
        yticks=y,
        yticklabels=[],
        xlabel="Change in output anomaly (percentage points)",
        title="Diversity interaction per 1 SD",
    )
    for a in ax:
        a.invert_yaxis()
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.22, top=0.86, wspace=0.15)
    finish(
        fig,
        "04_extreme_drought_power",
        "Country and year fixed effects • 95% country-clustered intervals • FAOSTAT constant-price value and December SPEI-12",
    )


@plt.style.context("default")
def heat_figure():
    table = pd.read_csv(RESULT / "heat_effects.csv")
    key = ("presented_full_sample", "presented_CRU_Dec12", "ERA5_30_Dec12", "ERA5_30_season")
    names = (
        "Presented CRU and December drought, full sample",
        "CRU and December drought, matched sample",
        "ERA5 heat and December drought, matched sample",
        "ERA5 heat and seasonal drought, matched sample",
    )
    table = pd.concat([pd.read_csv(RESULT / "heat_presented_benchmark.csv"), table], ignore_index=True)
    f = table[table.model.isin(key)].copy()
    f["unit"] = np.where(
        f.outcome.eq("anom"), "percentage points of output trend", "VCI points (0–100)"
    )
    scale = np.where(f.outcome.eq("anom"), 100.0, 1.0)
    for col in ["beta", "ci_lo", "ci_hi"]:
        f[col + "_plot"] = f[col] * scale
    f.to_csv(RESULT / "figure_03_heat.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5))
    cats = ["dry_only", "hot_only", "compound"]
    for ax, out, label, title in zip(
        axes,
        ["anom", "vci"],
        ["Output difference (% of trend)", "Vegetation Condition Index difference (points)"],
        ["Crop output", "Satellite vegetation"],
    ):
        for i, (model, name) in enumerate(zip(key, names)):
            g = f[(f.outcome == out) & (f.model == model)].set_index("term")
            if g.empty:
                continue
            g = g.loc[cats]
            x = g.beta_plot.to_numpy()
            ax.errorbar(
                x,
                np.arange(3) + (i - 1.5) * 0.14,
                xerr=np.vstack([x - g.ci_lo_plot, g.ci_hi_plot - x]),
                fmt="o",
                capsize=3,
                color=COLORS[i],
                label=name,
            )
        ax.axvline(0, color="#888888", lw=0.9)
        ax.set(
            yticks=np.arange(3),
            yticklabels=["Dry only", "Hot only", "Hot and dry"],
            xlabel=label,
            title=title,
        )
        ax.invert_yaxis()
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.115),
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.36, top=0.9, wspace=0.25)
    finish(
        fig,
        "03_heat_and_vegetation",
        "1993–2023 • reference is neither hot nor dry • 95% country-clustered intervals • ERA5, NOAA STAR, CRU, FAOSTAT",
    )


@plt.style.context("default")
def vegetation_figure():
    data = pd.read_csv(RESULT / "figure_03b_vegetation.csv")
    fig, ax = plt.subplots(figsize=(6.7, 5.4))
    ax.scatter(data.vci, data.grid_vci, s=20, color=COLORS[2], alpha=0.48, edgecolors="none")
    low = min(data.vci.min(), data.grid_vci.min()) - 2
    high = max(data.vci.max(), data.grid_vci.max()) + 2
    ax.plot([low, high], [low, high], "--", color=COLORS[0], lw=1, label="Equal readings")
    ax.set(
        xlim=(low, high),
        ylim=(low, high),
        aspect="equal",
        xlabel="Seasonal VCI on NOAA’s country-crop masks (points)",
        ylabel="Seasonal VCI on the heat study’s crop footprint (points)",
        title="How closely do the two vegetation measures agree?",
    )
    ax.legend(loc="upper left", frameon=False)
    fig.subplots_adjust(left=0.16, right=0.96, bottom=0.18, top=0.92)
    finish(
        fig,
        "03b_vegetation_footprint",
        f"2020–2023 • {len(data)} matched country-years • 0–100 VCI scale",
    )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("study", choices=["season", "history", "heat", "vegetation"])
    {
        "season": season_figures,
        "history": history_figure,
        "heat": heat_figure,
        "vegetation": vegetation_figure,
    }[p.parse_args().study]()
