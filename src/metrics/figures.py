"""Result figures -> figures/results/ (PNG, 300 dpi, 8x5 in).

  01_faces_2x2.png           hill1 vs each of the four faces
  02_count_vs_response.png   standardised OLS betas of hill1 vs resp_div per face
  03_decomposition.png       log(stability) split into crop-level and synchrony parts
  04_maps.png                world maps of hill1 and resp_div

Run: python -m src.metrics.figures   (after src.metrics.run and src.metrics.stats)
"""
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker  # noqa: F401
import numpy as np
import pandas as pd

from ..common import paths
from .data import FACES, analysis_table, source_line

FIGSIZE, DPI = (8, 5), 300
INK, MUTED, GRID = "#222222", "#666666", "#e3e3e3"
# face -> (panel title, y-axis label)
FACE_LABELS = {
    "stability": ("Stability", "1 / CV of national value (log scale)"),
    "resistance": ("Resistance", "$I_t$ / mean $I$ of 3 prior years"),
    "recovery": ("Recovery", "mean $I$ of 3 later years / $I_t$"),
    "vulnerability": ("Vulnerability", "share of years with $I_t$ < 0.9"),
}
N_DECOMP = 15

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})


def save(fig, name, n):
    fig.text(0.01, 0.01, source_line(n), fontsize=6, color=MUTED, ha="left", va="bottom")
    out = paths.fig_dir("results") / name
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out.relative_to(paths.ROOT)}")


def region_colors(df):
    regions = sorted(df["region"].unique())
    assert len(regions) <= 10, "more regions than tab10 colours"
    return dict(zip(regions, plt.get_cmap("tab10").colors))


def fig_faces(df, spearman):
    colors = region_colors(df)
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE, sharex=True)
    for ax, face in zip(axes.ravel(), FACES):
        d = df.dropna(subset=["hill1", face])
        for region, g in d.groupby("region"):
            ax.scatter(g["hill1"], g[face], s=16, color=colors[region], edgecolor="white", linewidth=0.4, label=region)
        s = spearman[(spearman["face"] == face) & (spearman["predictor"] == "hill1")].iloc[0]
        text = f"Spearman ρ = {s.rho:+.2f} [{s.ci_lo:+.2f}, {s.ci_hi:+.2f}], n = {s.n}" if np.isfinite(s.rho) else f"no data (n = {s.n})"
        title, ylabel = FACE_LABELS[face]
        ax.set_title(f"{title}: {text}", fontsize=7.5, color=INK, loc="left")
        ax.set_ylabel(ylabel, fontsize=7)
        if face == "stability" and len(d):
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
            ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.set_yticks([t for t in (2, 5, 10, 20, 50, 100, 200) if d[face].min() / 1.5 <= t <= d[face].max() * 1.5])
    for ax in axes[1]:
        ax.set_xlabel("Crop diversity, hill1 (effective number of crops)")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, markersize=4, label=r) for r, c in colors.items()]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, 0.035))
    fig.suptitle("Crop diversity and the four faces of resilience, by country", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.11, 1, 0.97))
    save(fig, "01_faces_2x2.png", len(df))


def fig_coefficients(df, ols):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    c = plt.get_cmap("tab10").colors
    styles = {"hill1": (c[0], "o", "Crop count diversity (hill1)", -0.12), "resp_div": (c[1], "s", "Response diversity (resp_div)", 0.12)}
    for term, (color, marker, label, dy) in styles.items():
        d = ols[ols["term"] == term].set_index("face").reindex(FACES)
        y = np.arange(len(FACES)) + dy
        ax.errorbar(d["beta"], y, xerr=[d["beta"] - d["ci_lo"], d["ci_hi"] - d["beta"]], fmt=marker, color=color,
                    markersize=6, elinewidth=2, capsize=0, label=label)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    n = ols.set_index("face")["n"].groupby(level=0).first().reindex(FACES)
    ax.set_yticks(range(len(FACES)), [f"{f.capitalize()}\n(n = {int(n[f])})" for f in FACES])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Standardised coefficient (SD of face per SD of predictor), 95% CI, HC3 standard errors")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2)
    ax.set_title("Crop count vs response diversity as predictors of each face\n"
                 "OLS with log GDP per capita, fertiliser use and irrigation share as controls",
                 fontsize=10, color=INK, loc="left")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "02_count_vs_response.png", len(df))


def fig_decomposition(df):
    d = df.dropna(subset=["hill1", "phi_sync", "mean_crop_cv"]).sort_values("hill1").reset_index(drop=True)
    pick = d.iloc[np.unique(np.linspace(0, len(d) - 1, min(N_DECOMP, len(d))).round().astype(int))]
    crop_part = np.log(1 / pick["mean_crop_cv"])
    sync_part = -0.5 * np.log(pick["phi_sync"])
    c = plt.get_cmap("tab10").colors
    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = np.arange(len(pick))
    ax.bar(x, crop_part, width=0.7, color=c[0], edgecolor="white", linewidth=1, label="Stability of individual crops: log(1 / mean_crop_cv)")
    ax.bar(x, sync_part, width=0.7, bottom=crop_part, color=c[1], edgecolor="white", linewidth=1, label="Asynchrony between crops: −0.5 · log(phi_sync)")
    ax.set_xticks(x, [f"{i}\n{h:.1f}" for i, h in zip(pick["iso3"], pick["hill1"])], fontsize=7)
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("Country (ISO3) and its hill1 (effective number of crops), ordered by hill1")
    ax.set_ylabel("Contribution to log(stability) (log units)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    ax.set_title("Where national stability comes from: crop-level stability vs asynchrony\n"
                 "Bar height = log(1 / CV) of the summed value of crops reported in all 31 years",
                 fontsize=10, color=INK, loc="left")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "03_decomposition.png", len(df))


def fig_maps(df):
    world = gpd.read_file(paths.COUNTRIES_GEOJSON)
    world = world[world["iso3"] != "ATA"].merge(df[["iso3", "hill1", "resp_div"]], on="iso3", how="left")
    fig, axes = plt.subplots(2, 1, figsize=FIGSIZE)
    panels = [("hill1", "Crop diversity (hill1)", "Effective number\nof crops"),
              ("resp_div", "Response diversity (resp_div)", "Value-weighted SD of\nyield response per unit\nSPEI-12 (scale capped at\nthe 95th percentile)")]
    for ax, (col, name, label) in zip(axes, panels):
        world.plot(ax=ax, color="#eeeeee", edgecolor="white", linewidth=0.15)
        if world[col].notna().any():
            # Cap the colour scale so one outlier does not flatten the rest.
            vmax = world[col].quantile(0.95) if col == "resp_div" else world[col].max()
            world.dropna(subset=[col]).plot(
                ax=ax, column=col, cmap="viridis", vmin=world[col].min(), vmax=vmax, edgecolor="white", linewidth=0.15,
                legend=True, legend_kwds={"orientation": "vertical", "pad": 0.02, "shrink": 0.85, "label": label,
                                          "extend": "max" if col == "resp_div" else "neither"})
        ax.set_xlim(-180, 180)
        ax.set_ylim(-58, 85)
        ax.set_axis_off()
        ax.set_title(f"{name}, n = {int(world[col].notna().sum())} countries (grey = not in sample)",
                     fontsize=9, color=INK, loc="left")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "04_maps.png", len(df))


def main():
    df = analysis_table()
    spearman = pd.read_csv(paths.STATS_DIR / "spearman.csv")
    ols = pd.read_csv(paths.STATS_DIR / "ols.csv")
    fig_faces(df, spearman)
    fig_coefficients(df, ols)
    fig_decomposition(df)
    fig_maps(df)
    made = sorted(p.name for p in paths.fig_dir("results").glob("0*.png"))
    assert len(made) == 4, made
    print(f"figures: {len(made)} for {len(df)} countries")


if __name__ == "__main__":
    main()
