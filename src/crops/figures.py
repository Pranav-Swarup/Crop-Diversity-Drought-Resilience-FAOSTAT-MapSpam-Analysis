"""Data-quality figures -> figures/data/.

01_imputed_share_map.png    share of each country's 1993-2023 crop value that FAOSTAT flags E or I
02_example_flat_series.png  three imputed flat series next to one reported series of the same crop

Run: python -m src.crops.figures
"""
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ..common import paths
from ..common.contract import load
from .quality import FLAT_SERIES

OUT = paths.ROOT / "figures" / "data"
SOURCE = "Source: FAOSTAT QCL and QV (bulk download), flags E and I = estimated or imputed."
FLAT_EXAMPLES = [("VNM", "Chillies and peppers, dry"), ("EGY", "Chillies and peppers, dry"), ("THA", "Other fruits, n.e.c.")]
REPORTED_ITEM = "Chillies and peppers, dry"


def setup():
    plt.rcParams.update({"font.family": "sans-serif", "figure.facecolor": "white", "axes.facecolor": "white",
                         "savefig.facecolor": "white", "font.size": 10})
    OUT.mkdir(parents=True, exist_ok=True)


def imputed_share_map():
    q = load("quality")
    geo = gpd.read_file(paths.CLEAN / "countries.geojson").merge(q[["iso3", "share_imputed"]], on="iso3", how="left")
    fig, ax = plt.subplots(figsize=(8, 5))
    geo.plot(ax=ax, color="#d9d9d9", linewidth=0.2, edgecolor="white")
    geo.dropna(subset=["share_imputed"]).plot(ax=ax, column="share_imputed", cmap="viridis", vmin=0, vmax=1,
                                              linewidth=0.2, edgecolor="white", legend=True,
                                              legend_kwds={"label": "Share of crop value imputed or estimated (fraction, 1993-2023)",
                                                           "orientation": "horizontal", "shrink": 0.6, "pad": 0.22})
    ax.set_xlim(-180, 180)
    ax.set_ylim(-58, 84)
    ax.set_xlabel("Longitude (degrees)")
    ax.set_ylabel("Latitude (degrees)")
    ax.set_title("Imputed share of agricultural production value, by country")
    fig.text(0.01, 0.01, SOURCE + " Grey = no data.", fontsize=7, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "01_imputed_share_map.png", dpi=300)
    plt.close(fig)
    return geo["share_imputed"].notna().sum()


def example_flat_series():
    crops = load("crops")
    flat = pd.read_csv(FLAT_SERIES)
    cols = plt.get_cmap("tab10").colors
    fig, ax = plt.subplots(figsize=(8, 5))

    def draw(iso3, item, colour, label):
        g = crops[(crops["iso3"] == iso3) & crops["item"].str.startswith(item) & (crops["prod_t"] > 0)].sort_values("year")
        idx = g["prod_t"] / g["prod_t"].mean()
        ax.plot(g["year"], idx, color=colour, lw=1.5, label=label)
        imp = g["is_imputed"].to_numpy()
        ax.scatter(g["year"][imp], idx[imp], s=22, facecolors="white", edgecolors=colour, zorder=3)
        ax.scatter(g["year"][~imp], idx[~imp], s=22, color=colour, zorder=3)

    for i, (iso3, item) in enumerate(FLAT_EXAMPLES):
        row = flat[(flat["iso3"] == iso3) & flat["item"].str.startswith(item)]
        assert len(row) == 1, (iso3, item)
        draw(iso3, item, cols[i], f"{iso3}, {row['item'].iloc[0].split(',')[0]} (flat, {100 * row['share_years_imputed'].iloc[0]:.0f}% of years imputed)")

    # reported comparison: same crop, never imputed, full 31 years, not on the flat list, largest value
    c = crops[crops["item"].str.startswith(REPORTED_ITEM) & (crops["prod_t"] > 0)]
    s = c.groupby("iso3").agg(n=("year", "size"), imp=("is_imputed", "mean"), v=("value_const", "mean"))
    s = s[(s["n"] == 31) & (s["imp"] == 0) & ~s.index.isin(flat["iso3"][flat["item"].str.startswith(REPORTED_ITEM)])]
    iso_rep = s["v"].idxmax()
    draw(iso_rep, REPORTED_ITEM, cols[3], f"{iso_rep}, Chillies and peppers (reported, 0% imputed)")

    ax.set_xlabel("Year")
    ax.set_ylabel("Production relative to own 1993-2023 mean (ratio)")
    ax.set_title("Imputed series are smooth or repeat values; a reported series varies")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.text(0.99, 0.02, "Filled marker = reported, hollow = imputed or estimated", transform=ax.transAxes,
            ha="right", fontsize=7, color="#555555")
    fig.text(0.01, 0.01, SOURCE, fontsize=7, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "02_example_flat_series.png", dpi=300)
    plt.close(fig)
    return iso_rep


def main():
    setup()
    n = imputed_share_map()
    iso_rep = example_flat_series()
    for f in ("01_imputed_share_map.png", "02_example_flat_series.png"):
        assert (OUT / f).stat().st_size > 20_000, f
    print(f"figures/data: 2 figures; map colours {n} countries; reported comparison series: {iso_rep}")


if __name__ == "__main__":
    main()
