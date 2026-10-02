"""Generate climate exposure figures -> figures/climate/.

Run: python -m src.climate.figures
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize

from ..common import paths

def setup_plot_style():
    """Apply styling rules from CLAUDE.md."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 300,
        "figure.figsize": (8, 5)
    })

def plot_weighted_vs_unweighted(climate, out_dir):
    """01_weighted_vs_unweighted.png"""
    countries = ["RUS", "AUS", "BRA", "IND", "USA", "CHN"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.flatten()
    
    for i, iso3 in enumerate(countries):
        ax = axes[i]
        df = climate[climate["iso3"] == iso3].sort_values("year")
        
        ax.plot(df["year"], df["spei12_unw"], label="Unweighted", color="tab:orange", linestyle="--", alpha=0.8)
        ax.plot(df["year"], df["spei12_w"], label="Cropland-weighted", color="tab:blue", linewidth=2)
        
        # Shade drought years based on weighted SPEI
        drought_years = df[df["spei12_w"] <= -1.0]["year"]
        for dy in drought_years:
            ax.axvspan(dy - 0.5, dy + 0.5, color="tab:red", alpha=0.2, lw=0)
            
        ax.set_title(iso3)
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
        ax.axhline(-1, color="tab:red", linewidth=0.5, linestyle=":")
        if i % 3 == 0:
            ax.set_ylabel("SPEI-12 (December)")
        if i >= 3:
            ax.set_xlabel("Year")
            
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    fig.text(0.01, 0.01, "Source: SPEIbase v2.10, MapSPAM 2020", fontsize=8, color="gray")
    plt.savefig(out_dir / "01_weighted_vs_unweighted.png")
    plt.close()

def plot_cropland_weights(out_dir):
    """02_cropland_weights.png and 02_cropland_weights_lognorm.png"""
    crop_file = paths.RAW / "derived" / "cropland_05deg.npy"
    if not crop_file.exists():
        print(f"Warning: {crop_file} not found. Skipping 02_cropland_weights.png")
        return
        
    crop = np.load(crop_file)  # (360, 720) in hectares
    
    # 1. Linear Scale (Original)
    fig, ax = plt.subplots(figsize=(8, 5))
    crop_masked = np.ma.masked_where(crop <= 0, crop)
    im = ax.imshow(crop_masked, cmap="viridis", extent=[-180, 180, -90, 90])
    
    ax.set_title("Global Cropland Weights (0.5° grid) - Linear Scale")
    ax.set_xlabel("Longitude (degrees)")
    ax.set_ylabel("Latitude (degrees)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Physical Crop Area (Hectares)")
    
    plt.tight_layout()
    fig.text(0.01, 0.01, "Source: MapSPAM 2020 v2", fontsize=8, color="gray")
    plt.savefig(out_dir / "02_cropland_weights.png")
    plt.close()

    # 2. LogNorm Scale (New)
    fig, ax = plt.subplots(figsize=(8, 5))
    im_log = ax.imshow(crop_masked, cmap="viridis", norm=mpl.colors.LogNorm(vmin=max(1, crop_masked.min()), vmax=crop_masked.max()), extent=[-180, 180, -90, 90])
    
    ax.set_title("Global Cropland Weights (0.5° grid) - Log Scale")
    ax.set_xlabel("Longitude (degrees)")
    ax.set_ylabel("Latitude (degrees)")
    cbar_log = fig.colorbar(im_log, ax=ax, fraction=0.03, pad=0.04)
    cbar_log.set_label("Physical Crop Area (Hectares, Log Scale)")
    
    plt.tight_layout()
    fig.text(0.01, 0.01, "Source: MapSPAM 2020 v2", fontsize=8, color="gray")
    plt.savefig(out_dir / "02_cropland_weights_lognorm.png")
    plt.close()

def plot_event_counts(events, geojson_path, out_dir):
    """03_event_counts_map.png"""
    if not geojson_path.exists():
        print(f"Warning: {geojson_path} not found. Skipping 03_event_counts_map.png")
        return
        
    gdf = gpd.read_file(geojson_path)
    counts = events.groupby("iso3").size().reset_index(name="n_events")
    
    merged = gdf.merge(counts, on="iso3", how="left")
    merged["n_events"] = merged["n_events"].fillna(0)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    merged.plot(
        column="n_events", 
        ax=ax, 
        cmap="viridis", 
        legend=True,
        legend_kwds={'label': "Number of Drought Events (1993-2023)"},
        missing_kwds={'color': 'lightgrey'}
    )
    
    merged.boundary.plot(ax=ax, linewidth=0.2, color="white")
    
    ax.set_title("Drought Event Counts per Country")
    ax.set_axis_off()
    
    plt.tight_layout()
    fig.text(0.01, 0.01, "Source: SPEIbase v2.10, MapSPAM 2020, Natural Earth", fontsize=8, color="gray")
    plt.savefig(out_dir / "03_event_counts_map.png")
    plt.close()

def main():
    setup_plot_style()
    out_dir = paths.FIGURES / "climate"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    climate = pd.read_parquet(paths.CLIMATE)
    events = pd.read_csv(paths.EVENTS)
    
    # 1. Generate Figures
    plot_weighted_vs_unweighted(climate, out_dir)
    plot_cropland_weights(out_dir)
    plot_event_counts(events, paths.COUNTRIES_GEOJSON, out_dir)
    
    # 2. Print Summary Metrics
    print(f"countries covered: {climate['iso3'].nunique()}")
    print(f"total events: {len(events)}")
    
    climate["diff"] = (climate["spei12_w"] - climate["spei12_unw"]).abs()
    top10 = climate.groupby("iso3")["diff"].mean().nlargest(10)
    print("mean |spei12_w - spei12_unw| by country (top 10):")
    print(", ".join([f"{k} {v:.2f}" for k, v in top10.items()]))
    
if __name__ == "__main__":
    main()
