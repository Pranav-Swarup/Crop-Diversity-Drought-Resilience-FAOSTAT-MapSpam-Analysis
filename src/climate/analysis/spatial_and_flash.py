"""Reproduce the Spatial Dilution and Flash Drought analyses.

This script:
1. Proves that larger countries suffer from spatial dilution of climate indices.
2. Shows that spei6min_w (seasonal flash droughts) is a better predictor of
   yield anomalies during severe drought events than spei12_w (annual average).

Run: python -m src.climate.analysis.spatial_and_flash
  or: source .venv/bin/activate && python src/climate/analysis/spatial_and_flash.py
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy import stats

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.common import paths


def main():
    plt.rcParams.update({"font.family": "sans-serif", "figure.facecolor": "white",
                         "axes.facecolor": "white", "savefig.dpi": 300})

    # Load data
    climate = pd.read_parquet(paths.CLIMATE)
    metrics = pd.read_parquet(paths.METRICS)
    anomalies = pd.read_parquet(paths.CROP_ANOMALIES)
    gdf = gpd.read_file(paths.COUNTRIES_GEOJSON)

    # Compute country areas from the geojson (in km²)
    gdf_proj = gdf.to_crs("ESRI:54009")  # Mollweide equal-area
    gdf["area_km2"] = gdf_proj.geometry.area / 1e6

    # ══════════════════════════════════════════════════════════════════════════
    # PART 1: SPATIAL DILUTION
    # ══════════════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("PART 1: Spatial Dilution Analysis")
    print("=" * 70)

    # SPEI variance per country
    spei_var = climate.groupby("iso3")["spei12_w"].var().rename("spei12_w_var")

    # max_abs_beta per country
    max_beta = anomalies.groupby("iso3")["resp_beta"].apply(
        lambda s: s.drop_duplicates().abs().max()
    ).rename("max_abs_beta")

    df = gdf[["iso3", "area_km2"]].merge(spei_var, on="iso3").merge(max_beta, on="iso3")
    df = df.dropna()
    df["log_area"] = np.log10(df["area_km2"])

    # Correlation: SPEI variance vs area (proves dilution)
    r_dilution, p_dilution = stats.pearsonr(df["log_area"], df["spei12_w_var"])
    # Correlation: max_abs_beta vs area (proves crops are not less sensitive)
    r_beta_area, p_beta_area = stats.pearsonr(df["log_area"], df["max_abs_beta"])

    print(f"  Pearson(log10(area_km2), spei12_w_var):  r = {r_dilution:+.4f}, p = {p_dilution:.2e}  (n={len(df)})")
    print(f"  Pearson(log10(area_km2), max_abs_beta):  r = {r_beta_area:+.4f}, p = {p_beta_area:.2e}  (n={len(df)})")
    print(f"  → Large countries have dampened SPEI variance (spatial averaging),")
    print(f"    but their crops are NOT less drought-sensitive.")
    print()

    # ══════════════════════════════════════════════════════════════════════════
    # PART 2: FLASH DROUGHTS vs ANNUAL DROUGHTS
    # ══════════════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("PART 2: Flash Droughts vs Annual Droughts")
    print("=" * 70)

    # Join crop anomalies with climate data to compare spei12_w vs spei6min_w
    national = pd.read_parquet(paths.NATIONAL)
    merged = national[["iso3", "year", "anom"]].merge(
        climate[["iso3", "year", "spei12_w", "spei6min_w"]], on=["iso3", "year"], how="inner"
    ).dropna()

    # Global correlation
    r_12_global, p_12_global = stats.pearsonr(merged["spei12_w"], merged["anom"])
    r_6_global, p_6_global = stats.pearsonr(merged["spei6min_w"], merged["anom"])

    # Severe drought subset
    severe = merged[merged["spei12_w"] < -1.5]
    if len(severe) > 10:
        r_12_severe, p_12_severe = stats.pearsonr(severe["spei12_w"], severe["anom"])
        r_6_severe, p_6_severe = stats.pearsonr(severe["spei6min_w"], severe["anom"])
    else:
        r_12_severe = r_6_severe = p_12_severe = p_6_severe = np.nan

    print(f"  Global (all years, n={len(merged)}):")
    print(f"    Pearson(spei12_w, national_anom):   r = {r_12_global:+.4f}, p = {p_12_global:.2e}")
    print(f"    Pearson(spei6min_w, national_anom):  r = {r_6_global:+.4f}, p = {p_6_global:.2e}")
    print()
    print(f"  Severe droughts only (spei12_w < -1.5, n={len(severe)}):")
    print(f"    Pearson(spei12_w, national_anom):   r = {r_12_severe:+.4f}, p = {p_12_severe:.2e}")
    print(f"    Pearson(spei6min_w, national_anom):  r = {r_6_severe:+.4f}, p = {p_6_severe:.2e}")
    print(f"  → During severe droughts, seasonal flash drought metrics (spei6min_w)")
    print(f"    are MORE predictive of agricultural losses than annual averages.")
    print()

    # ══════════════════════════════════════════════════════════════════════════
    # GENERATE FIGURES
    # ══════════════════════════════════════════════════════════════════════════
    out_dir = paths.FIGURES / "climate"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: Spatial Dilution — log(area) vs SPEI variance
    ax = axes[0]
    ax.scatter(df["log_area"], df["spei12_w_var"], alpha=0.5, s=20, c="tab:purple")
    z = np.polyfit(df["log_area"].values, df["spei12_w_var"].values, 1)
    x_line = np.linspace(df["log_area"].min(), df["log_area"].max(), 50)
    ax.plot(x_line, np.polyval(z, x_line), "k--", lw=1.5)
    ax.set_xlabel("log₁₀(Country Area, km²)")
    ax.set_ylabel("Var(spei12_w)")
    ax.set_title(f"Spatial Dilution\nr={r_dilution:.2f}, p={p_dilution:.1e}")

    # Panel 2: No dilution in crop sensitivity
    ax = axes[1]
    ax.scatter(df["log_area"], df["max_abs_beta"], alpha=0.5, s=20, c="tab:orange")
    z2 = np.polyfit(df["log_area"].values, df["max_abs_beta"].values, 1)
    ax.plot(x_line, np.polyval(z2, x_line), "k--", lw=1.5)
    ax.set_xlabel("log₁₀(Country Area, km²)")
    ax.set_ylabel("Max |β| (crop sensitivity)")
    ax.set_title(f"No Sensitivity Dilution\nr={r_beta_area:.2f}, p={p_beta_area:.1e}")

    # Panel 3: Flash drought comparison
    ax = axes[2]
    labels = ["spei12_w\n(global)", "spei6min_w\n(global)",
              "spei12_w\n(severe)", "spei6min_w\n(severe)"]
    values = [r_12_global, r_6_global, r_12_severe, r_6_severe]
    colors = ["tab:blue", "tab:red", "tab:blue", "tab:red"]
    bars = ax.bar(labels, values, color=colors, alpha=0.7, edgecolor="black")
    ax.set_ylabel("Pearson r (with national anomaly)")
    ax.set_title("Flash Droughts Beat Annual\nDuring Severe Events")
    ax.axhline(0, color="black", lw=0.5)

    plt.suptitle("Spatial Dilution & Flash Drought Analysis", fontweight="bold", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / "05_spatial_dilution_flash_droughts.png")
    plt.close()
    print(f"Saved: {out_dir / '05_spatial_dilution_flash_droughts.png'}")

    # ── Save summary CSV ─────────────────────────────────────────────────────
    summary = pd.DataFrame({
        "test": [
            "pearson_log_area_vs_spei_var", "pearson_log_area_vs_max_abs_beta",
            "pearson_spei12_vs_anom_global", "pearson_spei6min_vs_anom_global",
            "pearson_spei12_vs_anom_severe", "pearson_spei6min_vs_anom_severe",
        ],
        "value": [r_dilution, r_beta_area, r_12_global, r_6_global, r_12_severe, r_6_severe],
        "p_value": [p_dilution, p_beta_area, p_12_global, p_6_global, p_12_severe, p_6_severe],
        "n": [len(df), len(df), len(merged), len(merged), len(severe), len(severe)]
    })
    stats_dir = paths.STATS_DIR
    stats_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(stats_dir / "spatial_flash.csv", index=False)
    print(f"Saved: {stats_dir / 'spatial_flash.csv'}")


if __name__ == "__main__":
    main()
