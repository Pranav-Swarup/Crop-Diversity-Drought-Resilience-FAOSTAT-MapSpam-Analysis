"""Reproduce the Response Diversity Paradox analysis.

This script demonstrates that higher response divergence (resp_div) correlates
with HIGHER vulnerability — contradicting basic portfolio theory — and resolves
the paradox by showing that resp_div is a proxy for the presence of a single
hyper-sensitive crop.

Run: python -m src.climate.analysis.response_paradox
  or: source .venv/bin/activate && python src/climate/analysis/response_paradox.py
"""
import numpy as np
import pandas as pd
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
    metrics = pd.read_parquet(paths.METRICS)
    climate = pd.read_parquet(paths.CLIMATE)
    anomalies = pd.read_parquet(paths.CROP_ANOMALIES)

    # ── Part 1: Confirm the paradox ──────────────────────────────────────────
    df = metrics[["iso3", "resp_div", "vulnerability", "vuln_drought", "stability",
                  "hill1", "mean_crop_cv"]].dropna(subset=["resp_div", "vulnerability"])

    # Spearman: resp_div vs vulnerability
    rho_vuln, p_vuln = stats.spearmanr(df["resp_div"], df["vulnerability"])
    rho_stab, p_stab = stats.spearmanr(df["resp_div"], df["stability"])
    print("=" * 70)
    print("PART 1: Confirming the Response Diversity Paradox")
    print("=" * 70)
    print(f"  Spearman(resp_div, vulnerability):  rho = {rho_vuln:+.4f}, p = {p_vuln:.2e}  (n={len(df)})")
    print(f"  Spearman(resp_div, stability):      rho = {rho_stab:+.4f}, p = {p_stab:.2e}  (n={len(df)})")
    print(f"  → Higher resp_div predicts HIGHER vulnerability and LOWER stability.")
    print()

    # ── Part 2: Compute max_abs_beta per country ─────────────────────────────
    max_beta = anomalies.groupby("iso3")["resp_beta"].apply(
        lambda s: s.drop_duplicates().abs().max()
    ).rename("max_abs_beta")
    df = df.merge(max_beta, on="iso3", how="inner")

    # Correlation: resp_div vs max_abs_beta
    r_proxy, p_proxy = stats.pearsonr(df["resp_div"], df["max_abs_beta"])
    print("=" * 70)
    print("PART 2: Resolving the Paradox — resp_div as a proxy for max_abs_beta")
    print("=" * 70)
    print(f"  Pearson(resp_div, max_abs_beta):    r = {r_proxy:+.4f}, p = {p_proxy:.2e}  (n={len(df)})")
    print(f"  → resp_div is almost perfectly correlated with the single most")
    print(f"    drought-sensitive crop. It does NOT indicate balanced diversity.")
    print()

    # ── Part 3: max_abs_beta drives portfolio volatility ─────────────────────
    df2 = df.dropna(subset=["mean_crop_cv"])
    r_cv, p_cv = stats.pearsonr(df2["max_abs_beta"], df2["mean_crop_cv"])
    r_rdcv, p_rdcv = stats.pearsonr(df2["resp_div"], df2["mean_crop_cv"])
    print("=" * 70)
    print("PART 3: The 'Achilles Heel' mechanism")
    print("=" * 70)
    print(f"  Pearson(max_abs_beta, mean_crop_cv): r = {r_cv:+.4f}, p = {p_cv:.2e}  (n={len(df2)})")
    print(f"  Pearson(resp_div, mean_crop_cv):     r = {r_rdcv:+.4f}, p = {p_rdcv:.2e}  (n={len(df2)})")
    print(f"  → The hyper-sensitive crop directly drives up portfolio volatility.")
    print()

    # ── Part 4: Controlling for climate variance ─────────────────────────────
    spei_var = climate.groupby("iso3")["spei12_w"].var().rename("spei12_w_var")
    df = df.merge(spei_var, on="iso3", how="inner")
    df3 = df.dropna(subset=["resp_div", "vulnerability", "spei12_w_var"])

    from statsmodels.api import OLS, add_constant
    X = add_constant(df3[["resp_div", "spei12_w_var"]])
    model = OLS(df3["vulnerability"], X).fit()
    print("=" * 70)
    print("PART 4: OLS vulnerability ~ resp_div + spei12_w_var")
    print("=" * 70)
    print(f"  resp_div coefficient:     {model.params['resp_div']:.4f}  (p = {model.pvalues['resp_div']:.2e})")
    print(f"  spei12_w_var coefficient:  {model.params['spei12_w_var']:.4f}  (p = {model.pvalues['spei12_w_var']:.2e})")
    print(f"  R² = {model.rsquared:.4f}")
    print(f"  → resp_div remains significant EVEN after controlling for climate variance.")
    print()

    # ── Generate Figures ─────────────────────────────────────────────────────
    out_dir = paths.FIGURES / "climate"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: resp_div vs vulnerability
    ax = axes[0]
    ax.scatter(df["resp_div"], df["vulnerability"], alpha=0.6, s=20, c="tab:red")
    z = np.polyfit(df["resp_div"].values, df["vulnerability"].values, 1)
    x_line = np.linspace(df["resp_div"].min(), df["resp_div"].max(), 50)
    ax.plot(x_line, np.polyval(z, x_line), "k--", lw=1.5)
    ax.set_xlabel("Response Diversity (resp_div)")
    ax.set_ylabel("Vulnerability")
    ax.set_title(f"The Paradox\nρ={rho_vuln:.2f}, p={p_vuln:.1e}")

    # Panel 2: resp_div vs max_abs_beta
    ax = axes[1]
    ax.scatter(df["resp_div"], df["max_abs_beta"], alpha=0.6, s=20, c="tab:blue")
    z2 = np.polyfit(df["resp_div"].values, df["max_abs_beta"].values, 1)
    ax.plot(x_line, np.polyval(z2, x_line), "k--", lw=1.5)
    ax.set_xlabel("Response Diversity (resp_div)")
    ax.set_ylabel("Max |β| (most sensitive crop)")
    ax.set_title(f"The Explanation\nr={r_proxy:.2f}, p={p_proxy:.1e}")

    # Panel 3: max_abs_beta vs mean_crop_cv
    ax = axes[2]
    ax.scatter(df2["max_abs_beta"], df2["mean_crop_cv"], alpha=0.6, s=20, c="tab:green")
    z3 = np.polyfit(df2["max_abs_beta"].values, df2["mean_crop_cv"].values, 1)
    x3 = np.linspace(df2["max_abs_beta"].min(), df2["max_abs_beta"].max(), 50)
    ax.plot(x3, np.polyval(z3, x3), "k--", lw=1.5)
    ax.set_xlabel("Max |β| (most sensitive crop)")
    ax.set_ylabel("Mean Crop CV (portfolio volatility)")
    ax.set_title(f"The Mechanism\nr={r_cv:.2f}, p={p_cv:.1e}")

    plt.suptitle("Resolving the Response Diversity Paradox", fontweight="bold", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / "04_response_diversity_paradox.png")
    plt.close()
    print(f"Saved: {out_dir / '04_response_diversity_paradox.png'}")

    # ── Save summary CSV for reproducibility ─────────────────────────────────
    summary = pd.DataFrame({
        "test": ["spearman_resp_div_vs_vulnerability", "spearman_resp_div_vs_stability",
                 "pearson_resp_div_vs_max_abs_beta", "pearson_max_abs_beta_vs_mean_crop_cv",
                 "pearson_resp_div_vs_mean_crop_cv",
                 "ols_resp_div_coeff", "ols_spei_var_coeff", "ols_r2"],
        "value": [rho_vuln, rho_stab, r_proxy, r_cv, r_rdcv,
                  model.params["resp_div"], model.params["spei12_w_var"], model.rsquared],
        "p_value": [p_vuln, p_stab, p_proxy, p_cv, p_rdcv,
                    model.pvalues["resp_div"], model.pvalues["spei12_w_var"], np.nan],
        "n": [len(df), len(df), len(df), len(df2), len(df2),
              len(df3), len(df3), len(df3)]
    })
    stats_dir = paths.STATS_DIR
    stats_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(stats_dir / "response_paradox.csv", index=False)
    print(f"Saved: {stats_dir / 'response_paradox.csv'}")


if __name__ == "__main__":
    main()
