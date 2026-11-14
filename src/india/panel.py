"""Phase 4 analysis: does a more diverse crop mix reduce the district loss from drought? (docs/phase4-design.md)

Core (DES, 1998-2022, `crops.parquet`) and the 5-crop extended window (mirror, 1993-2015, `crops_ext5.parquet`) share all code.
Estimator and rules come from src.metrics (fe_fit, drought_bins, spearman_boot, find_events, detrend, faces).

Core model, district d, year t (hill1_tv = Hill-1 of area over t-5..t-1, z = standardised over the estimation sample):
    anom_dt = b*spei_dt + g*spei_dt*z_dt + d*z_dt + alpha_d + lambda_{state x year} + e,   clustered by district.
The drought variable is spei12_w (cropland weighted) when data/raw/derived/cropland_05deg.npy was available to src.india.climate,
otherwise spei12_unw (area weighted). The label is written to every output (`spei_variable`).

Outputs: outputs/india/{metrics,panel,crop_anomalies}.parquet, outputs/stats/india_*.csv, figures/india/*.png.
Run: python -m src.india.panel [--perm 499]
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st

from ..common import paths
from ..metrics.panel import coef_rows, fe_fit
from ..metrics.severity import drought_bins
from ..metrics.stats import spearman_boot
from .clean import YEARS as YEARS_CORE
from .metrics import run

IN = paths.CLEAN_REAL / "india"
OUT = paths.OUT_REAL / "india"
STATS = paths.STATS_DIR
FIG = paths.FIGURES / "india"
SEED = 20231231
FE = ("dist_code", "sy")
FACES = ["stability", "resistance", "recovery", "vulnerability", "vuln_drought"]


# ------------------------------------------------------------------ data

def spei_variable(clim):
    return "spei12_w" if clim.spei12_w.notna().any() else "spei12_unw"


def build(window):
    """window: 'core' (DES 1997-2022) or 'ext5' (mirror 1993-2015). Returns dict of frames and settings."""
    clim = pd.read_parquet(IN / "climate.parquet")
    col = spei_variable(clim)
    if window == "core":
        crops, years = pd.read_parquet(IN / "crops.parquet"), list(YEARS_CORE)
        dist = pd.read_csv(IN / "districts.csv", dtype={"dist_code": str})
    else:
        crops, years = pd.read_parquet(IN / "crops_ext5.parquet"), list(range(1993, 2016))
        dist = pd.read_csv(IN / "districts_ext5.csv", dtype={"dist_code": str})
        names = pd.read_csv(IN / "districts.csv", dtype={"dist_code": str})[["dist_code", "state_name"]]
        dist = dist.drop(columns=[c for c in ("state_name",) if c in dist]).merge(names, on="dist_code", how="left")
    matched = set(pd.read_csv(IN / "district_match.csv", dtype={"dist_code": str}).query("matched").dist_code)
    dist = dist[dist.dist_code.isin(matched) & dist.dist_code.isin(clim.dist_code)]
    out = {}
    for sample, flag in (("core", dist.core), ("stable", dist.stable)):
        keep = set(dist.dist_code[flag])
        m, p, ca, dropped = run(crops[crops.dist_code.isin(keep)], clim, years, col)
        p = p.merge(dist[["dist_code", "state_name"]], on="dist_code")
        out[sample] = {"metrics": m, "panel": p, "crop_anom": ca, "dropped": dropped, "n_selected": len(keep)}
    return out, col, years, dist


def add_controls(p):
    k = pd.read_parquet(IN / "mirror_controls.parquet")
    k = k.assign(year=k.year + 1).rename(columns={"irrig_share": "irrig_lag", "fert_kg_ha": "fert_lag"})   # value at t-1, used at t
    return p.merge(k, on=["dist_code", "year"], how="left")


def prep(p):
    """Estimation sample: rows with anom, spei and hill1_tv. Adds sy, z terms and interactions."""
    d = p.dropna(subset=["anom", "spei", "hill1_tv"]).copy()
    d["sy"] = d.state_name + "_" + d.year.astype(str)
    for h, z in (("hill1_tv", "z"), ("hill1_tv3", "z3")):
        d[z] = (d[h] - d[h].mean()) / d[h].std()
    d["spei_z"], d["spei_z3"] = d.spei * d.z, d.spei * d.z3
    d["dry"], d["wet"] = np.minimum(d.spei, 0), np.maximum(d.spei, 0)
    d["dry_z"], d["wet_z"] = d.dry * d.z, d.wet * d.z
    return d


def fit_rows(model, d, terms, cluster="dist_code", note=""):
    f = fe_fit(d, "anom", terms, cluster=cluster, fe=FE)
    rows = coef_rows(f)
    for r in rows:
        r.update(model=model, n_obs=f["nobs"], n_districts=d.dist_code.nunique(), r2_within=f["r2_within"], note=note)
    return rows, f


# ------------------------------------------------------------------ severity + permutation

def severity_frame(d):
    b = drought_bins(d.spei)
    d = d.copy()
    for k in ("moderate", "severe", "extreme"):
        d[k] = b[k].to_numpy()
        d[k + "_z"] = d[k] * d.z
    return d


def fe_basis(d):
    D = np.column_stack([pd.get_dummies(d[FE[0]], dtype=float).to_numpy(), pd.get_dummies(d[FE[1]], drop_first=True, dtype=float).to_numpy()])
    q, _ = np.linalg.qr(D)
    return q


def fast_t(q, y, X, groups, k_fe, j):
    """Cluster-robust (CR1, as fe_fit) t-statistic of column j. y, X are raw; q is the FE basis."""
    r = np.column_stack([y, X])
    r = r - q @ (q.T @ r)
    yt, xt = r[:, 0], r[:, 1:]
    n, k = xt.shape
    bread = np.linalg.inv(xt.T @ xt)
    beta = bread @ xt.T @ yt
    u = yt - xt @ beta
    g_codes, g_idx = np.unique(groups, return_inverse=True)
    scores = np.zeros((len(g_codes), k))
    np.add.at(scores, g_idx, xt * u[:, None])
    c = len(g_codes) / (len(g_codes) - 1) * (n - 1) / (n - k - k_fe + 1)
    se = np.sqrt(c * (bread @ (scores.T @ scores) @ bread)[j, j])
    return beta[j] / se, beta[j]


def permutation_t(d, terms, focus, n_perm, seed=SEED):
    """Studentised permutation: each district's hill1_tv path is replaced by that of a random district of the SAME state
    (same years), keeping the state-year structure. p = share of permuted |t| >= observed |t| (p_perm_t)."""
    q = fe_basis(d)
    k_fe = d.dist_code.nunique() + d.sy.nunique() - 1
    dd = d.sort_values(["dist_code", "year"]).reset_index(drop=True)
    q = fe_basis(dd)
    years = sorted(dd.year.unique())
    assert dd.groupby("dist_code").year.nunique().eq(len(years)).all(), "permutation needs a balanced panel"
    zmat = dd.pivot(index="dist_code", columns="year", values="z")
    state = dd.drop_duplicates("dist_code").set_index("dist_code").state_name.reindex(zmat.index)
    y = dd.anom.to_numpy()
    groups = dd.dist_code.to_numpy()
    order = {c: i for i, c in enumerate(zmat.index)}
    row_i = dd.dist_code.map(order).to_numpy()
    col_i = dd.year.map({yr: j for j, yr in enumerate(years)}).to_numpy()
    j = terms.index(focus)

    def build(z):
        X = {"spei": dd.spei.to_numpy(), "z": z, "spei_z": dd.spei.to_numpy() * z}
        for kname in ("moderate", "severe", "extreme"):
            X[kname + "_z"] = dd[kname].to_numpy() * z
            X[kname] = dd[kname].to_numpy()
        return np.column_stack([X[t] for t in terms])

    t_obs, _ = fast_t(q, y, build(dd.z.to_numpy()), groups, k_fe, j)
    rng = np.random.default_rng(seed)
    stats = []
    idx_by_state = {s: np.where(state.to_numpy() == s)[0] for s in state.unique()}
    for _ in range(n_perm):
        perm = np.arange(len(zmat))
        for s, ix in idx_by_state.items():
            perm[ix] = rng.permutation(ix)
        z = zmat.to_numpy()[perm[row_i], col_i]
        stats.append(fast_t(q, y, build(z), groups, k_fe, j)[0])
    stats = np.asarray(stats)
    return float(t_obs), float((1 + (np.abs(stats) >= abs(t_obs)).sum()) / (n_perm + 1))


# ------------------------------------------------------------------ analyses

def panel_models(d, label):
    rows = []
    r, f = fit_rows(f"{label}:core", d, ["spei", "spei_z", "z"]); rows += r
    r, _ = fit_rows(f"{label}:hill3", d, ["spei", "spei_z3", "z3"]); rows += r
    r, _ = fit_rows(f"{label}:dry_wet", d, ["dry", "wet", "dry_z", "wet_z", "z"]); rows += r
    r, _ = fit_rows(f"{label}:even_years", d[d.year % 2 == 0], ["spei", "spei_z", "z"]); rows += r
    r, _ = fit_rows(f"{label}:cluster_state", d, ["spei", "spei_z", "z"], cluster="state_name",
                    note="clustered by state: spatial-correlation check (Conley errors not implemented)"); rows += r
    # leave-one-state-out
    loso = []
    for s in sorted(d.state_name.unique()):
        sub = d[d.state_name != s]
        if sub.dist_code.nunique() < 30:
            continue
        ff = fe_fit(sub, "anom", ["spei", "spei_z", "z"], cluster="dist_code", fe=FE)
        se = np.sqrt(ff["cov"].loc["spei_z", "spei_z"])
        loso.append({"model": f"{label}:leave_out_state", "dropped_state": s, "beta_spei_z": ff["params"]["spei_z"], "se": se,
                     "p": 2 * st.t.sf(abs(ff["params"]["spei_z"] / se), ff["df_t"]), "n_obs": ff["nobs"],
                     "n_districts": sub.dist_code.nunique()})
    return rows, pd.DataFrame(loso), f


def irrigation_models(p_all, label):
    """Does diversity survive irrigation? Needs mirror controls, so years <= 2015 only."""
    rows = []
    d = prep(add_controls(p_all)).query("year <= 2015")
    base = d.dropna(subset=["irrig_lag", "fert_lag"]).copy()
    for c in ("irrig_lag", "fert_lag"):
        base["z_" + c] = (base[c] - base[c].mean()) / base[c].std()
        base["spei_x_" + c] = base.spei * base["z_" + c]
    r, _ = fit_rows(f"{label}:le2015_core", base, ["spei", "spei_z", "z"], note="same sample as the irrigation models"); rows += r
    r, _ = fit_rows(f"{label}:le2015_irrig", base, ["spei", "spei_z", "z", "spei_x_irrig_lag", "z_irrig_lag"],
                    note="irrig_lag = gross irrigated / gross cropped area at t-1, from the ICRISAT mirror"); rows += r
    r, _ = fit_rows(f"{label}:le2015_irrig_fert", base, ["spei", "spei_z", "z", "spei_x_irrig_lag", "z_irrig_lag", "spei_x_fert_lag", "z_fert_lag"]); rows += r
    mean_irr = base.groupby("dist_code").irrig_lag.mean()
    rain = base[base.dist_code.isin(mean_irr.index[mean_irr < mean_irr.median()])]
    r, _ = fit_rows(f"{label}:le2015_rainfed_half", rain, ["spei", "spei_z", "z"], note="districts below the median mean irrigation share"); rows += r
    return rows


def faces_table(m, label):
    rows = []
    for face in FACES:
        for pred in ("hill1", "resp_div", "resp_divergence"):
            r = spearman_boot(m[pred], m[face])
            rows.append({"window": label, "face": face, "predictor": pred, **r})
    return pd.DataFrame(rows)


def validation_table(clim, years, col):
    doc = [2002, 2009, 2014, 2015]          # years with documented deficient all-India monsoon rainfall; citation to be added by the validation owner
    c = clim[clim.year.between(years[0], years[-1])]
    g = c.groupby("year")[col].agg(share_drought=lambda s: (s <= -1).mean(), mean_spei="mean").reset_index()
    g["documented_drought_year"] = g.year.isin(doc)
    return g


def figures(core_rows, ext_rows, sev, spei_label):
    FIG.mkdir(parents=True, exist_ok=True)
    zs = np.linspace(-2, 2, 41)
    fig, ax = plt.subplots(figsize=(8, 5))
    for rows, name, colr in ((core_rows, "DES core (1998-2022, all crops)", "tab:blue"), (ext_rows, "5-crop extended (1993-2015)", "tab:orange")):
        if rows is None:
            continue
        r = rows.set_index("term")
        cov = rows.attrs["cov"]
        b = r.beta["spei"] + r.beta["spei_z"] * zs
        v = cov.loc["spei", "spei"] + zs ** 2 * cov.loc["spei_z", "spei_z"] + 2 * zs * cov.loc["spei", "spei_z"]
        ax.plot(zs, b, color=colr, label=name)
        ax.fill_between(zs, b - 1.96 * np.sqrt(v), b + 1.96 * np.sqrt(v), color=colr, alpha=0.2)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("District crop diversity, Hill-1 of the previous 5 years (standard deviations from the mean)")
    ax.set_ylabel("Effect of +1 SPEI-12 on production anomaly\n(share of trend)")
    ax.set_title("Drought sensitivity vs crop diversity within Indian districts")
    ax.legend(frameon=False)
    fig.text(0.01, 0.01, f"Source: DES district APY, ICRISAT DLD mirror, SPEIbase v2.10 ({spei_label}). District + state-year FE, 95% CI clustered by district.", fontsize=6)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG / "01_panel_interaction.png", dpi=300, facecolor="white")
    plt.close(fig)

    if sev is not None:
        fig, ax = plt.subplots(figsize=(8, 5))
        s = sev[sev.term.isin(["moderate_z", "severe_z", "extreme_z"])]
        for i, (win, colr) in enumerate((("core", "tab:blue"), ("ext5", "tab:orange"))):
            t = s[s.model.str.startswith(win + ":")].set_index("term").reindex(["moderate_z", "severe_z", "extreme_z"])
            if t.beta.isna().all():
                continue
            x = np.arange(3) + (i - 0.5) * 0.25
            ax.errorbar(x, t.beta, yerr=[t.beta - t.ci_lo, t.ci_hi - t.beta], fmt="o", color=colr, capsize=3, label=win)
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_xticks(range(3)); ax.set_xticklabels(["Moderate\n-1.5 < SPEI <= -1.0", "Severe\n-2.0 < SPEI <= -1.5", "Extreme\nSPEI <= -2.0"])
        ax.set_ylabel("Diversity x drought-bin coefficient\n(change in anomaly per SD of diversity)")
        ax.set_title("Does diversity matter more in worse droughts? Indian districts")
        ax.legend(title="window", frameon=False)
        fig.text(0.01, 0.01, f"Source: as Figure 1 ({spei_label}). 95% CI clustered by district.", fontsize=6)
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        fig.savefig(FIG / "02_severity.png", dpi=300, facecolor="white")
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perm", type=int, default=499)
    n_perm = ap.parse_args().perm
    OUT.mkdir(parents=True, exist_ok=True); STATS.mkdir(parents=True, exist_ok=True)
    clim = pd.read_parquet(IN / "climate.parquet")
    col = spei_variable(clim)
    print(f"drought variable: {col}")

    panel_rows, loso, sev_rows, sev_counts, faces, perm_rows, fig_rows, summ = [], [], [], [], [], [], {}, []
    core_frames = None
    for window in ("core", "ext5"):
        data, _, years, dist = build(window)
        for sample in ("core", "stable"):
            D = data[sample]
            m, p = D["metrics"], D["panel"]
            label = f"{window}" if sample == "core" else f"{window}_allstable"
            print(f"\n[{label}] districts selected {D['n_selected']}, usable {len(m)}, dropped {len(D['dropped'])} "
                  f"({D['dropped'].reason.value_counts().to_dict()})")
            assert m.dist_code.is_unique and np.isfinite(m.phi_sync).all() and (m.hill1 >= 1).all()
            d = prep(p)
            # severity bins
            sv = severity_frame(d)
            sv_terms = ["moderate", "severe", "extreme", "moderate_z", "severe_z", "extreme_z", "z"]
            r, _ = fit_rows(f"{label}:severity", sv, sv_terms); sev_rows += r
            tert = pd.qcut(sv.groupby("dist_code").z.transform("mean"), 3, labels=["low", "mid", "high"])
            cnt = pd.DataFrame({"model": label, "bin": ["moderate", "severe", "extreme"],
                                "n_district_years": [int(sv[b].sum()) for b in ("moderate", "severe", "extreme")],
                                "n_high_diversity_tercile": [int(sv.loc[tert == "high", b].sum()) for b in ("moderate", "severe", "extreme")]})
            sev_counts.append(cnt)
            if sample == "core" and n_perm > 0:
                t, pperm = permutation_t(sv, sv_terms, "extreme_z", n_perm)
                perm_rows.append({"model": label, "term": "extreme_z", "t_obs": t, "p_perm_t": pperm, "n_perm": n_perm})
                print(f"  extreme x diversity: t = {t:.2f}, p_perm_t = {pperm:.3f} ({n_perm} permutations)")
            r, ld, f = panel_models(d, label)
            panel_rows += r; loso.append(ld)
            if sample == "core":
                tab = pd.DataFrame(coef_rows(f)); tab.attrs["cov"] = f["cov"]; fig_rows[window] = tab
                panel_rows += irrigation_models(p, label)
                faces.append(faces_table(m, window))
                m.assign(window=window, spei_variable=col).to_parquet(OUT / ("metrics.parquet" if window == "core" else "metrics_ext5.parquet"), index=False)
                p.assign(window=window, spei_variable=col).to_parquet(OUT / ("panel.parquet" if window == "core" else "panel_ext5.parquet"), index=False)
                if window == "core":
                    D["crop_anom"].to_parquet(OUT / "crop_anomalies.parquet", index=False)
                    D["dropped"].to_csv(OUT / "dropped_districts.csv", index=False)
                    validation_table(clim, years, col).to_csv(paths.VALIDATION_DIR / "india_drought_years.csv", index=False)
                n_ev = int(m.n_events.sum())
                summ.append({"window": window, "districts": len(m), "years": f"{years[0]}-{years[-1]}", "n_events": n_ev,
                             "districts_with_event": int((m.n_events > 0).sum()), "median_hill1": m.hill1.median(),
                             "median_value_coverage": m.value_coverage.median(), "spei_variable": col})

    cols = ["model", "term", "beta", "se", "ci_lo", "ci_hi", "p", "n_obs", "n_districts", "r2_within", "note"]
    pd.DataFrame(panel_rows)[cols].assign(spei_variable=col).to_csv(STATS / "india_panel.csv", index=False)
    sev = pd.DataFrame(sev_rows)[cols].assign(spei_variable=col)
    sev.to_csv(STATS / "india_severity.csv", index=False)
    pd.concat(sev_counts).to_csv(STATS / "india_severity_counts.csv", index=False)
    pd.DataFrame(perm_rows).to_csv(STATS / "india_severity_perm.csv", index=False)
    pd.concat(loso).to_csv(STATS / "india_leave_out_state.csv", index=False)
    fa = pd.concat(faces)
    nat = pd.read_csv(STATS / "spearman.csv").rename(columns={"rho": "rho_national", "p": "p_national", "n": "n_national"})[["face", "predictor", "rho_national", "p_national", "n_national"]]
    fa.to_csv(STATS / "india_spearman.csv", index=False)
    fa.merge(nat, on=["face", "predictor"], how="left").to_csv(STATS / "india_vs_national.csv", index=False)
    pd.DataFrame(summ).to_csv(STATS / "india_summary.csv", index=False)
    figures(fig_rows.get("core"), fig_rows.get("ext5"), sev, col)

    show = pd.DataFrame(panel_rows)
    print("\nspei x diversity (core model), by window:")
    print(show[show.term == "spei_z"][["model", "beta", "se", "p", "n_obs", "n_districts"]].round(4).to_string(index=False))
    print("\nfaces vs hill1 (district Spearman):")
    print(fa[fa.predictor == "hill1"][["window", "face", "rho", "p", "n"]].round(3).to_string(index=False))
    print("\n" + pd.DataFrame(summ).to_string(index=False))


if __name__ == "__main__":
    main()
