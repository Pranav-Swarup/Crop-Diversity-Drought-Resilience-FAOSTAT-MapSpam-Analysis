"""Kharif-season drought variable as a robustness check -> outputs/stats/india_kharif.csv.

The core drought variable is the December SPEI-12 (covers the whole agricultural year). Most Indian district area is monsoon
(kharif) cropped, so this variant replaces it by the SPEI-4 of September (June-September water balance), cropland weighted
(`spei4kh_w`, src.india.climate; needs data/raw/spei/spei04.nc). Everything else is the core design: same districts, same
crop table, same model and fixed effects, same event rule (variable <= -1.0, first year of a run, 3 years either side), faces and
diversity recomputed with the new events. Rabi-season crops are in the district totals, so this is a variant of the exposure
measure, not a kharif-only crop analysis (a crop-season split would need season-level DES rows; not done).

Run: python -m src.india.kharif
"""
import pandas as pd

from ..common import paths
from . import panel as P

VAR = "spei4kh_w"


def main():
    clim = pd.read_parquet(P.IN / "climate.parquet")
    assert VAR in clim and clim[VAR].notna().all(), "run src.india.climate with data/raw/spei/spei04.nc first"
    rows, summ, perm = [], [], []
    for window in ("core", "ext5"):
        data, col, years, _ = P.build(window, col=VAR)
        D = data["core"]
        m, d = D["metrics"], P.prep(D["panel"])
        r, f = P.fit_rows(f"{window}:kharif", d, ["spei", "spei_z", "z"]); rows += r
        r, _ = P.fit_rows(f"{window}:kharif_state_cluster", d, ["spei", "spei_z", "z"], cluster="state_name"); rows += r
        sv = P.severity_frame(d)
        terms = ["moderate", "severe", "extreme", "moderate_z", "severe_z", "extreme_z", "z"]
        r, _ = P.fit_rows(f"{window}:kharif_severity", sv, terms); rows += r
        if window == "core":
            t, pp = P.permutation_t(sv, terms, "extreme_z", 499)
            perm.append({"model": window, "term": "extreme_z", "t_obs": t, "p_perm_t": pp, "n_perm": 499})
        summ.append({"window": window, "districts": len(m), "n_events": int(m.n_events.sum()), "districts_with_event": int((m.n_events > 0).sum()),
                     "drought_district_years": int((D["panel"].spei <= -1).sum()), "variable": VAR})
        for face in P.FACES:
            ok = m[["hill1", face]].dropna()
            from scipy.stats import spearmanr
            rho, p = spearmanr(ok.hill1, ok[face]); rows.append({"model": f"{window}:kharif_spearman_hill1", "term": face, "beta": rho, "p": p, "n_obs": len(ok)})
    cols = ["model", "term", "beta", "se", "ci_lo", "ci_hi", "p", "n_obs", "n_districts", "r2_within", "note"]
    out = pd.DataFrame(rows).reindex(columns=cols).assign(spei_variable=VAR)
    assert out.beta.notna().all()
    out.to_csv(paths.STATS_DIR / "india_kharif.csv", index=False)
    pd.DataFrame(perm).to_csv(paths.STATS_DIR / "india_kharif_perm.csv", index=False)
    print(pd.DataFrame(summ).to_string(index=False))
    print("\nspei x diversity:\n" + out[out.term == "spei_z"][["model", "beta", "se", "p", "n_obs"]].round(4).to_string(index=False))
    print("\nextreme x diversity (clustered):\n" + out[out.term == "extreme_z"][["model", "beta", "se", "p"]].round(4).to_string(index=False))
    print("\nstudentised permutation:", perm)
    print("\nSpearman vs hill1:\n" + out[out.model.str.contains("spearman")][["model", "term", "beta", "p", "n_obs"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
