"""Kharif-only crop table from the DES district APY -> data/clean/india/{crops_kharif.parquet,districts_kharif.csv}, and the panel on it.

The core table (src.india.clean) takes the "Total" row or sums seasons, so district totals include rabi, summer and whole-year crops while
the monsoon drought variable only describes the kharif water balance. Here only monsoon-season rows are used:
  primary   season in {Kharif, Autumn, Winter}  (Autumn and Winter are the eastern-state monsoon rice seasons, sown May-July)
  strict    season == Kharif only               (variant)
Rabi, Summer, Whole Year and Total rows are left out; a crop is in the table only if it has monsoon-season rows in all 25 years
(no "Other ..."). Stable district: monsoon-season rows in every year. Everything else (price, unit and yield rules, split flag, value index,
event rule, model) is the core design; the crop table is built with the same functions and constants of src.india.clean.
Caveat: states that report only a "Total" row (no season split) drop out, so this sample is tilted to states that publish seasons;
`n_districts` and the state counts are printed.

Models: kharif crops x SPEI-4 of September (matching variable) and kharif crops x December SPEI-12 (the core variable).
Writes outputs/stats/india_kharif_crops.csv, india_kharif_crops_perm.csv.
Run: python -m src.india.kharif_crops
"""
import numpy as np
import pandas as pd

from ..common import paths
from . import panel as P
from .clean import IMPLAUSIBLE, OUT_DIR, PRICE_ITEM, RAW, UNIT_ISSUE, YEARS, load_prices, split_flags

PRIMARY = ("Kharif", "Autumn", "Winter")
STRICT = ("Kharif",)


def build_table(raw, seasons):
    d = raw[raw.season.isin(seasons)]
    yrs = d.groupby("crop_name").y.nunique()
    fixed = sorted(c for c in yrs[yrs == len(YEARS)].index if not c.startswith("Other"))
    d = d[d.crop_name.isin(fixed)]
    key = ["y", "district_code", "crop_name"]
    a = d.groupby(key).agg(area_ha=("area", "sum"), prod_t=("production", lambda s: s.sum(min_count=1))).reset_index()
    a = a[a.area_ha > 0]
    state_of = raw.sort_values("y").groupby("district_code").state_name.last()
    first = raw.groupby("district_code").y.min()
    n_years = a.groupby("district_code").y.nunique()
    stable = set(n_years[n_years == len(YEARS)].index)
    flags = split_flags(a.rename(columns={"crop_name": "crop"}), state_of, first)
    crops = a.rename(columns={"y": "year", "crop_name": "crop"})
    crops["dist_code"] = crops.district_code.astype(int).astype(str)
    y0 = crops.prod_t / crops.area_ha
    bad = (crops.crop == UNIT_ISSUE) | (y0 > IMPLAUSIBLE * y0.groupby(crops.crop).transform("median"))
    crops.loc[bad, "prod_t"] = np.nan
    prices = load_prices()
    price = {k: float(prices[v]) for k, v in PRICE_ITEM.items() if v in prices.index}
    crops["yield_t_ha"] = crops.prod_t / crops.area_ha
    crops["price_intl_dollars_t"] = crops.crop.map(price)
    crops["value_const"] = crops.prod_t * crops.price_intl_dollars_t
    crops["is_missing"] = crops.prod_t.isna()
    crops = crops[["dist_code", "year", "crop", "area_ha", "prod_t", "yield_t_ha", "price_intl_dollars_t", "value_const", "is_missing"]]
    crops = crops.sort_values(["dist_code", "year", "crop"]).reset_index(drop=True)
    assert not crops.duplicated(["dist_code", "year", "crop"]).any() and crops.area_ha.gt(0).all()
    dist = pd.DataFrame({"dist_code": state_of.index.astype(int).astype(str)})
    dist["stable"] = state_of.index.isin(stable)
    dist["split_flag"] = state_of.index.isin(flags)
    dist["core"] = dist.stable & ~dist.split_flag
    return crops, dist, fixed


def fit(label, var, rows, perm_rows, n_perm, tables):
    data, col, years, dist = P.build("kharif", col=var, tables=tables)
    D = data["core"]
    m, d = D["metrics"], P.prep(D["panel"])
    print(f"[{label}] drought variable {col}: districts selected {D['n_selected']}, usable {len(m)}, events {int(m.n_events.sum())}")
    for model, terms, kw in ((f"{label}", ["spei", "spei_z", "z"], {}), (f"{label}_state_cluster", ["spei", "spei_z", "z"], {"cluster": "state_name"})):
        r, _ = P.fit_rows(model, d, terms, **kw); rows += r
    sv = P.severity_frame(d)
    st = ["moderate", "severe", "extreme", "moderate_z", "severe_z", "extreme_z", "z"]
    r, _ = P.fit_rows(f"{label}_severity", sv, st); rows += r
    t, pp = P.permutation_t(sv, st, "extreme_z", n_perm)
    perm_rows.append({"model": label, "term": "extreme_z", "t_obs": t, "p_perm_t": pp, "n_perm": n_perm})
    from scipy.stats import spearmanr
    for face in P.FACES:
        ok = m[["hill1", face]].dropna()
        rho, p = spearmanr(ok.hill1, ok[face]); rows.append({"model": f"{label}_spearman_hill1", "term": face, "beta": rho, "p": p, "n_obs": len(ok)})
    return m, d


def main(n_perm=499):
    raw = pd.read_csv(RAW, low_memory=False)
    raw["y"] = raw.year.str[:4].astype(int)
    raw = raw[raw.y.isin(YEARS)].copy()
    assert (raw.area_unit == "Hectare").all() and (raw.production_unit == "Tonnes").all()
    core_area = pd.read_parquet(OUT_DIR / "crops.parquet").groupby("year").area_ha.sum()
    for name, seasons in (("strict", STRICT), ("primary", PRIMARY)):
        crops, dist, fixed = build_table(raw, seasons)
        share = crops.groupby("year").area_ha.sum() / core_area
        print(f"{name} {seasons}: {len(fixed)} crops, rows {len(crops):,}, districts {crops.dist_code.nunique()}, stable {int(dist.stable.sum())}, "
              f"core {int(dist.core.sum())}; area vs core table {share.min():.0%}-{share.max():.0%}")
        if name == "primary":
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            crops.to_parquet(OUT_DIR / "crops_kharif.parquet", index=False)
            dist.to_csv(OUT_DIR / "districts_kharif.csv", index=False)
            prim = (crops, dist)
        else:
            strict = (crops, dist)
    rows, perm = [], []
    m1, d1 = fit("kharif_crops_spei4", "spei4kh_w", rows, perm, n_perm, prim)
    fit("kharif_crops_spei12", "spei12_w", rows, perm, n_perm, prim)
    fit("kharif_strict_spei4", "spei4kh_w", rows, perm, n_perm, strict)
    cols = ["model", "term", "beta", "se", "ci_lo", "ci_hi", "p", "n_obs", "n_districts", "r2_within", "note"]
    out = pd.DataFrame(rows).reindex(columns=cols)
    assert out.beta.notna().all()
    out.to_csv(paths.STATS_DIR / "india_kharif_crops.csv", index=False)
    pd.DataFrame(perm).to_csv(paths.STATS_DIR / "india_kharif_crops_perm.csv", index=False)
    print("\nspei x diversity:\n" + out[out.term == "spei_z"][["model", "beta", "se", "p", "n_obs", "n_districts"]].round(4).to_string(index=False))
    print("\nextreme x diversity (clustered):\n" + out[out.term == "extreme_z"][["model", "beta", "se", "p"]].round(4).to_string(index=False))
    print("\nstudentised permutation:\n" + pd.DataFrame(perm).round(3).to_string(index=False))
    print("\nSpearman vs hill1:\n" + out[out.model.str.contains("spearman")][["model", "term", "beta", "p", "n_obs"]].round(3).to_string(index=False))
    st = pd.read_csv(OUT_DIR / "districts.csv", dtype={"dist_code": str}).set_index("dist_code").state_name
    used = m1.dist_code.map(st).value_counts()
    print("\nstates in the primary sample (districts):", used.head(12).to_dict(), "... total states", len(used))


if __name__ == "__main__":
    main()
