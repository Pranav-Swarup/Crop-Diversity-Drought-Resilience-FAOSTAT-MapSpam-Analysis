"""Phase 4 India district data loader and GeoJSON preparation for the interactive explorer.

Reads data/clean/india/*, outputs/india/*, outputs/stats/india_*.csv, and outputs/validation/india_drought_years.csv.
Simplifies district polygons for display. Tabular and time-series values retain source precision.
"""
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import orient_polygons, set_precision, get_num_geometries

from src.common import paths
from src.metrics.events import find_events
from src.india.metrics import one_district

SIMPLIFY_TOL = 0.012
COORD_DIGITS = 3

INDIA_EXTRA_METRICS = {
    "vuln_drought": (
        "Drought-year Vulnerability (% drought years < 0.9)",
        "Share of district drought years (SPEI-12 <= -1.0) in which district agricultural value drops more than 10% below trend.",
        "Reds",
    ),
    "resp_divergence": (
        "Response Divergence (Ross et al. 2023)",
        "Unweighted divergence of crop drought response slopes (0 = all slopes share the same sign; 1 = symmetric opposite responses).",
        "Plasma",
    ),
    "mean_crop_cv": (
        "Mean Crop CV",
        "Value-share-weighted average coefficient of variation across individual crops in the district.",
        "YlOrRd",
    ),
    "n_events": (
        "Drought Event Count",
        "Number of usable district drought events (first year of a run with SPEI-12 <= -1.0, preceded by 3 drought-free years).",
        "Oranges",
    ),
    "value_coverage": (
        "Balanced Value Coverage (share)",
        "Share of the district's priced agricultural value held by crops with a complete balanced series over the window (>= 70% required).",
        "YlGnBu",
    ),
    "irrig_share": (
        "Mean Irrigation Share (ICRISAT mirror, <= 2015)",
        "District gross irrigated area divided by gross cropped area (mean over 1993-2015 from the ICRISAT mirror controls).",
        "Blues",
    ),
}


def _clean(v):
    if pd.isna(v):
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v) if np.isfinite(v) else None
    return v


def _records_clean(df):
    return [{k: _clean(v) for k, v in row.items()} for row in df.to_dict("records")]


def _build_ext5_crop_anomalies(usable_ext5_codes, clim_df):
    """Compute per-crop yield anomalies and response slopes for the usable ext5 districts using src.india.metrics.one_district."""
    if not paths.INDIA_CROPS_EXT5.exists():
        return pd.DataFrame(columns=["dist_code", "year", "crop", "yield_anom", "value_share", "resp_beta"])
    crops_ext5 = pd.read_parquet(paths.INDIA_CROPS_EXT5)
    crops_ext5["dist_code"] = crops_ext5["dist_code"].astype(str)
    crops_ext5 = crops_ext5[crops_ext5["dist_code"].isin(usable_ext5_codes)]
    years = list(range(1993, 2016))
    clim = clim_df[["dist_code", "year", "spei12_w"]].copy()
    clim = clim[clim["year"].between(years[0], years[-1])]
    ev = find_events(clim.rename(columns={"dist_code": "iso3"}), year_min=years[0], year_max=years[-1])
    ev_by = {k: v for k, v in ev.groupby("iso3")}
    cg = {k: v for k, v in clim.groupby("dist_code")}
    ca_list = []
    for code, g in crops_ext5.groupby("dist_code"):
        if code not in cg:
            continue
        r = one_district(g, cg[code], years, ev_by.get(code, pd.DataFrame({"year": []})))
        if r[0] is not None and r[2] is not None:
            ca_list.append(r[2])
    if not ca_list:
        return pd.DataFrame(columns=["dist_code", "year", "crop", "yield_anom", "value_share", "resp_beta"])
    return pd.concat(ca_list, ignore_index=True)


def load_india_data():
    """Load and validate all Phase 4 Indian district datasets for the interactive explorer."""
    if paths.USE_DUMMY or not paths.INDIA_METRICS.exists() or not paths.INDIA_GEOJSON.exists():
        print("  Phase 4 India district files not loaded (USE_DUMMY=1 or files missing).")
        return None

    print("  Loading Phase 4 India district data...")
    dist_df = pd.read_csv(paths.INDIA_DISTRICTS, dtype={"dist_code": str}).set_index("dist_code")
    dist_ext5_df = (
        pd.read_csv(paths.INDIA_DISTRICTS_EXT5, dtype={"dist_code": str}).set_index("dist_code")
        if paths.INDIA_DISTRICTS_EXT5.exists()
        else pd.DataFrame()
    )
    dropped_df = (
        pd.read_csv(paths.INDIA_DROPPED, dtype={"dist_code": str}).set_index("dist_code")
        if paths.INDIA_DROPPED.exists()
        else pd.DataFrame()
    )

    m_core_df = pd.read_parquet(paths.INDIA_METRICS)
    m_core_df["dist_code"] = m_core_df["dist_code"].astype(str)
    m_ext5_df = pd.read_parquet(paths.INDIA_METRICS_EXT5) if paths.INDIA_METRICS_EXT5.exists() else pd.DataFrame()
    if not m_ext5_df.empty:
        m_ext5_df["dist_code"] = m_ext5_df["dist_code"].astype(str)

    p_core_df = pd.read_parquet(paths.INDIA_PANEL)
    p_core_df["dist_code"] = p_core_df["dist_code"].astype(str)
    p_ext5_df = pd.read_parquet(paths.INDIA_PANEL_EXT5) if paths.INDIA_PANEL_EXT5.exists() else pd.DataFrame()
    if not p_ext5_df.empty:
        p_ext5_df["dist_code"] = p_ext5_df["dist_code"].astype(str)

    clim_df = pd.read_parquet(paths.INDIA_CLIMATE)
    clim_df["dist_code"] = clim_df["dist_code"].astype(str)

    ca_core_df = pd.read_parquet(paths.INDIA_CROP_ANOMALIES)
    ca_core_df["dist_code"] = ca_core_df["dist_code"].astype(str)

    # Controls (irrigation & fertilizer from ICRISAT mirror, 1993-2015)
    irrig_mean, fert_mean = {}, {}
    if paths.INDIA_CONTROLS.exists():
        ctrl_df = pd.read_parquet(paths.INDIA_CONTROLS)
        ctrl_df["dist_code"] = ctrl_df["dist_code"].astype(str)
        irrig_mean = ctrl_df.groupby("dist_code")["irrig_share"].mean().to_dict()
        fert_mean = ctrl_df.groupby("dist_code")["fert_kg_ha"].mean().to_dict()

    # 1. Simplify and round district GeoJSON
    gdf = gpd.read_file(paths.INDIA_GEOJSON)
    gdf["dist_code"] = gdf["dist_code"].astype(str)
    simplified = gdf.geometry.simplify(SIMPLIFY_TOL, preserve_topology=True)
    snapped = set_precision(simplified, grid_size=10 ** -COORD_DIGITS)
    # Retain tiny islands and complete districts if snapping collapses a component.
    keep_original = snapped.is_empty | (get_num_geometries(snapped) < get_num_geometries(simplified))
    gdf["geometry"] = orient_polygons(snapped.where(~keep_original, simplified), exterior_cw=True)
    assert gdf.geometry.is_valid.all() and not gdf.geometry.is_empty.any()
    geojson_data = json.loads(gdf[["dist_code", "geometry"]].to_json(drop_id=True))

    # 2. Build metadata per matched district
    core_set = set(m_core_df["dist_code"])
    ext5_set = set(m_ext5_df["dist_code"]) if not m_ext5_df.empty else set()
    meta = {}
    for code in gdf["dist_code"]:
        if code not in dist_df.index:
            continue
        r = dist_df.loc[code]
        if code in core_set:
            c_status = "included"
        elif code in dropped_df.index:
            c_status = str(dropped_df.loc[code, "reason"])
        elif bool(r["split_flag"]):
            detail = r["split_detail"] if pd.notna(r["split_detail"]) and str(r["split_detail"]) else "area drop"
            c_status = f"split-flagged parent district ({detail})"
        elif not bool(r["stable"]):
            c_status = f"incomplete series ({int(r['n_years'])}/25 years, from {int(r['first_year'])})"
        else:
            c_status = "not in core sample"

        if code in ext5_set:
            e_status = "included"
        elif not dist_ext5_df.empty and code in dist_ext5_df.index:
            er = dist_ext5_df.loc[code]
            if bool(er.get("split_flag", False)):
                e_status = "split-flagged in 1993-2015 mirror"
            elif not bool(er.get("stable", False)):
                e_status = "incomplete series in 1993-2015 mirror"
            else:
                e_status = "value coverage < 70% or < 3 balanced crops"
        else:
            e_status = "not in ICRISAT 5-crop mirror"

        meta[code] = {
            "name": str(r["dist_name"]),
            "state": str(r["state_name"]),
            "n_years": int(r["n_years"]),
            "first_year": int(r["first_year"]),
            "stable": bool(r["stable"]),
            "split_flag": bool(r["split_flag"]),
            "core_status": c_status,
            "ext5_status": e_status,
            "irrig_share": _clean(irrig_mean.get(code)),
            "fert_kg_ha": _clean(fert_mean.get(code)),
        }

    # 3. Build metrics dicts for core and ext5
    metric_cols = [
        "hill1", "resp_div", "resp_divergence", "stability", "resistance", "recovery",
        "vulnerability", "vuln_drought", "phi_sync", "mean_crop_cv",
        "n_events", "n_drought_years", "n_crops_value", "value_coverage", "n_crops_beta",
    ]
    metrics_by_win = {"core": {}, "ext5": {}}
    for win, df in (("core", m_core_df), ("ext5", m_ext5_df)):
        if df.empty:
            continue
        for _, row in df.iterrows():
            code = str(row["dist_code"])
            d = {}
            for col in metric_cols:
                if col in ("n_events", "n_drought_years", "n_crops_value", "n_crops_beta"):
                    d[col] = _clean(int(row[col])) if pd.notna(row[col]) else None
                else:
                    d[col] = _clean(row[col])
            d["irrig_share"] = _clean(irrig_mean.get(code))
            d["fert_kg_ha"] = _clean(fert_mean.get(code))
            metrics_by_win[win][code] = d

    # 4. Build district timeseries (core 1998-2022 and ext5 1993-2015)
    ca_ext5_df = _build_ext5_crop_anomalies(ext5_set, clim_df)
    ts_by_win = {"core": {}, "ext5": {}}
    for win, p_df, ca_df, yr_range in (
        ("core", p_core_df, ca_core_df, (1998, 2022)),
        ("ext5", p_ext5_df, ca_ext5_df, (1993, 2015)),
    ):
        if p_df.empty:
            continue
        clim_sub = clim_df[clim_df["year"].between(yr_range[0], yr_range[1])].copy()
        ev_df = find_events(
            clim_sub[["dist_code", "year", "spei12_w"]].rename(columns={"dist_code": "iso3"}),
            year_min=yr_range[0],
            year_max=yr_range[1],
        )
        ev_by = {str(k): v["year"].astype(int).tolist() for k, v in ev_df.groupby("iso3")}
        clim_by = {str(k): v.set_index("year") for k, v in clim_sub.groupby("dist_code")}
        ca_by = {str(k): v for k, v in ca_df.groupby("dist_code")} if not ca_df.empty else {}

        for code, grp in p_df.groupby("dist_code"):
            code = str(code)
            grp = grp.sort_values("year")
            years = grp["year"].astype(int).tolist()
            spei = [_clean(v) for v in grp["spei"].tolist()]
            anom = [_clean(v) for v in grp["anom"].tolist()]
            hill1_tv = [_clean(v) for v in grp["hill1_tv"].tolist()]

            c_sub = clim_by.get(code)
            if c_sub is not None and "spei4kh_w" in c_sub.columns:
                spei_kh = [_clean(v) for v in c_sub.reindex(years)["spei4kh_w"].tolist()]
            else:
                spei_kh = [None] * len(years)

            drought_years = ev_by.get(code, [])
            all_drought_years = [int(y) for y, s in zip(years, spei) if s is not None and s <= -1.0]

            crops_data = {}
            crop_meta = {}
            a_df = ca_by.get(code)
            if a_df is not None and not a_df.empty:
                shares = a_df.groupby("crop")["value_share"].first().sort_values(ascending=False)
                top_crops = shares.head(6).index.tolist()
                for crop in top_crops:
                    cdf = a_df[a_df["crop"] == crop].set_index("year")
                    aligned = cdf.reindex(years)["yield_anom"].tolist()
                    crops_data[str(crop)] = [_clean(v) for v in aligned]
                    beta_val = cdf["resp_beta"].iloc[0] if len(cdf) else None
                    crop_meta[str(crop)] = {
                        "share": _clean(shares[crop]),
                        "beta": _clean(beta_val),
                    }

            ts_by_win[win][code] = {
                "years": years,
                "spei": spei,
                "spei_kh": spei_kh,
                "anom": anom,
                "hill1_tv": hill1_tv,
                "drought_years": drought_years,
                "all_drought_years": all_drought_years,
                "crops": crops_data,
                "crop_meta": crop_meta,
            }

    # 5. Load Phase 4 statistical and validation CSVs
    stats_files = {
        "summary": paths.STATS_DIR / "india_summary.csv",
        "panel": paths.STATS_DIR / "india_panel.csv",
        "severity": paths.STATS_DIR / "india_severity.csv",
        "severity_counts": paths.STATS_DIR / "india_severity_counts.csv",
        "severity_perm": paths.STATS_DIR / "india_severity_perm.csv",
        "vs_national": paths.STATS_DIR / "india_vs_national.csv",
        "conley": paths.STATS_DIR / "india_conley.csv",
        "kharif": paths.STATS_DIR / "india_kharif.csv",
        "kharif_perm": paths.STATS_DIR / "india_kharif_perm.csv",
        "kharif_crops": paths.STATS_DIR / "india_kharif_crops.csv",
        "kharif_crops_perm": paths.STATS_DIR / "india_kharif_crops_perm.csv",
        "leave_out_state": paths.STATS_DIR / "india_leave_out_state.csv",
        "drought_years": paths.INDIA_DROUGHT_YEARS,
    }
    stats_data = {}
    for key, fpath in stats_files.items():
        stats_data[key] = _records_clean(pd.read_csv(fpath, float_precision="round_trip"))

    # 6. Assertions on loaded Phase 4 data
    poly_codes = {f["properties"]["dist_code"] for f in geojson_data["features"]}
    assert len(metrics_by_win["core"]) == len(m_core_df)
    assert len(metrics_by_win["ext5"]) == len(m_ext5_df)
    assert set(metrics_by_win["core"]) <= poly_codes, "core districts missing from GeoJSON"
    assert set(metrics_by_win["ext5"]) <= poly_codes, "ext5 districts missing from GeoJSON"
    assert set(ts_by_win["core"]) == set(metrics_by_win["core"]), "core ts_data keys mismatch"
    assert set(ts_by_win["ext5"]) == set(metrics_by_win["ext5"]), "ext5 ts_data keys mismatch"
    assert len(stats_data["panel"]) > 0 and len(stats_data["vs_national"]) > 0, "Phase 4 stats tables empty"

    print(
        f"  Phase 4 India loaded: {len(poly_codes)} polygons, "
        f"{len(metrics_by_win['core'])} core districts (1998-2022), "
        f"{len(metrics_by_win['ext5'])} ext5 districts (1993-2015), "
        f"{len(stats_data)} stats tables"
    )
    return {
        "geojson": geojson_data,
        "meta": meta,
        "metrics": metrics_by_win,
        "ts": ts_by_win,
        "stats": stats_data,
        "extra_metrics": {k: list(v) for k, v in INDIA_EXTRA_METRICS.items()},
    }
