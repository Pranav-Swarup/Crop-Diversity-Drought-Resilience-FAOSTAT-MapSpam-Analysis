"""Archived ICRISAT/TCI district tables, source reconciliation and estimates. Author: Pmjarvis."""

import hashlib
import io
import json
from zipfile import ZipFile

import geopandas as gpd
import numpy as np
import pandas as pd

from . import district
from .storage import BASE, RAW, fetch

PINS = BASE / "district_sources.json"
ARCHIVE = RAW / "icrisat_archive"
PERIOD = list(range(1966, 2018))
BASE_YEAR = 2015
DLD = "RR_IND_2025_488/Reproducbility Package/raw/DLD/"
IDS = {"Dist Code": "district_code", "State Code": "state_code", "Year": "year"}
FHP_LABEL = {
    "SESAMUM": "SEASMUM",
    "RAPESEED AND MUSTARD": "RAPE AND MUSTARD",
    "SUGARCANE": "SUGARCANE GUR",
}
CROP_EXCLUSIONS = {
    "KHARIF SORGHUM": "included in annual sorghum",
    "RABI SORGHUM": "included in annual sorghum",
    "OILSEEDS": "aggregate overlaps individual oilseeds",
    "MINOR PULSES": "residual group of several species",
}
DES_NAMES = {
    "Rice": "RICE",
    "Wheat": "WHEAT",
    "Jowar": "SORGHUM",
    "Bajra": "PEARL MILLET",
    "Maize": "MAIZE",
    "Ragi": "FINGER MILLET",
    "Barley": "BARLEY",
    "Gram": "CHICKPEA",
    "Arhar/Tur": "PIGEONPEA",
    "Groundnut": "GROUNDNUT",
    "Sesamum": "SESAMUM",
    "Rapeseed &Mustard": "RAPESEED AND MUSTARD",
    "Safflower": "SAFFLOWER",
    "Castor Seed": "CASTOR",
    "Linseed": "LINSEED",
    "Sunflower": "SUNFLOWER",
    "Soyabean": "SOYABEAN",
    "Cotton(Lint)": "COTTON",
    "Sugarcane": "SUGARCANE",
}


def files():
    result = {}
    for role, spec in json.loads(PINS.read_text()).items():
        urls = [spec["source_url"]]
        if "captures" in spec:
            urls = [f"https://web.archive.org/web/{t}id_/{urls[0]}" for t in spec["captures"]]
        for url in urls:
            try:
                result[role] = fetch(url, ARCHIVE / spec["file"], "sha256:" + spec["sha256"])
                break
            except OSError:
                if url == urls[-1]:
                    raise
    return result


def table(path):
    payload = json.loads(path.read_bytes())
    if not payload["description"].endswith(": APPORTIONED"):
        raise ValueError(f"Unexpected series in {path.name}")
    headers = {h["header"]: h["unit"] for h in payload["headers"]}
    if len(headers) != len(payload["headers"]):
        raise ValueError(f"Repeated field in {path.name}")
    data = pd.DataFrame(payload["data"], columns=list(headers)).rename(columns=IDS)
    data["year"] = pd.to_numeric(data.year, errors="raise").astype(int)
    if (
        data.duplicated(["district_code", "year"]).any()
        or data.groupby("district_code").state_code.nunique().ne(1).any()
    ):
        raise ValueError(f"Ambiguous district-year identifiers in {path.name}")
    return data, headers, payload["items"]


def measurements(data, headers, field, unit, factor):
    if field not in headers:
        return np.nan
    if headers[field] != unit:
        raise ValueError(f"Unexpected units for {field}: {headers[field]}")
    values = pd.to_numeric(data[field], errors="raise").astype(float)
    if (~np.isfinite(values) | ((values < 0) & (values != -1))).any():
        raise ValueError(f"Unrecognized missing or negative value in {field}")
    return values.mask(values.eq(-1)) * factor


def crop_tables(raw):
    tables = {r: table(raw[r]) for r in ["apy", "prices", "irrigation"]}
    apy, headers, items = tables["apy"]
    ids = ["state_code", "district_code", "year"]
    rows = {r: [] for r in tables}
    codes = []
    for crop in items:
        price = FHP_LABEL.get(crop, crop) + " HARVEST PRICE"
        irrig = ("RAPE AND MUSTARD" if crop == "RAPESEED AND MUSTARD" else crop) + " IRRIGATED AREA"
        reason = CROP_EXCLUSIONS.get(crop, "")
        if crop + " PRODUCTION" not in headers:
            reason = "area only; no production series"
        codes.append(
            {
                "crop": crop,
                "inclusion": "excluded" if reason else "included",
                "reason": reason,
                "price_field": price if price in tables["prices"][1] else None,
                "irrigation_field": irrig if irrig in tables["irrigation"][1] else None,
                "production_form": {
                    "RICE": "rice, milled",
                    "SUGARCANE": "gur, raw sugar",
                    "COTTON": "lint",
                }.get(crop, crop.lower()),
            }
        )
        if reason:
            continue
        specs = {
            "apy": [
                ("area_ha", crop + " AREA", "1000 ha", 1000),
                ("prod_t", crop + " PRODUCTION", "1000 tons", 1000),
            ],
            "prices": [("price_inr_t", price, "Rs per Quintal", 10)],
            "irrigation": [("irrigated_ha", irrig, "1000 ha", 1000)],
        }
        for role, fields in specs.items():
            data, units, _ = tables[role]
            d = data[ids].assign(crop=crop)
            for target, source, unit, factor in fields:
                d[target] = measurements(data, units, source, unit, factor)
            if role == "prices":
                d["price_inr_t"] = d.price_inr_t.where(d.price_inr_t.gt(0))
            rows[role].append(d)
    norm = {r: pd.concat(frames, ignore_index=True) for r, frames in rows.items()}
    names = apy[["state_code", "district_code", "State Name", "Dist Name"]].drop_duplicates()
    names = names.rename(columns={"State Name": "state_name", "Dist Name": "district_name"})
    names["dist_code"] = names.state_code + "/" + names.district_code
    if names.district_code.duplicated().any():
        raise ValueError("District codes must uniquely identify the archived geography")
    for role in ["prices", "irrigation"]:
        their = tables[role][0][["district_code", "state_code"]].drop_duplicates()
        overlap = names.merge(their, on="district_code")
        if overlap.state_code_x.ne(overlap.state_code_y).any():
            raise ValueError(f"Different state assignments in {role}")
    crop = district.prepare_crop(**norm, reference_year=BASE_YEAR, rounded_areas=True)
    crop = crop.merge(names[["dist_code", "state_name"]], on="dist_code", validate="m:1")
    # The publisher explicitly marks these places/periods unavailable in the 1966–2017 manual.
    crop["irrigation_manual_conflict"] = crop.irrigated_ha.notna() & (
        crop.state_name.eq("West Bengal") | (crop.state_name.eq("Maharashtra") & crop.year.gt(2002))
    )
    crop.loc[crop.irrigation_manual_conflict, "irrigation_share"] = np.nan
    crop["reference_price_inr_t_reported"] = crop.reference_price_inr_t
    crop["price_manual_conflict"] = (
        crop.state_name.eq("Orissa") & crop.year.ge(1988) & crop.price_inr_t.notna()
    )
    crop.loc[crop.state_name.eq("Orissa"), ["reference_price_inr_t", "value_const"]] = np.nan
    return crop, names, pd.DataFrame(codes), tables


def official_map(path, names):
    g = gpd.read_file(path)
    if (
        g.crs is None
        or g.geometry.is_empty.any()
        or g.geometry.isna().any()
        or not g.geometry.is_valid.all()
    ):
        raise ValueError("Invalid official district geography")
    g["district_code"] = g.ID.astype(str)
    mapped = g.merge(names[["district_code", "dist_code"]], on="district_code", validate="m:1")
    if set(mapped.dist_code) != set(names.dist_code):
        raise ValueError("An APY district lacks an official polygon")
    return (
        mapped[["dist_code", "geometry"]].to_crs(4326).dissolve(by="dist_code", as_index=False),
        len(g),
        len(mapped),
    )


def bank_source(archive, name):
    with ZipFile(archive) as z:
        hashes = pd.read_csv(z.open("RR_IND_2025_488/data_hash_report.csv"))
        entry = hashes.loc[hashes.path.eq("/raw/DLD/" + name), "sha256sum"].item()
        content = z.read(DLD + name)
    if hashlib.sha256(content).hexdigest() != entry:
        raise ValueError(f"World Bank file disagrees with its published hash: {name}")
    return io.BytesIO(content)


def reconcile_apy(tables, archive):
    apy, units, _ = tables["apy"]
    wb = pd.read_csv(
        bank_source(archive, "Apportioned/Crops_Area Production Yield.csv"), dtype=str
    ).rename(columns=IDS)
    wb["year"] = wb.year.astype(int)
    key = ["district_code", "year"]
    wb, api = wb.set_index(key), apy.set_index(key)
    if not wb.index.is_unique or not wb.index.symmetric_difference(api.index).empty:
        raise ValueError("World Bank and portal APY district-years differ")
    api = api.loc[wb.index]
    rows = []
    for col, unit in units.items():
        if not unit:
            continue
        left = pd.to_numeric(api[col])
        right = pd.to_numeric(wb[f"{col} ({unit})"])
        rows.append(
            {
                "field": col,
                "unit": unit,
                "cells": len(left),
                "missing_sentinel": int(left.eq(-1).sum()),
                "different": int(left.ne(right).sum()),
                "max_abs_difference": float((left - right).abs().max()),
            }
        )
    for col in ["state_code", "State Name", "Dist Name"]:
        if not api[col].str.strip().eq(wb[col].str.strip()).all():
            raise ValueError(f"Geography name/code mismatch in {col}")
    out = pd.DataFrame(rows)
    if out.different.sum():
        raise ValueError("Portal APY values disagree with the deposited export")
    return out


def reconcile_farm_prices(tables, archive, crosswalk):
    native, headers, _ = tables["prices"]
    other = pd.read_csv(
        bank_source(archive, "Unapportioned/Prices_Farm Harvest Prices.csv"), dtype=str
    ).rename(columns=IDS)
    other["year"] = other.year.astype(int)
    key = ["state_code", "district_code", "year"]
    merged = native.merge(other, on=key, validate="1:1").merge(
        crosswalk[["district_code", "history_change"]], on="district_code", validate="m:1"
    )
    rows = []
    for col, unit in headers.items():
        if not unit:
            continue
        a = pd.to_numeric(merged[col])
        b = pd.to_numeric(merged[f"{col} ({unit})"])
        for changed, ix in merged.groupby("history_change").groups.items():
            x, y = a.loc[ix], b.loc[ix]
            ok = x.gt(0) & y.gt(0)
            diff = (x[ok] - y[ok]).abs()
            rows.append(
                {
                    "field": col,
                    "unit": unit,
                    "known_boundary_change": changed,
                    "first_year": merged.loc[ix, "year"].min(),
                    "last_year": merged.loc[ix, "year"].max(),
                    "shared_district_years": len(ix),
                    "joint_prices": int(ok.sum()),
                    "identical": int(diff.eq(0).sum()),
                    "only_apportioned": int((x.gt(0) & ~y.gt(0)).sum()),
                    "only_unapportioned": int((~x.gt(0) & y.gt(0)).sum()),
                    "neither_price": int((~x.gt(0) & ~y.gt(0)).sum()),
                    "median_absolute_difference": diff.median(),
                    "max_absolute_difference": diff.max(),
                }
            )
    return pd.DataFrame(rows)


def coverage_table(crop, names, climate, metrics, years):
    usable = district.value_sample(crop, years)
    key = ["dist_code", "crop"]
    balanced = usable.groupby(key).value_const.transform("count").eq(len(years))
    measured = usable.assign(
        balanced_area=usable.area_ha.where(balanced),
        observed_price_area=usable.area_ha.where(usable.reference_price_inr_t.notna()),
    )
    totals = measured.groupby("dist_code").agg(
        reported_area=("area_ha", "sum"),
        priced_area=("observed_price_area", "sum"),
        balanced_area=("balanced_area", "sum"),
    )
    totals["priced_area_share"] = totals.priced_area / totals.reported_area.where(
        totals.reported_area.gt(0)
    )
    totals["balanced_area_share"] = totals.balanced_area / totals.reported_area.where(
        totals.reported_area.gt(0)
    )
    out = names.merge(totals.reset_index(), on="dist_code", how="left", validate="1:1")
    out = out.merge(
        metrics[["dist_code", "n_crops_value", "value_coverage"]],
        how="left",
        on="dist_code",
        validate="1:1",
    )
    out["eligible"] = out.n_crops_value.notna()
    cc = climate[climate.year.isin(years)].groupby("dist_code").spei12_w.count()
    out["climate_years"] = out.dist_code.map(cc).fillna(0).astype(int)
    return out


def archive_exclusions(crop, climate, names, years, dropped):
    """Record the first failed eligibility rule for each district dropped by the presented estimator."""
    from src.metrics.detrend import detrend

    source = crop[crop.year.isin(years)]
    measured = district.value_sample(source, years)
    available = climate[climate.year.isin(years)].groupby("dist_code").spei12_w.count()
    rows = []
    for code in dropped.dist_code:
        raw = source[source.dist_code.eq(code)]
        priced = raw.loc[raw.reference_price_inr_t.notna(), "crop"].nunique()
        balanced, share = district.metrics.district_value(
            measured[measured.dist_code.eq(code)], years
        )
        if available.get(code, 0) < len(years):
            reason = "incomplete district climate"
        elif not priced:
            reason = "no usable 2015 district-crop price"
        elif len(balanced) < district.metrics.MIN_CROPS_VALUE:
            reason = "fewer than three complete positively producing priced crops"
        elif not np.isfinite(share) or share < district.metrics.MIN_COVERAGE:
            reason = "complete crops cover less than 70% of priced value"
        else:
            index = detrend(balanced.sum(axis=0).to_numpy(), x=years)["index"]
            if np.isfinite(index).all():
                raise ValueError(f"Unexplained model exclusion for {code}")
            reason = "output index undefined for at least one fitted trend year"
        rows.append(
            {
                "dist_code": code,
                "reason": reason,
                "crops_with_reference_price": priced,
                "balanced_crops": len(balanced),
                "balanced_share_of_priced_value": share,
                "missing_apy_start_years": ", ".join(map(str, sorted(set(years) - set(raw.year)))),
            }
        )
    frame = pd.DataFrame(
        rows,
        columns=[
            "dist_code",
            "reason",
            "crops_with_reference_price",
            "balanced_crops",
            "balanced_share_of_priced_value",
            "missing_apy_start_years",
        ],
    )
    return frame.merge(
        names[["dist_code", "state_name", "district_name"]], on="dist_code", validate="1:1"
    )


def price_sensitivity(crop, main_panel):
    """Hold district, year, crop quantities, diversity and climate fixed across four price schedules."""
    from src.india import panel
    from src.india.clean import PRICE_ITEM, load_prices
    from src.metrics.detrend import detrend

    key = ["dist_code", "crop"]
    local_names = {year: f"district_{year}" for year in (2014, 2015, 2016)}
    rates = (
        crop[crop.year.isin(local_names)]
        .pivot(index=key, columns="year", values="price_inr_t")
        .rename(columns=local_names)
    )
    intl = load_prices()
    price_by_crop = {
        c: intl.loc[PRICE_ITEM[k]]
        for k, c in DES_NAMES.items()
        if k in PRICE_ITEM and c != "SUGARCANE"
    }
    rates["presented_fao"] = rates.index.get_level_values("crop").map(price_by_crop)
    rates = rates.dropna()
    sources = list(rates.columns)
    d = crop.merge(rates.reset_index(), on=key, validate="m:1")
    d = d[d.dist_code.isin(main_panel.dist_code) & d.area_ha.notna()].copy()
    by = d.groupby(key).prod_t
    d = d[by.transform("count").eq(len(PERIOD)) & by.transform("max").gt(0)].copy()
    groups = d.groupby("dist_code").crop.nunique()
    d = d[d.dist_code.isin(groups[groups.ge(3)].index)]
    if d.empty:
        raise ValueError("No complete district crop baskets for the price comparison")
    for name in sources:
        d[name] = d[name] * d.prod_t
    totals = d.groupby(["dist_code", "year"])[sources].sum().sort_index()
    # Account for all valued production in the main 2015 price index when restricting the basket.
    full = district.value_sample(crop, PERIOD)
    total_value = full.groupby("dist_code").value_const.sum(min_count=1)
    coverage = totals.groupby("dist_code").district_2015.sum() / total_value
    totals = totals.loc[
        totals.index.get_level_values("dist_code").isin(coverage[coverage.ge(0.70)].index)
    ]
    inputs = main_panel[["dist_code", "year", "state_name", "spei", "hill1_tv", "hill1_tv3"]]
    price_panel = totals.reset_index().merge(inputs, on=["dist_code", "year"], validate="1:1")
    for name in sources:
        price_panel["anom_" + str(name)] = price_panel.groupby("dist_code")[name].transform(
            lambda s: detrend(s.to_numpy(), x=PERIOD)["anom"].to_numpy()
        )
    columns = ["anom_" + str(name) for name in sources]
    complete = (
        price_panel[columns].notna().all(axis=1).groupby(price_panel.dist_code).transform("all")
    )
    price_panel = price_panel.loc[complete].copy()
    price_panel["coverage_of_2015_priced_value"] = price_panel.dist_code.map(coverage)
    price_panel["basket_crops"] = price_panel.dist_code.map(groups)
    price_panel["anom"] = price_panel.anom_district_2015
    prep = panel.prep(price_panel)
    estimates = []
    for name in sources:
        rows, _ = panel.fit_rows(
            name, prep.assign(anom=prep["anom_" + name]), ["spei", "spei_z", "z"]
        )
        estimates.extend(rows)
    basket = rates.loc[
        rates.index.isin(pd.MultiIndex.from_frame(d[key].drop_duplicates()))
    ].reset_index()
    return (
        pd.DataFrame(estimates),
        price_panel.drop(columns="anom"),
        basket[basket.dist_code.isin(price_panel.dist_code)],
    )


def analyse_archived(destination):
    from . import district_comparison as compare

    paths = files()
    crop, names, codebook, tables = crop_tables(paths)
    source_check = reconcile_apy(tables, paths["world_bank"])
    shape, n_features, mapped_features = official_map(paths["geography"], names)
    climate = district.district_climate(shape, PERIOD)
    crosswalk = compare.aligned_units(names, shape, paths["world_bank"])
    farm_check = reconcile_farm_prices(tables, paths["world_bank"], crosswalk)
    pairs, external = compare.paired_data(crop, crosswalk, paths["des_irrigation"])
    agreement = compare.agreement(pairs)
    threshold_rows = []
    for threshold in [0.80, 0.85, 0.90, 0.95, 1.0]:
        eligible = crosswalk[crosswalk.included & crosswalk.intersection_over_union.ge(threshold)]
        group = compare.agreement(pairs[pairs.dist_code.isin(eligible.dist_code)])
        for row in agreement[agreement.crop.eq("All comparable crops")].itertuples():
            part = (
                group[
                    (group.source == row.source)
                    & (group.measure == row.measure)
                    & group.crop.eq("All comparable crops")
                ]
                if not group.empty
                else group
            )
            threshold_rows.append(
                {
                    "minimum_overlap": threshold,
                    "aligned_districts": len(eligible),
                    "source": row.source,
                    "measure": row.measure,
                    "observed_pairs": 0 if part.empty else int(part.observed_pairs.item()),
                    "positive_pairs": 0 if part.empty else int(part.positive_pairs.item()),
                    "within_tolerance": 0 if part.empty else int(part.within_tolerance.item()),
                }
            )
    source_check.to_csv(destination / "district_source_reconciliation.csv", index=False)
    farm_check.to_csv(destination / "district_farm_price_agreement.csv", index=False)
    codebook.to_csv(destination / "district_crop_definitions.csv", index=False)
    crosswalk.to_csv(destination / "district_boundary_comparison.csv", index=False)
    pairs.to_parquet(destination / "district_paired_quantities.parquet", index=False)
    agreement.to_csv(destination / "district_quantity_agreement.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(
        destination / "district_boundary_sensitivity.csv", index=False
    )
    crop.to_parquet(destination / "district_crop_year.parquet", index=False)
    climate.to_parquet(destination / "district_climate.parquet", index=False)
    estimates, eligibility, exclusions = [district.presented_benchmarks()], [], []
    for tag, years in [("archive", PERIOD), ("overlap", list(range(1998, 2018)))]:
        print(f"Fitting archived district models for {years[0]}–{years[-1]}", flush=True)
        m, panel, dropped, stats = district.models(crop, climate, years)
        stats["model"] = stats.model.str.replace("official", tag, regex=False)
        stats["basis"] = "observed 2015 district farm-harvest prices, INR"
        estimates.append(stats)
        coverage = coverage_table(crop, names, climate, m, years)
        eligibility.append(coverage.assign(window=tag))
        exclusions.append(
            archive_exclusions(crop, climate, names, years, dropped).assign(window=tag)
        )
        suffix = "" if tag == "archive" else "_overlap"
        panel.to_parquet(destination / f"district_panel{suffix}.parquet", index=False)
        m.to_csv(destination / f"district_metrics{suffix}.csv", index=False)
        if tag == "archive":
            primary_districts = len(m)
            price, matched, basket = price_sensitivity(crop, panel)
            price.to_csv(destination / "district_price_sensitivity.csv", index=False)
            basket.to_csv(destination / "district_price_basket.csv", index=False)
            matched.to_parquet(destination / "district_price_panel.parquet", index=False)
    pd.concat(eligibility, ignore_index=True).to_csv(
        destination / "district_coverage.csv", index=False
    )
    pd.concat(exclusions, ignore_index=True).to_csv(
        destination / "district_exclusions.csv", index=False
    )
    comparison = pd.concat(estimates, ignore_index=True)
    comparison.to_csv(destination / "district_comparison.csv", index=False)
    as_printed = crop.assign(
        reference_price_inr_t=crop.reference_price_inr_t_reported,
        value_const=crop.prod_t * crop.reference_price_inr_t_reported,
        irrigation_share=(crop.irrigated_ha / crop.area_ha.where(crop.area_ha.gt(0))).mask(
            crop.irrigation_invalid
        ),
    )
    print("Checking estimates using the values disputed by the publisher manual", flush=True)
    _, p_printed, _, printed = district.models(as_printed, climate, PERIOD)
    printed["model"] = printed.model.str.replace("official", "as_printed", regex=False)
    printed.to_csv(destination / "district_manual_sensitivity.csv", index=False)
    p_printed.to_parquet(destination / "district_manual_sensitivity_panel.parquet", index=False)
    inventory = {}
    for role, (data, _, _) in tables.items():
        inventory[role] = {
            "rows": len(data),
            "district_codes": int(data.district_code.nunique()),
            "first_year": int(data.year.min()),
            "last_year": int(data.year.max()),
            "codes_outside_apy": sorted(set(data.district_code) - set(names.district_code)),
        }
    return {
        "status": "archived_complete",
        "current_release_status": "unverified",
        "release_id": "ICRISAT/TCI apportioned archive, 1966–2017 crop release",
        "boundary_vintage": "DLD apportioned to 1966 parent districts on the publisher map",
        "analysis_years": [PERIOD[0], PERIOD[-1]],
        "price_reference_year": BASE_YEAR,
        "source_files": json.loads(PINS.read_text()),
        "official_table_coverage": inventory,
        "map_features": n_features,
        "features_assigned_to_apy": mapped_features,
        "mapped_apy_districts": len(shape),
        "source_cells_checked": int(source_check.cells.sum()),
        "source_cell_differences": int(source_check.different.sum()),
        "modern_aligned_districts": int(crosswalk.included.sum()),
        "equal_polygon_pairs": int(crosswalk.exact_geometry.sum()),
        "external_land_use": external,
        "summary": f"Archived ICRISAT/TCI data cover {len(names)} parent districts. The 1966–2017 analysis values production at observed 2015 farm-harvest prices and admits {primary_districts} districts. The live portal’s latest apportioned edition could not be independently established.",
    }
