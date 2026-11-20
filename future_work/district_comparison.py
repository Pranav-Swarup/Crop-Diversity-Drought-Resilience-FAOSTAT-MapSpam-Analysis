"""Compare observed district quantities across independently encoded Indian source tables. Author: Pmjarvis."""

import json

import geopandas as gpd
import numpy as np
import pandas as pd

from .district import crop_year
from .district_archive import DES_NAMES, bank_source
from .district_land_use import FIELDS as LAND_USE_FIELDS
from .district_land_use import verify_export
from .figures import finish, plt
from .storage import BASE, ROOT

CLEAN = ROOT / "data/clean/india"
FIVE_NAMES = {c.title(): c for c in ["CHICKPEA", "GROUNDNUT", "RICE", "SUGARCANE"]}
FIVE_NAMES["Pearl millet"] = "PEARL MILLET"
LUS_NAMES = {
    n: c
    for n, c in DES_NAMES.items()
    if n not in {"Arhar/Tur", "Rapeseed &Mustard", "Cotton(Lint)"}
}
LUS_NAMES.update(
    {"Arhar(Tur)": "PIGEONPEA", "Rapeseed and Mustard": "RAPESEED AND MUSTARD", "Cotton": "COTTON"}
)
KEY = ["dist_code", "old_code", "crop", "year"]


def canonical(values):
    return (
        values.astype("string")
        .str.lower()
        .str.strip()
        .str.replace(r"[^a-z0-9]", "", regex=True)
        .replace({"orissa": "odisha"})
    )


def aligned_units(names, shape, archive):
    history = pd.read_excel(bank_source(archive, "Apportionment.xlsx"), dtype=str)
    if list(history.columns) != ["var"] or len(history) % 6:
        raise ValueError(
            "Unexpected format for the deposit's transcription of the district history"
        )
    # Appendix 3 records six fields per parent-child allocation, in this order.
    history = pd.DataFrame(
        history["var"].to_numpy().reshape(-1, 6),
        columns=["child", "child_code", "year_formed", "parent", "parent_code", "percent"],
    )
    affected = set(history.parent_code) | set(history.child_code)
    old = pd.read_csv(CLEAN / "district_match.csv", dtype={"dist_code": str})
    old = old.loc[old.matched & old.core].rename(columns={"dist_code": "old_code"})
    geo = (
        gpd.read_file(CLEAN / "districts.geojson")
        .to_crs(6933)
        .rename(columns={"dist_code": "old_code"})
    )
    old = old.merge(geo, on="old_code", validate="1:1")
    old["state_key"] = canonical(old.state_name)
    old["name_key"] = canonical(old.dist_name)
    keys = ["state_key", "name_key"]
    old = old.loc[~old.duplicated(keys, keep=False)]
    locations = names.assign(
        state_key=canonical(names.state_name), name_key=canonical(names.district_name)
    )
    locations["history_change"] = locations.district_code.isin(affected)
    locations["ambiguous_name"] = locations.duplicated(keys, keep=False)
    match = locations.merge(
        old[keys + ["old_code", "geometry"]], on=keys, how="left", validate="m:1"
    )
    match = match.rename(columns={"geometry": "presented_geometry"}).merge(
        shape.to_crs(6933), on="dist_code", validate="1:1"
    )

    def similarity(row):
        if pd.isna(row.old_code):
            return np.nan, False
        a, b = row.geometry, row.presented_geometry
        return a.intersection(b).area / a.union(b).area, a.equals(b)

    overlap = [similarity(r) for r in match.itertuples()]
    match[["intersection_over_union", "exact_geometry"]] = pd.DataFrame(overlap, index=match.index)
    match["included"] = (
        ~match.history_change & ~match.ambiguous_name & match.intersection_over_union.ge(0.80)
    )
    if match.loc[match.included, "old_code"].duplicated().any():
        raise ValueError("Non-unique historical to modern district correspondence")
    return match.drop(columns=["geometry", "presented_geometry"])


def read_irrigation(path, units):
    src = pd.read_csv(
        path, dtype={"district_code": "string", "state_name": "string", "district_name": "string"}
    )
    coverage = {
        "rows": len(src),
        "years": sorted(src.year.dropna().unique()),
        "missing_area": int(src.area.isna().sum()),
    }
    primary = src.set_index("id", verify_integrity=True)
    checked = 0
    for receipt in json.loads((BASE / "district_land_use_checks.json").read_text()):
        if len(src) != receipt["total"]:
            raise ValueError("Land-use export row count differs from its separate datastore")
        for record in receipt["records"]:
            current = primary.loc[record["id"]]
            for field in current.index:
                expected = record[field]
                actual = current[field]
                if pd.isna(actual) and (expected is None or str(expected).lower() == "nan"):
                    continue
                if pd.isna(actual) or str(actual) != str(expected):
                    raise ValueError(f"Land-use record {record['id']} differs in {field}")
            checked += 1
    coverage["illustrative_datastore_records"] = checked
    coverage["datastore_records_checked"] = verify_export(path)
    coverage["datastore_fields_checked"] = coverage["datastore_records_checked"] * len(
        LAND_USE_FIELDS
    )
    src["old_code"] = (
        pd.to_numeric(src.district_code, errors="raise").astype("Int64").astype("string")
    )
    src["state_key"] = canonical(src.state_name)
    src["name_key"] = canonical(src.district_name)
    src = src.merge(
        units[["old_code", "state_key", "name_key", "dist_code"]],
        on=["old_code", "state_key", "name_key"],
        validate="m:1",
    )
    src = src[src.crop_name.isin(LUS_NAMES)].copy()
    src["crop"] = src.crop_name.map(LUS_NAMES)
    src["year"] = src.year.map(crop_year)
    if (src.area.lt(0) | np.isinf(src.area)).any() or src.duplicated(KEY + ["season"]).any():
        raise ValueError("Ambiguous or negative land-use irrigation measurement")
    # Use annual records or the complete, nonoverlapping seasonal partition for this crop.
    # Require all printed seasonal components even when a blank might mean a non-growing season.
    seasonal = {
        "Rice": {"Autumn", "Kharif", "Summer", "Winter"},
        "Jowar": {"Kharif", "Rabi", "Summer"},
    }
    for crop_name, seasons in seasonal.items():
        part = src[src.crop_name.eq(crop_name)]
        if not set(part.season.dropna()) <= seasons or part.season.isna().any():
            raise ValueError(f"Unrecognized seasonal coverage for {crop_name}")
    if src.loc[~src.crop_name.isin(seasonal), "season"].notna().any():
        raise ValueError("Unexpected seasonal rows for an annual DES commodity")
    groups = src.groupby(KEY, as_index=False)
    result = groups.agg(
        reference=("area", lambda v: v.sum(min_count=len(v))),
        n_components=("area", "size"),
        n_observed=("area", "count"),
    )
    expected = result.crop.map({"RICE": 4, "SORGHUM": 3}).fillna(1)
    result["reference"] = result.reference.where(result.n_components.eq(expected))
    return result, coverage


def paired_data(crop, match, irrigation_path):
    units = match[match.included]
    official = crop.merge(units[["dist_code", "old_code"]], on="dist_code", validate="m:1")
    output = []
    for label, name, mapping in [
        ("Presented DES", "crops", DES_NAMES),
        ("Presented five crops", "crops_ext5", FIVE_NAMES),
    ]:
        other = pd.read_parquet(CLEAN / f"{name}.parquet").rename(columns={"dist_code": "old_code"})
        other = other[other.crop.isin(mapping)].copy()
        other["crop"] = other.crop.map(mapping)
        both = official.merge(
            other[["old_code", "crop", "year", "area_ha", "prod_t"]],
            on=["old_code", "crop", "year"],
            suffixes=("", "_reference"),
            validate="1:1",
        )
        for field, current, reference in [
            ("crop area (ha)", "area_ha_reported", "area_ha_reference"),
            ("production (t)", "prod_t", "prod_t_reference"),
        ]:
            use = both
            if current == "prod_t":
                # DLD sugarcane is gur. The DES extract also lacks a row-level bale/tonne
                # indicator for cotton lint. Compare the corresponding areas only.
                use = use[~use.crop.isin(["SUGARCANE", "COTTON"])]
            d = use[KEY + [current, reference]].rename(
                columns={current: "archived", reference: "reference"}
            )
            output.append(d.assign(source=label, measure=field))
    external, coverage = read_irrigation(irrigation_path, units)
    both = official.merge(external, on=KEY, validate="1:1")
    both["archived"] = both.irrigated_ha.mask(both.irrigation_manual_conflict)
    output.append(
        both[KEY + ["archived", "reference", "n_components", "n_observed"]].assign(
            source="DES land use / India Data Portal", measure="irrigated area (ha)"
        )
    )
    out = (
        pd.concat(output, ignore_index=True)
        .sort_values(["source", "measure", *KEY])
        .reset_index(drop=True)
    )
    out["difference"] = out.archived - out.reference
    if out.duplicated(["source", "measure", *KEY]).any():
        raise ValueError("Repeated paired physical measurement")
    return out, coverage


def agreement(pairs):
    rows = []
    for (source, measure), d in pairs.groupby(["source", "measure"], sort=True):
        for crop, sub in [("All comparable crops", d), *d.groupby("crop", sort=True)]:
            observed = sub.dropna(subset=["archived", "reference"])
            positive = observed[(observed.archived > 0) | (observed.reference > 0)]
            delta = positive.difference.abs()
            rows.append(
                {
                    "source": source,
                    "measure": measure,
                    "crop": crop,
                    "candidate_rows": len(sub),
                    "observed_pairs": len(observed),
                    "both_zero": len(observed) - len(positive),
                    "positive_pairs": len(positive),
                    "districts": positive.dist_code.nunique(),
                    "first_year": positive.year.min(),
                    "last_year": positive.year.max(),
                    "exactly_equal": int(positive.difference.eq(0).sum()),
                    "within_tolerance": int(
                        delta.le(np.maximum(10, positive.reference * 0.01)).sum()
                    ),
                    "median_absolute_difference": delta.median(),
                    "mean_absolute_difference": delta.mean(),
                    "max_absolute_difference": delta.max(),
                }
            )
    return pd.DataFrame(rows)


def plot_agreement(stats):
    d = stats[stats.crop.eq("All comparable crops") & stats.positive_pairs.gt(0)].copy()
    d["percent"] = 100 * d.within_tolerance / d.positive_pairs
    labels = {
        "Presented DES": "Presented DES",
        "Presented five crops": "Presented five crops",
        "DES land use / India Data Portal": "DES land use",
    }
    fig, ax = plt.subplots(figsize=(8, 4.1))
    ys = np.arange(len(d))
    ax.barh(
        ys, d.percent, color=["#247A78" if m != "production (t)" else "#81613D" for m in d.measure]
    )
    for y, r in enumerate(d.itertuples()):
        ax.text(
            r.percent - 1.5,
            y,
            f"{r.percent:.2f}%",
            color="white",
            ha="right",
            va="center",
        )
    ax.set(
        yticks=ys,
        yticklabels=[f"{labels[s]} · {m}" for s, m in zip(d.source, d.measure)],
        xlim=(0, 105),
        xlabel="Paired measurements within tolerance (%)",
        title="Agreement on aligned district boundaries",
    )
    ax.invert_yaxis()
    fig.subplots_adjust(left=0.35, bottom=0.23, top=0.87)
    finish(
        fig,
        "05b_district_quantity_agreement",
        "Tolerance: 10 ha or tonnes, or 1% of the comparator, whichever is larger. Both-zero pairs excluded.",
    )


def plot_prices(stats):
    d = (
        stats[stats.term.eq("spei_z")]
        .set_index("model")
        .loc[["presented_fao", "district_2014", "district_2015", "district_2016"]]
    )
    fig, ax = plt.subplots(figsize=(8, 3.9))
    y = np.arange(len(d))
    x = d.beta.to_numpy() * 100
    ax.errorbar(
        x, y, xerr=[x - 100 * d.ci_lo, 100 * d.ci_hi - x], fmt="o", capsize=4, color="#247A78"
    )
    ax.axvline(0, color="#858585", lw=1)
    ax.invert_yaxis()
    ax.set(
        yticks=y,
        yticklabels=[
            "Presented FAOSTAT prices",
            "2014 district prices",
            "2015 district prices",
            "2016 district prices",
        ],
        xlabel="Output difference (% of trend) per unit SPEI × one SD of lagged diversity",
        title="How much does the choice of crop prices change the estimate?",
    )
    fig.subplots_adjust(left=0.31, bottom=0.23, top=0.85)
    finish(
        fig,
        "05c_district_price_comparison",
        f"Same {int(d.n_districts.iloc[0])} districts, {int(d.n_obs.iloc[0]):,} district-years and crop baskets • 95% district-clustered intervals",
    )
