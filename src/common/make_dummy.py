"""Write SYNTHETIC versions of every contract file to data/dummy/ and outputs/dummy/.

Nothing here is real data. Country codes are real so that code written against the
dummy files runs unchanged on the real ones, but every number is generated from a
fixed seed. Never writes to data/clean/ or outputs/ proper.

Run: python -m src.common.make_dummy
"""
import json

import numpy as np
import pandas as pd

from . import paths
from .contract import CONTRACT, YEAR_MAX, YEAR_MIN, read, summary, validate, validate_geojson

SEED = 1993
YEARS = np.arange(YEAR_MIN, YEAR_MAX + 1)

# iso3, name, region, member_iso3, lon, lat, half-width of the dummy rectangle (degrees)
COUNTRIES = [
    ("IND", "India", "South Asia", "", 79, 22, 5),
    ("PAK", "Pakistan", "South Asia", "", 69, 30, 3),
    ("USA", "United States of America", "Northern America", "", -98, 39, 6),
    ("CAN", "Canada", "Northern America", "", -105, 56, 6),
    ("AUS", "Australia", "Oceania", "", 134, -25, 6),
    ("BRA", "Brazil", "Latin America", "", -52, -10, 6),
    ("ARG", "Argentina", "Latin America", "", -64, -34, 4),
    ("MEX", "Mexico", "Latin America", "", -102, 23, 3),
    ("JAM", "Jamaica", "Latin America", "", -77.3, 18.1, 0.5),
    ("CHN", "China", "East and South-East Asia", "", 104, 35, 5),
    ("MNG", "Mongolia", "East and South-East Asia", "", 104, 46, 3),
    ("IDN", "Indonesia", "East and South-East Asia", "", 114, -1, 4),
    ("THA", "Thailand", "East and South-East Asia", "", 101, 15, 3),
    ("RUS", "Russian Federation", "Europe", "", 60, 58, 6),
    ("UKR", "Ukraine", "Europe", "", 32, 49, 2.5),
    ("FRA", "France", "Europe", "", 2, 46.5, 2.5),
    ("DEU", "Germany", "Europe", "", 10, 51, 2.5),
    ("ESP", "Spain", "Europe", "", -4, 40, 2.5),
    ("POL", "Poland", "Europe", "", 19, 52, 2.5),
    ("BLX", "Belgium-Luxembourg", "Europe", "BEL;LUX", 4.5, 50.5, 1),
    ("SCG", "Serbia and Montenegro", "Europe", "SRB;MNE", 20.5, 44, 1.5),
    ("ETH", "Ethiopia", "Sub-Saharan Africa", "", 40, 9, 3),
    ("KEN", "Kenya", "Sub-Saharan Africa", "", 38, 0, 3),
    ("SOM", "Somalia", "Sub-Saharan Africa", "", 46, 5, 2),
    ("ZAF", "South Africa", "Sub-Saharan Africa", "", 25, -29, 3),
    ("NGA", "Nigeria", "Sub-Saharan Africa", "", 8, 9, 3),
    ("SDX", "Sudan (former)", "Sub-Saharan Africa", "SDN;SSD", 30, 13, 4),
    ("EGY", "Egypt", "North Africa and West Asia", "", 30, 27, 3),
    ("LBY", "Libya", "North Africa and West Asia", "", 17, 27, 3),
    ("TUR", "Türkiye", "North Africa and West Asia", "", 35, 39, 2.5),
]

# item_code, item, typical yield (t/ha), price (constant I$/t). Magnitudes only.
CROP_POOL = [
    (15, "Wheat", 3.0, 250), (27, "Rice", 4.0, 400), (56, "Maize (corn)", 4.5, 200),
    (44, "Barley", 2.8, 200), (83, "Sorghum", 1.5, 220), (79, "Millet", 1.0, 300),
    (116, "Potatoes", 18.0, 250), (125, "Cassava, fresh", 11.0, 150), (236, "Soya beans", 2.5, 450),
    (242, "Groundnuts, excluding shelled", 1.6, 900), (156, "Sugar cane", 65.0, 35),
    (270, "Rape or colza seed", 2.0, 500), (267, "Sunflower seed", 1.7, 550),
    (328, "Seed cotton, unginned", 2.0, 700), (176, "Beans, dry", 0.9, 800), (191, "Chick peas, dry", 1.0, 750),
]

# Countries built to fail one inclusion rule each, so the `included` filter is exercised.
SMALL_AREA = "JAM"      # mean harvested area < 100,000 ha
FEW_CROPS = "MNG"       # < 5 crops
MOSTLY_IMPUTED = "LBY"  # < 25 non-imputed years


def make_countries():
    return pd.DataFrame([c[:4] for c in COUNTRIES], columns=["iso3", "name", "region", "member_iso3"])


def make_geojson():
    features = []
    for iso3, name, _, _, lon, lat, h in COUNTRIES:
        ring = [[lon - h, lat - h], [lon + h, lat - h], [lon + h, lat + h], [lon - h, lat + h], [lon - h, lat - h]]
        features.append({
            "type": "Feature",
            "properties": {"iso3": iso3},
            "geometry": {"type": "Polygon", "coordinates": [ring]},
        })
    return {"type": "FeatureCollection", "features": features}


def make_climate(rng, countries):
    # Regional factor gives neighbours shared drought years.
    regional = {r: rng.normal(size=len(YEARS)) for r in countries["region"].unique()}
    rows = []
    for iso3, region in zip(countries["iso3"], countries["region"]):
        w = np.clip(0.6 * regional[region] + 0.8 * rng.normal(size=len(YEARS)), -3, 3)
        unw = np.clip(w + rng.normal(0, 0.25, len(YEARS)), -3, 3)
        s6 = np.clip(np.minimum(w, 0) - np.abs(rng.normal(0.6, 0.3, len(YEARS))), -3, 3)
        rows.append(pd.DataFrame({"iso3": iso3, "year": YEARS, "spei12_w": w, "spei12_unw": unw, "spei6min_w": s6}))
    return pd.concat(rows, ignore_index=True)


def make_events(climate):
    """Event rule from CLAUDE.md, applied to the dummy climate (the real one is Prathmesh's)."""
    rows = []
    for iso3, g in climate.groupby("iso3", sort=False):
        s = g.set_index("year")["spei12_w"]
        drought = s <= -1.0
        for y in s.index[drought]:
            if y - 3 < YEAR_MIN or y + 3 > YEAR_MAX:
                continue
            if drought.loc[y - 3:y - 1].any():  # also rules out later years of a run
                continue
            v = s.loc[y]
            severity = "extreme" if v <= -2.0 else "severe" if v <= -1.5 else "moderate"
            rows.append((iso3, int(y), v, severity))
    return pd.DataFrame(rows, columns=["iso3", "year", "spei12_w", "severity"])


def make_crops(rng, countries, climate):
    t = YEARS - YEAR_MIN
    rows = []
    for iso3 in countries["iso3"]:
        spei = climate.loc[climate["iso3"] == iso3, "spei12_w"].to_numpy()
        n = 4 if iso3 == FEW_CROPS else 8
        picks = rng.choice(len(CROP_POOL), size=n, replace=False)
        # Low concentration -> one or two dominant crops; high -> even shares.
        shares = rng.dirichlet(np.full(n, rng.uniform(0.3, 5.0)))
        shares = np.maximum(shares, 0.01) / np.maximum(shares, 0.01).sum()
        total_area = 60_000 if iso3 == SMALL_AREA else np.exp(rng.uniform(np.log(5e5), np.log(1e8)))
        for share, p in zip(shares, picks):
            code, item, y0, price = CROP_POOL[p]
            beta = rng.normal(0.06, 0.05)  # yield response to SPEI; differs by crop
            area = total_area * share * (1 + rng.uniform(-0.01, 0.01)) ** t * (1 + rng.normal(0, 0.03, len(t)))
            yld = y0 * rng.uniform(0.6, 1.4) * (1 + rng.uniform(0.0, 0.02)) ** t
            yld = yld * np.clip(1 + beta * spei + rng.normal(0, 0.05, len(t)), 0.2, None)
            prod = area * yld

            imputed = np.zeros(len(t), dtype=bool)
            if iso3 == MOSTLY_IMPUTED:
                imputed[:12] = True
            elif rng.random() < 0.4:  # a block of imputed years in some series, ~10% of rows overall
                start = rng.integers(0, len(t) - 8)
                imputed[start:start + rng.integers(4, 9)] = True
            rows.append(pd.DataFrame({
                "iso3": iso3, "year": YEARS, "item_code": code, "item": item,
                "area_ha": area, "prod_t": prod, "yield_t_ha": prod / area,
                "value_const": prod * price, "is_imputed": imputed,
            }))
    return pd.concat(rows, ignore_index=True)


def make_controls(rng, countries):
    t = YEARS - YEAR_MIN
    rows = []
    for iso3 in countries["iso3"]:
        gdp = np.exp(rng.uniform(np.log(500), np.log(50_000))) * (1 + rng.uniform(0.0, 0.04)) ** t
        fert = rng.uniform(5, 250) * (1 + rng.uniform(-0.01, 0.03)) ** t * (1 + rng.normal(0, 0.05, len(t)))
        irrig = np.clip(rng.uniform(0.01, 0.6) * (1 + rng.uniform(0.0, 0.01)) ** t, 0, 1)
        rows.append(pd.DataFrame({"iso3": iso3, "year": YEARS, "gdp_pc": gdp, "fert_kg_ha": fert, "irrig_share": irrig}))
    return pd.concat(rows, ignore_index=True)


def make_quality(crops):
    """Inclusion rules from CLAUDE.md. A year counts as imputed if most of its value is imputed."""
    rows = []
    for iso3, g in crops.groupby("iso3", sort=False):
        n_crops = g["item_code"].nunique()
        mean_area = g.groupby("year")["area_ha"].sum().mean()
        share_imputed = g.loc[g["is_imputed"], "value_const"].sum() / g["value_const"].sum()
        by_year = g.assign(imp_value=g["value_const"] * g["is_imputed"]).groupby("year")[["imp_value", "value_const"]].sum()
        good_years = int((by_year["imp_value"] / by_year["value_const"] <= 0.5).sum())
        reasons = []
        if mean_area < 100_000:
            reasons.append("mean harvested area < 100,000 ha")
        if n_crops < 5:
            reasons.append("fewer than 5 crops")
        if good_years < 25:
            reasons.append("fewer than 25 non-imputed years")
        rows.append((iso3, n_crops, share_imputed, not reasons, "; ".join(reasons)))
    return pd.DataFrame(rows, columns=["iso3", "n_crops", "share_imputed", "included", "reason"])


def make_crop_anomalies(crops, climate, included):
    """Placeholder: linear detrend instead of LOWESS. src.metrics.run overwrites this file."""
    df = crops[crops["iso3"].isin(included)].merge(climate[["iso3", "year", "spei12_w"]], on=["iso3", "year"])
    total_value = df.groupby("iso3")["value_const"].transform("sum")
    rows = []
    for (iso3, item), g in df.groupby(["iso3", "item"], sort=False):
        trend = np.polyval(np.polyfit(g["year"], g["yield_t_ha"], 1), g["year"])
        anom = (g["yield_t_ha"].to_numpy() - trend) / trend
        beta = np.polyfit(g["spei12_w"], anom, 1)[0]
        share = g["value_const"].sum() / total_value.loc[g.index[0]]
        rows.append(pd.DataFrame({"iso3": iso3, "year": g["year"].to_numpy(), "item": item,
                                  "yield_anom": anom, "value_share": share, "resp_beta": beta}))
    return pd.concat(rows, ignore_index=True)


def make_metrics(rng, crops, events, included):
    """Placeholder: hill1 is computed from the dummy areas, every other column is random.
    src.metrics.run overwrites this file."""
    rows = []
    n_events = events.groupby("iso3").size()
    for iso3 in included:
        area = crops[crops["iso3"] == iso3].groupby("item_code")["area_ha"].mean()
        p = area / area.sum()
        n_ev = int(n_events.get(iso3, 0))
        has_ev = n_ev > 0
        rows.append({
            "iso3": iso3,
            "hill1": float(np.exp(-(p * np.log(p)).sum())),
            "stability": rng.uniform(5, 30),
            "resistance": rng.uniform(0.8, 1.05) if has_ev else np.nan,
            "recovery": rng.uniform(0.95, 1.2) if has_ev else np.nan,
            "vulnerability": rng.uniform(0.0, 0.25),
            "vuln_drought": rng.uniform(0.0, 0.6),
            "n_events": n_ev,
            "resp_div": rng.uniform(0.0, 0.08),
            "resp_divergence": rng.uniform(0.0, 1.0),
            "phi_sync": rng.uniform(0.15, 0.9),
            "mean_crop_cv": rng.uniform(0.04, 0.25),
        })
    return pd.DataFrame(rows)


DUMMY_README = """# Dummy data

Everything in this folder is SYNTHETIC, generated by `python -m src.common.make_dummy`
(fixed seed). Country codes are real; every number is made up. Do not use for results.
"""


def main():
    rng = np.random.default_rng(SEED)
    clean, out = paths.CLEAN_DUMMY, paths.OUT_DUMMY
    clean.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    countries = make_countries()
    climate = make_climate(rng, countries)
    events = make_events(climate)
    crops = make_crops(rng, countries, climate)
    controls = make_controls(rng, countries)
    quality = make_quality(crops)
    included = quality.loc[quality["included"], "iso3"].tolist()
    tables = {
        "countries": countries, "crops": crops, "controls": controls, "quality": quality,
        "climate": climate, "events": events,
        "metrics": make_metrics(rng, crops, events, included),
        "crop_anomalies": make_crop_anomalies(crops, climate, included),
    }

    files = {}
    for name, df in tables.items():
        validate(df, name)
        real = CONTRACT[name]["path"]
        path = (out if paths.OUT in real.parents else clean) / real.name
        if path.suffix == ".parquet":
            df.to_parquet(path, index=False)
        else:
            df.to_csv(path, index=False)
        files[name] = path
    geojson_path = clean / "countries.geojson"
    geojson_path.write_text(json.dumps(make_geojson()))
    (clean / "README.md").write_text(DUMMY_README)
    (out / "README.md").write_text(DUMMY_README)

    # Assertions on what was actually written (round trip through disk).
    print("Dummy files (synthetic):")
    for name, path in files.items():
        assert paths.CLEAN_REAL not in path.parents and path.parent != paths.OUT_REAL, path
        df = validate(read(name, path), name)
        assert len(df) == len(tables[name]), f"{name}: row count changed on round trip"
        print(f"  {summary(df, name)}  -> {path.relative_to(paths.ROOT)}")
    iso_geo = validate_geojson(geojson_path)
    assert set(iso_geo) == set(countries["iso3"]), "geojson iso3 do not match countries.csv"
    print(f"  {'geojson':<15} features={len(iso_geo)}  -> {geojson_path.relative_to(paths.ROOT)}")

    assert len(countries) == 30
    assert set(crops["iso3"]) == set(climate["iso3"]) == set(controls["iso3"]) == set(countries["iso3"])
    assert np.allclose(crops["yield_t_ha"], crops["prod_t"] / crops["area_ha"])
    excluded = quality[~quality["included"]]
    assert set(excluded["iso3"]) == {SMALL_AREA, FEW_CROPS, MOSTLY_IMPUTED}, excluded

    print(f"\ncountries: {len(countries)} ({len(included)} included, {len(excluded)} excluded: "
          + ", ".join(f"{r.iso3} [{r.reason}]" for r in excluded.itertuples()) + ")")
    print(f"crops per country (median): {int(quality['n_crops'].median())}")
    print(f"rows imputed: {100 * crops['is_imputed'].mean():.1f}%")
    print(f"drought years (spei12_w <= -1): {int((climate['spei12_w'] <= -1).sum())}, "
          f"events: {len(events)} in {events['iso3'].nunique()} countries "
          f"({events['severity'].value_counts().to_dict()})")


if __name__ == "__main__":
    main()
