"""Official ICRISAT apportioned district study. Inputs, estimates and figures remain in future_work."""

import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from urllib.parse import urlsplit

import geopandas as gpd
import netCDF4
import numpy as np
import pandas as pd
import requests

from src.climate.aggregate import coverage_fractions, load_spei, weighted_mean
from src.india import metrics, panel
from src.metrics.panel import residualise

from .figures import finish, plt
from .freeze import digest
from .seasons import indices_for, spei_grid
from .storage import RAW, RESULT, ROOT, spei_path

DOI = "10.21421/D2/XFB1BZ"
PORTAL = "https://data.icrisat.org/dld/"
RECORD = f"https://dataverse.icrisat.org/api/datasets/:persistentId/?persistentId=doi:{DOI}"
LANDING = "https://www.icrisat.org/resources/data/district-level-data"
CORNELL = "https://tci.cornell.edu/data/"
INPUT = RAW / "icrisat"
KEY = ["state_code", "district_code", "year", "crop"]
FIELDS = {"apy": ("area_ha", "prod_t"), "prices": ("price_inr_t",), "irrigation": ("irrigated_ha",)}
UNITS = {
    "area_ha": {"ha": 1, "1000 ha": 1000},
    "irrigated_ha": {"ha": 1, "1000 ha": 1000},
    "prod_t": {"kg": 0.001, "t": 1, "1000 t": 1000},
    "price_inr_t": {"INR/t": 1, "INR/quintal": 10, "INR/kg": 1000},
}


def official(url):
    parts = urlsplit(url)
    host = parts.hostname or ""
    return parts.scheme in {"https", "http"} and (
        host == "icrisat.org" or host.endswith(".icrisat.org") or host == "tci.cornell.edu"
    )


def discovery():
    urls = [PORTAL, PORTAL.replace("https:", "http:") + "src/crops.html", LANDING, RECORD, CORNELL,
            "https://tci.cornell.edu/?projects=district-level-database-for-indian-agriculture-and-allied-sectors"]
    urls.extend("https://data.icrisat.org/dldAPI/apportioned/" + slug for slug in
                ["area-production-yield", "farm-harvest-prices", "cropwise-irrigated-area"])

    def inspect(url):
        try:
            with requests.get(url, timeout=(8, 25)) as r:
                item = {"url": url, "http_status": r.status_code, "final_url": r.url}
                if r.ok and url == RECORD:
                    v = r.json()["data"]["latestVersion"]
                    item.update(
                        version=f"{v['versionNumber']}.{v['versionMinorNumber']}",
                        published=v.get("releaseTime"),
                        listed_files=[f["dataFile"]["filename"] for f in v["files"]],
                    )
                elif r.ok and "/dldAPI/" in url:
                    data = r.json()
                    item.update(description=data["description"], years=data["years"], district_years=len(data["data"]))
                elif r.ok:
                    links = re.findall(r"""href=["']([^"']+)["']""", r.text, flags=re.I)
                    item["download_links"] = [
                        s for s in links if re.search(r"\.(csv|xlsx?|zip|gpkg)(\?|$)", s, re.I)
                    ]
                return item
        except (requests.RequestException, ValueError, KeyError) as e:
            return {"url": url, "error": type(e).__name__}

    with ThreadPoolExecutor(len(urls)) as pool:
        return list(pool.map(inspect, urls))


def release_file(spec, directory):
    url = spec["source_url"]
    if not official(url):
        raise ValueError(f"District release source is outside the publishers’ domains: {url}")
    path = (directory / spec["file"]).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError("District input must be inside its release directory")
    if not path.is_file() or digest(path) != spec["sha256"]:
        raise ValueError(f"Missing file or checksum mismatch: {path.name}")
    return path


def strings(spec, directory):
    path = release_file(spec, directory)
    opts = dict(dtype=str, keep_default_na=False, na_values=spec.get("missing_values", []))
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, **opts)
    if path.suffix.lower() in {".xls", ".xlsx"}:
        return pd.read_excel(path, sheet_name=spec.get("sheet", 0), **opts)
    raise ValueError(f"Unsupported official table format: {path.name}")


def crop_year(label):
    match = re.fullmatch(r"(\d{4})(?:\s*[-–/]\s*(\d{2}|\d{4}))?", str(label).strip())
    if not match:
        raise ValueError(f"Unrecognized agricultural year: {label}")
    year = int(match[1])
    end = match[2]
    if end and end != str(year + 1)[-len(end) :]:
        raise ValueError(f"Mismatched agricultural year: {label}")
    return year


def clean_keys(d):
    d = d.copy()
    for col in ["state_code", "district_code"]:
        d[col] = d[col].astype("string").str.strip()
        if d[col].isna().any() or d[col].eq("").any() or d[col].str.contains("/").any():
            raise ValueError(f"Invalid official geography identifier in {col}")
    return d


def observed(values, field, unit):
    if unit not in UNITS[field]:
        raise ValueError(f"Unrecognized {field} unit: {unit}")
    numeric = (
        pd.to_numeric(values.replace("", np.nan), errors="raise").astype(float) * UNITS[field][unit]
    )
    if np.isinf(numeric).any() or numeric.lt(0).any():
        raise ValueError(f"Invalid value or undeclared missing code in {field}")
    return numeric.where(numeric.gt(0)) if field == "price_inr_t" else numeric


def normalize(role, spec, directory):
    raw = strings(spec, directory)
    keys = clean_keys(pd.DataFrame({key: raw[src] for key, src in spec["keys"].items()}))
    keys["year"] = keys.year.map(crop_year)
    fields = FIELDS[role]
    frames = []
    if "crops" in spec:
        for name, columns in spec["crops"].items():
            values = {
                field: observed(raw[columns[field]["column"]], field, columns[field]["unit"])
                for field in fields
            }
            frames.append(keys.assign(crop=name, **values))
    else:
        names = raw[spec["crop_column"]].astype("string").str.strip()
        if "crop_labels" in spec:
            unknown = set(names.dropna()) - set(spec["crop_labels"])
            if unknown:
                raise ValueError(f"Undeclared crop labels: {sorted(unknown)}")
            names = names.map(spec["crop_labels"])
        values = {
            field: observed(raw[spec["values"][field]["column"]], field, spec["values"][field]["unit"])
            for field in fields
        }
        frames.append(keys.assign(crop=names, **values))
    d = pd.concat(frames, ignore_index=True)
    if d.crop.isna().any() or d.crop.eq("").any() or d.duplicated(KEY).any():
        raise ValueError(f"Missing crop or duplicate crop-year in {role}")
    return d[KEY + list(fields)]


def prepare_crop(apy, prices, irrigation, reference_year, rounded_areas=False):
    out = apy.merge(prices, on=KEY, how="left", validate="1:1").merge(
        irrigation, on=KEY, how="left", validate="1:1"
    )
    rounded = out.area_ha.eq(0) & out.prod_t.gt(0)
    if rounded.any() and not rounded_areas:
        raise ValueError("Positive production on zero reported area")
    out["area_ha_reported"] = out.area_ha
    out["area_below_reporting_precision"] = rounded
    out["irrigation_invalid"] = out.irrigated_ha.gt(out.area_ha_reported)
    out["area_ha"] = out.area_ha.mask(rounded)
    out["irrigation_share"] = (out.irrigated_ha / out.area_ha.where(out.area_ha.gt(0))).mask(
        out.irrigation_invalid
    )
    fixed = (
        prices[prices.year.eq(reference_year)]
        .drop(columns="year")
        .rename(columns={"price_inr_t": "reference_price_inr_t"})
    )
    out = out.merge(fixed, on=[k for k in KEY if k != "year"], how="left", validate="m:1")
    if not out.reference_price_inr_t.notna().any():
        raise ValueError("No district-crop farm prices in the price reference year")
    out["value_const"] = out.prod_t * out.reference_price_inr_t
    out["yield_t_ha"] = out.prod_t / out.area_ha.where(out.area_ha.gt(0))
    out["dist_code"] = out.state_code + "/" + out.district_code
    return out


def boundaries(spec, directory):
    if "poly_id" in spec:
        members = strings(spec, directory)
        d = clean_keys(pd.DataFrame({k: members[v] for k, v in spec["keys"].items()}))
        d["poly_id"] = members[spec["poly_id"]].astype("string").str.strip()
        if d.poly_id.duplicated().any() or d.poly_id.isna().any():
            raise ValueError("Duplicate or missing child polygon")
        source = gpd.read_file(ROOT / "data/raw/india_geo/geoBoundaries-IND-ADM2.geojson")[
            ["shapeID", "geometry"]
        ]
        shape = source.merge(d, left_on="shapeID", right_on="poly_id", validate="1:1")
        if len(shape) != len(d):
            raise ValueError("Official membership uses polygons missing from the named geography")
    else:
        g = gpd.read_file(release_file(spec, directory))
        d = clean_keys(pd.DataFrame({k: g[v] for k, v in spec["keys"].items()}))
        shape = gpd.GeoDataFrame(d, geometry=g.geometry, crs=g.crs)
    if (
        shape.crs is None
        or shape.geometry.isna().any()
        or shape.geometry.is_empty.any()
        or not shape.geometry.is_valid.all()
    ):
        raise ValueError("District boundaries require valid, nonempty and georeferenced polygons")
    shape["dist_code"] = shape.state_code + "/" + shape.district_code
    return shape[["dist_code", "geometry"]].to_crs(4326).dissolve(by="dist_code", as_index=False)


def district_climate(shape, years):
    template = (
        load_spei(12).isel(time=0).rio.write_crs(4326).rio.set_spatial_dims(x_dim="lon", y_dim="lat")
    )
    cover = coverage_fractions(shape.rename(columns={"dist_code": "iso3"}), template)
    crop = np.load(ROOT / "data/raw/derived/cropland_05deg.npy").ravel()
    with netCDF4.Dataset(spei_path(12)) as ds:
        values = spei_grid(ds, indices_for(ds, years, 12))
    rows = []
    for code, (cell, frac) in cover.items():
        weights = crop[cell] * frac
        if not len(cell) or weights.sum() <= 0:
            continue
        spei, coverage = weighted_mean(values[:, cell], weights)
        spei = np.where(coverage >= 0.90, spei, np.nan)
        rows.append(
            pd.DataFrame(dict(dist_code=code, year=years, spei12_w=spei, climate_coverage=coverage))
        )
    if not rows:
        raise ValueError("The mapped districts have no observed cropland climate")
    return pd.concat(rows, ignore_index=True)


def value_sample(crops, years):
    measured = crops[crops.year.isin(years)].dropna(subset=["area_ha"])
    # A crop with no positive priced production anywhere in the window is not a producing crop.
    # Keep its area for diversity, but do not count it towards the three crops in the value index.
    ever_positive = measured.groupby(["dist_code", "crop"]).value_const.transform("max").gt(0)
    return measured.assign(value_const=measured.value_const.where(ever_positive))


def lagged_irrigation(crops, years):
    """Use the same balanced, priced crop basket as output; require irrigation for every positive-area crop."""
    measured = value_sample(crops, years)
    by_crop = measured.groupby(["dist_code", "crop"])
    balanced = by_crop.value_const.transform("count").eq(len(years))
    d = measured.loc[balanced].copy()
    d["missing_irrigation"] = (d.area_ha.gt(0) & d.irrigation_share.isna()) | d.irrigation_invalid
    d["valid_irrigated"] = d.irrigated_ha.where(d.area_ha.gt(0), 0)
    totals = d.groupby(["dist_code", "year"], as_index=False).agg(
        area=("area_ha", "sum"),
        irrigated=("valid_irrigated", "sum"),
        missing=("missing_irrigation", "any"),
    )
    totals["irrig_lag"] = (totals.irrigated / totals.area.where(totals.area.gt(0))).mask(totals.missing)
    return totals.assign(year=totals.year + 1)[["dist_code", "year", "irrig_lag"]]


def models(crop, climate, years):
    usable = value_sample(crop, years)
    # Preserve missing acreage as a missing crop/year in the legacy aggregation.
    m, p, _, excluded = metrics.run(usable, climate, years, "spei12_w")
    p = p.merge(
        crop[["dist_code", "state_code"]]
        .drop_duplicates()
        .rename(columns={"state_code": "state_name"}),
        on="dist_code",
        validate="m:1",
    )
    p = p.merge(lagged_irrigation(crop, years), on=["dist_code", "year"], how="left", validate="1:1")
    prep = panel.prep(p)
    primary = ["spei", "spei_z", "z"]
    estimates = []
    controlled = prep.dropna(subset=["irrig_lag"]).copy()
    controlled["spei_irrig_lag"] = controlled.spei * controlled.irrig_lag
    for name, frame, terms in [
        ("official", prep, primary),
        ("official_irrigation_sample", controlled, primary),
        ("official_crop_irrigation", controlled, primary + ["irrig_lag", "spei_irrig_lag"]),
    ]:
        if frame.dist_code.nunique() < 2 or len(frame) <= len(terms):
            raise ValueError(f"Insufficient observations for {name}")
        design = residualise(frame, terms, fe=panel.FE)
        if np.linalg.matrix_rank(design) < len(terms):
            raise ValueError(f"Unidentified terms in {name}")
        rows, _ = panel.fit_rows(name, frame, terms)
        estimates.extend(
            dict(r, start_year=years[0], end_year=years[-1], basis="observed district farm prices")
            for r in rows
        )
    return m, p, excluded, pd.DataFrame(estimates)


def presented_benchmarks():
    base = pd.read_csv(ROOT / "outputs/stats/india_panel.csv")
    references = []
    for model, start, end in [("core:core", 1998, 2022), ("ext5:core", 1993, 2015)]:
        references.append(
            base.loc[base.model.eq(model)].assign(
                start_year=start, end_year=end, basis="presented international constant prices"
            )
        )
    return pd.concat(references, ignore_index=True)


def comparable_quantities(crop, shape, manifest, directory):
    """Compare only certified common boundaries, commodities and physical units on complete child coverage."""
    codebook = manifest.get("comparison_crops", {})
    geo = manifest["geography"]
    dm = pd.read_csv(
        ROOT / "data/clean/india/district_match.csv", dtype={"dist_code": str, "poly_id": str}
    ).query("matched")
    if "poly_id" in geo:
        member = strings(geo, directory)
        overlay = clean_keys(pd.DataFrame({k: member[v] for k, v in geo["keys"].items()}))
        overlay["dist_code"] = overlay.state_code + "/" + overlay.district_code
        overlay["poly_id"] = member[geo["poly_id"]].str.strip()
        overlay = overlay.merge(
            dm[["poly_id", "dist_code"]].rename(columns={"dist_code": "old_code"}),
            on="poly_id",
            how="left",
            validate="1:1",
        )
        overlay = overlay[
            overlay.groupby("dist_code")
            .old_code.transform("count")
            .eq(overlay.groupby("dist_code").old_code.transform("size"))
        ]
    else:
        old = (
            gpd.read_file(ROOT / "data/clean/india/districts.geojson")
            .to_crs(4326)
            .rename(columns={"dist_code": "old_code"})
        )
        joined = shape.sjoin(old, predicate="intersects")
        same = [
            geom.equals(old.geometry.loc[j]) for geom, j in zip(joined.geometry, joined.index_right)
        ]
        overlay = joined.loc[same, ["dist_code", "old_code"]]
    measures = ["area_ha", "prod_t"]
    keys = ["dist_code", "crop", "year"]
    outs = []
    for source, name in [
        ("DES", "crops.parquet"),
        ("five crops", "crops_ext5.parquet"),
    ]:
        labels = codebook.get(source, {})
        if not labels or overlay.empty:
            continue
        if len(set(labels.values())) != len(labels):
            raise ValueError("Comparison crop correspondence must be one-to-one")
        original = pd.read_parquet(ROOT / "data/clean/india" / name).rename(
            columns={"dist_code": "old_code"}
        )
        if set(labels) - set(original.crop) or set(labels.values()) - set(crop.crop):
            raise ValueError("Unknown comparison commodity")
        original = original[original.crop.isin(labels)].copy()
        original["crop"] = original.crop.map(labels)
        original = original.merge(overlay[["dist_code", "old_code"]], on="old_code", validate="m:1")
        groups = original.groupby(keys, as_index=False)
        summed = groups[measures].sum(min_count=1)
        counts = groups[measures].count().set_index(keys)
        need = summed.dist_code.map(overlay.groupby("dist_code").size()).to_numpy()
        for field in measures:
            summed[field] = summed[field].where(counts[field].to_numpy() == need)
        pair = crop[keys + measures].merge(
            summed, on=keys, suffixes=("_official", "_presented"), validate="1:1"
        )
        for field in measures:
            pair[field + "_difference"] = pair[field + "_official"] - pair[field + "_presented"]
        outs.append(pair.assign(presented_source=source))
    cols = (
        keys
        + [v + z for v in measures for z in ["_official", "_presented", "_difference"]]
        + ["presented_source"]
    )
    return pd.concat(outs, ignore_index=True)[cols] if outs else pd.DataFrame(columns=cols)


def analyse(manifest, directory, destination=None):
    destination = RESULT if destination is None else destination
    if manifest.get("series") != "apportioned":
        raise ValueError("Only the official apportioned district series is supported")
    evidence = release_file(manifest["evidence"], directory).read_text()
    if (
        not manifest["release_id"].strip()
        or not manifest["boundary_vintage"].strip()
        or not manifest["latest_evidence_quote"].strip()
    ):
        raise ValueError("Document the publisher release and boundary vintage")
    if manifest["release_id"] not in evidence or manifest["latest_evidence_quote"] not in evidence:
        raise ValueError(
            "Release identity and latest-release evidence must occur in the saved publisher record"
        )
    for role in [*FIELDS, "geography"]:
        spec = manifest[role]
        if spec["boundary_vintage"] != manifest["boundary_vintage"]:
            raise ValueError(f"Boundary vintage mismatch for {role}")
    data = {role: normalize(role, manifest[role], directory) for role in FIELDS}
    crop = prepare_crop(**data, reference_year=manifest["price_reference_year"])
    shape = boundaries(manifest["geography"], directory)
    first, last = manifest["analysis_years"]
    years = list(range(first, last + 1))
    if not (1901 <= first < last <= 2023 and len(years) >= 15 and set(years) <= set(crop.year)):
        raise ValueError("Observed crop and climate years must cover the requested study window")
    missing_geography = set(crop.dist_code) - set(shape.dist_code)
    climate = district_climate(shape, years)
    if set(crop.dist_code).isdisjoint(climate.dist_code):
        raise ValueError("No release district is present in the climate aggregation")
    m, p, dropped, stats = models(crop, climate, years)
    summary = pd.concat([presented_benchmarks(), stats], ignore_index=True)
    matched = comparable_quantities(crop, shape, manifest, directory)
    matched.to_parquet(destination / "district_paired_quantities.parquet", index=False)
    if missing_geography:
        missing = pd.DataFrame(
            {"dist_code": sorted(missing_geography), "reason": "missing release-consistent boundary"}
        )
        dropped = pd.concat(
            [missing, dropped[~dropped.dist_code.isin(missing_geography)]], ignore_index=True
        )
    crop.to_parquet(destination / "district_crop_year.parquet", index=False)
    climate.to_parquet(destination / "district_climate.parquet", index=False)
    m.to_csv(destination / "district_metrics.csv", index=False)
    p.to_parquet(destination / "district_panel.parquet", index=False)
    dropped.to_csv(destination / "district_exclusions.csv", index=False)
    summary.to_csv(destination / "district_comparison.csv", index=False)
    return {
        "status": "complete",
        "release_id": manifest["release_id"],
        "boundary_vintage": manifest["boundary_vintage"],
        "paired_quantity_rows": len(matched),
        "source_files": {role: manifest[role] for role in [*FIELDS, "geography", "evidence"]},
        "summary": f"The official {manifest['release_id']} release yields {len(m)} eligible districts in {first}–{last}. The fixed price for each district and commodity is its observed {manifest['price_reference_year']} farm-harvest price. The comparison table reports the presented samples and the new sample separately.",
    }


@plt.style.context("default")
def render_if_ready():
    status = RESULT / "district_status.json"
    if not status.exists():
        return
    state = json.loads(status.read_text())["status"]
    if state not in {"complete", "archived_complete"}:
        return
    d = pd.read_csv(RESULT / "district_comparison.csv").query('term=="spei_z"').copy()
    labels = {
        "core:core": "Presented DES",
        "ext5:core": "Presented five crops",
        "official": "Official release",
        "official_irrigation_sample": "Official, irrigation sample",
        "official_crop_irrigation": "Official, irrigation adjusted",
        "archive": "ICRISAT 1966–2017",
        "archive_irrigation_sample": "1966–2017 · irrigation sample",
        "archive_crop_irrigation": "1966–2017 · irrigation adjusted",
        "overlap": "ICRISAT 1998–2017",
        "overlap_irrigation_sample": "1998–2017 · irrigation sample",
        "overlap_crop_irrigation": "1998–2017 · irrigation adjusted",
    }
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    y = np.arange(len(d))
    x = 100 * d.beta.to_numpy()
    ax.errorbar(x, y, xerr=[x - 100 * d.ci_lo, 100 * d.ci_hi - x], fmt="o", color="#007C83", capsize=4)
    ax.axvline(0, color="#858585", lw=1)
    ax.invert_yaxis()
    ax.set(
        yticks=y,
        yticklabels=d.model.map(labels),
        xlabel="Output difference (% of trend) per unit SPEI × one SD of lagged diversity",
        title="District diversity and drought sensitivity",
    )
    fig.subplots_adjust(left=0.34, right=0.96, bottom=0.22, top=0.87)
    finish(
        fig, "05_district_comparison", "Separate sampling windows • 95% district-clustered intervals"
    )
    if state == "archived_complete":
        from .district_comparison import plot_agreement, plot_prices
        plot_agreement(pd.read_csv(RESULT / "district_quantity_agreement.csv"))
        plot_prices(pd.read_csv(RESULT / "district_price_sensitivity.csv"))


def run():
    path = INPUT / "release.json"
    status_file = RESULT / "district_status.json"
    presented_benchmarks().to_csv(RESULT / "district_presented_benchmarks.csv", index=False)
    checked = discovery()
    saved = json.loads(status_file.read_text()) if status_file.exists() else {}
    if not path.exists() and saved.get("status") == "complete":
        raise FileNotFoundError(
            "The saved district results require their registered release to refresh"
        )
    scratch = RESULT.parent / "local"
    scratch.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix="district-", dir=scratch) as tmpdir:
        staging = Path(tmpdir)
        if path.exists():
            status = analyse(json.loads(path.read_text()), INPUT, staging)
            # Preserve the archived edition and publish a registered edition under distinct names.
            prefix = "registered_" if saved.get("status") == "archived_complete" else ""
        else:
            from .district_archive import analyse_archived
            status = analyse_archived(staging)
            prefix = ""
        for output in staging.iterdir():
            output.replace(RESULT / (prefix + output.name))
    if prefix:
        status_file = RESULT / "registered_district_status.json"
    status.update(
        registry_doi=DOI,
        partner_catalog=CORNELL,
        checked_utc=datetime.now(timezone.utc).isoformat(),
        checks=checked,
    )
    tmp = status_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(status_file)
    print(status["summary"])
    return status["status"] in {"complete", "archived_complete"}


if __name__ == "__main__":
    raise SystemExit(0 if run() else 2)
