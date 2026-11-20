"""Crop-calendar footprints, seasonal drought and matched comparisons. Author: Pmjarvis."""

from zipfile import ZipFile

import geopandas as gpd
import netCDF4
import numpy as np
import pandas as pd
import rasterio
import rioxarray  # noqa: F401 -- registers the raster accessor
from scipy.sparse import csr_matrix

from src.climate.aggregate import MIN_COVERAGE, coverage_fractions, load_spei
from src.metrics import faces, panel, response
from src.metrics.crop_classes import ITEMS
from src.metrics.events import find_events

from .storage import CACHE, RAW, RESULT, ROOT, calendars, spei_path

KEY = ["iso3", "mapspam_crop"]
YEARS = np.arange(1993, 2024)
MONTH_END = np.cumsum([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
THRESHOLD = -1.0
CROPS = {
    "WHEA": ("wwh", "swh"),
    "RICE": ("ri1", "ri2"),
    "MAIZ": ("mai",),
    "SOYB": ("soy",),
    "BARL": ("bar",),
    "MILL": ("mil",),
    "PMIL": ("mil",),
    "SORG": ("sor",),
    "BEAN": ("bea",),
    "SUGB": ("sgb",),
    "SUGC": ("sgc",),
    "CASS": ("cas",),
    "SUNF": ("sun",),
    "RAPE": ("rap",),
    "GROU": ("nut",),
    "COTT": ("cot",),
    "POTA": ("pot",),
}
MEASURES = ["december12", "harvest6", "season"]


def grid_cells():
    path = CACHE / "country_cells_presented.parquet"
    if path.exists():
        return pd.read_parquet(path)
    # The frozen national and crop aggregations intersect polygons with this valid-data mask.
    template = load_spei(12).isel(time=0)
    template = template.rio.write_crs(4326).rio.set_spatial_dims(x_dim="lon", y_dim="lat")
    cover = coverage_fractions(gpd.read_file(ROOT / "data/clean/countries.geojson"), template)
    crop = np.load(ROOT / "data/raw/derived/cropland_05deg.npy").ravel()
    rows = []
    for iso, (cells, frac) in cover.items():
        mask = (crop[cells] > 0) & (frac > 0)
        rows.append(pd.DataFrame({"iso3": iso, "cell": cells[mask], "land_fraction": frac[mask]}))
    frame = pd.concat(rows, ignore_index=True)
    frame.to_parquet(path, index=False)
    return frame


def windows(plant, maturity):
    """Inclusive whole calendar months, ending in the harvest year. A full-year wrap can touch 13."""
    a, b = np.asarray(plant), np.asarray(maturity)
    if not ((a >= 1) & (a <= 365) & (b >= 1) & (b <= 365)).all():
        raise ValueError("Invalid calendar day")
    start = np.searchsorted(MONTH_END, a)
    end = np.searchsorted(MONTH_END, b)
    return end + 1, end - start + 1 + 12 * (a > b)


def gridvar(ds, name):
    a = ds[name][:].astype(float).filled(np.nan)
    lat, lon = ds["lat"][:], ds["lon"][:]
    if lat[0] < lat[-1]:
        a, lat = a[::-1], lat[::-1]
    if not (
        np.allclose(lat, np.arange(89.75, -90, -0.5)) and np.allclose(lon, np.arange(-179.75, 180, 0.5))
    ):
        raise ValueError(f"Unexpected grid for {name}")
    return a.ravel()


def footprints():
    path = CACHE / "footprints_presented.parquet"
    if path.exists():
        return pd.read_parquet(path)
    calendars()
    cells = grid_cells()
    cohort = pd.read_parquet(ROOT / "outputs/metrics.parquet").iso3
    cells = cells.loc[cells.iso3.isin(cohort)].reset_index(drop=True)
    idx = cells.cell.to_numpy()
    spam = next((ROOT / "data/raw/mapspam").glob("*.zip"))
    rows = []
    with ZipFile(spam) as archive, netCDF4.Dataset(next((RAW / "wheat").glob("*.nc4"))) as wheat:
        names = archive.namelist()
        for crop, variants in CROPS.items():
            for regime, tag in [("rf", "R"), ("ir", "I")]:
                name = next(n for n in names if n.endswith(f"_A_{crop}_{tag}.tif"))
                with rasterio.open(f"zip://{spam}!{name}") as ds:
                    if ds.shape != (2160, 4320) or not np.allclose(
                        [ds.bounds.left, ds.bounds.top], [-180, 90]
                    ):
                        raise ValueError("Unexpected MapSPAM grid")
                    a = ds.read(1, masked=True).filled(0)
                    area = (
                        np.where(np.isfinite(a) & (a > 0), a, 0)
                        .reshape(360, 6, 720, 6)
                        .sum(axis=(1, 3), dtype=float)
                        .ravel()
                    )
                weight = area[idx] * cells.land_fraction.to_numpy()
                for variant in variants:
                    with netCDF4.Dataset(
                        next((RAW / "calendar").glob(f"{variant}_{regime}_*.nc4"))
                    ) as ds:
                        plant, end = (gridvar(ds, k)[idx] for k in ("planting_day", "maturity_day"))
                        if crop == "WHEA":
                            parts = [
                                np.nan_to_num(gridvar(wheat, f"{v}_{regime}_area")[idx])
                                for v in variants
                            ]
                            denom = sum(parts)
                            fraction = np.divide(
                                parts[variants.index(variant)],
                                denom,
                                out=np.zeros_like(denom),
                                where=denom > 0,
                            )
                        elif crop == "RICE":
                            fraction = gridvar(ds, "fraction_of_harvested_area")[idx]
                        else:
                            fraction = np.ones(len(idx))
                    w = weight * fraction
                    keep = (w > 0) & np.isfinite(plant) & np.isfinite(end)
                    p, e = plant[keep], end[keep]
                    # Source calendars occasionally contain fractional DOY. Match the enclosing day.
                    p, e = np.ceil(p).astype(int), np.ceil(e).astype(int)
                    month, scale = windows(p, e)
                    rows.append(
                        cells.loc[keep, ["iso3", "cell"]].assign(
                            mapspam_crop=crop,
                            variant=variant,
                            regime=regime,
                            plant=p.astype("int16"),
                            harvest=e.astype("int16"),
                            month=month.astype("int8"),
                            scale=scale.astype("int8"),
                            ha=w[keep],
                        )
                    )
    frame = pd.concat(rows, ignore_index=True)
    frame.to_parquet(path, index=False)
    return frame


def spei_grid(ds, indices):
    if not np.allclose(ds["lat"][:], np.arange(-89.75, 90, 0.5)) or not np.allclose(
        ds["lon"][:], np.arange(-179.75, 180, 0.5)
    ):
        raise ValueError("Unexpected SPEIbase grid")
    data = ds["spei"][indices, :, :].astype(np.float32).filled(np.nan)
    return data[:, ::-1, :].reshape(len(indices), -1)


def indices_for(ds, years, month):
    t = ds["time"]
    dates = netCDF4.num2date(t[:], t.units, calendar=getattr(t, "calendar", "standard"))
    ix = {(d.year, d.month): i for i, d in enumerate(dates)}
    return np.array([ix[(int(y), int(month))] for y in years])


def climate_table():
    foot = footprints()
    groups = foot.groupby(KEY, sort=True).ngroup().to_numpy()
    identifiers = foot[KEY].assign(group=groups).drop_duplicates().sort_values("group")
    size = len(identifiers)
    total = np.zeros((size, len(YEARS)))
    observed = np.zeros_like(total)
    foot = foot.assign(group=groups)
    for scale, byscale in foot.groupby("scale"):
        with netCDF4.Dataset(spei_path(int(scale))) as ds:
            if getattr(ds, "version", None) != "2.10.0":
                raise ValueError(f"Unexpected SPEI version for {scale} months")
            for month, g in byscale.groupby("month"):
                mat = csr_matrix((g.ha, (g.group, g.cell)), shape=(size, 360 * 720))
                vals = spei_grid(ds, indices_for(ds, YEARS, month))
                total += mat @ np.nan_to_num(vals).T
                observed += mat @ np.isfinite(vals).T
        print(f"Aggregated SPEI-{scale}", flush=True)
    base = pd.read_parquet(ROOT / "data/clean/mapspam_area.parquet")[
        KEY + ["area_ha", "irr_ha", "rf_ha"]
    ]
    frame = identifiers.merge(base, on=KEY, validate="1:1").sort_values("group")
    if len(frame) != size:
        raise ValueError("Missing all-regime crop areas")
    # The all-regime raster is rounded independently from its rainfed and irrigated rasters.
    # Use the actual source-regime total as the coverage denominator, preserving missing calendar area.
    frame["regime_area_ha"] = frame.irr_ha + frame.rf_ha
    cov = observed / frame.regime_area_ha.to_numpy()[:, None]
    expandable = frame.loc[frame.index.repeat(len(YEARS)), KEY + ["area_ha", "regime_area_ha"]].copy()
    expandable.insert(len(KEY), "year", np.tile(YEARS, size))
    expandable["observed_ha"] = observed.ravel()
    expandable["coverage"] = cov.ravel()
    expandable["season"] = np.divide(
        total, observed, out=np.full_like(total, np.nan), where=(observed > 0) & (cov >= MIN_COVERAGE)
    ).ravel()
    return expandable.reset_index(drop=True)


def matched(c):
    clean = ROOT / "data/clean"
    dec = pd.read_parquet(clean / "crop_climate.parquet").rename(
        columns={"crop_spei12_w": "december12"}
    )
    old = pd.read_parquet(clean / "crop_climate_season.parquet").rename(
        columns={"crop_spei6_harv_w": "harvest6"}
    )
    keys = KEY + ["year"]
    full = c.merge(dec, on=keys, how="left", validate="1:1").merge(
        old, on=keys, how="left", validate="1:1"
    )
    full["complete"] = full.groupby(KEY)[MEASURES].transform("count").min(axis=1).eq(len(YEARS))
    full.to_parquet(RESULT / "season_crop_year.parquet", index=False)
    common = full[full.complete].copy()
    area = pd.read_parquet(clean / "mapspam_area.parquet")
    bycountry = area.groupby("iso3").area_ha.sum()
    coverage = common.drop_duplicates(KEY).groupby("iso3").area_ha.sum().rename("matched_ha").to_frame()
    coverage["total_ha"] = bycountry
    coverage["share_all_cropland"] = coverage.matched_ha / coverage.total_ha
    covered_years = common.groupby(["iso3", "year"]).observed_ha.sum().groupby("iso3")
    coverage["mean_observed_ha"] = covered_years.mean()
    coverage["min_observed_share_all"] = covered_years.min() / coverage.total_ha
    coverage.to_csv(RESULT / "season_area_coverage.csv")
    scope = area[area.iso3.isin(coverage.index)].merge(
        full.groupby(KEY, as_index=False).agg(
            calendar_complete=("complete", "all"), min_coverage=("coverage", "min")
        ),
        on=KEY,
        how="left",
        validate="1:1",
    )
    scope["status"] = np.select(
        [
            ~scope.mapspam_crop.isin(CROPS),
            scope.min_coverage.isna(),
            scope.min_coverage.lt(MIN_COVERAGE),
            scope.calendar_complete.eq(True),
        ],
        [
            "no specific calendar",
            "no documented area and dates",
            "below 90% in some year",
            "matched three measures",
        ],
        default="missing frozen comparator",
    )
    scope.to_csv(RESULT / "season_crop_scope.csv", index=False)
    annual = (
        common.assign(**{v: common[v] * common.area_ha for v in MEASURES})
        .groupby(["iso3", "year"])[MEASURES + ["area_ha"]]
        .sum()
    )
    annual[MEASURES] = annual[MEASURES].div(annual.area_ha, axis=0)
    return common, annual.reset_index()


def comparisons(common, annual):
    national = pd.read_parquet(ROOT / "outputs/metrics_national.parquet")
    annual = annual.merge(
        national[["iso3", "year", "spei12_w", "index", "anom"]], on=["iso3", "year"], validate="1:1"
    )
    annual.to_csv(RESULT / "season_national_year.csv", index=False)
    out = []
    losses = []
    diversity = pd.read_parquet(ROOT / "outputs/metrics.parquet")[["iso3", "hill1"]]
    for name in ["spei12_w", *MEASURES]:
        climate = annual[["iso3", "year", name]].rename(columns={name: "spei12_w"})
        events = find_events(climate)
        sample = annual.merge(diversity, on="iso3", validate="m:1").reset_index(drop=True)
        sample["dry"] = sample[name].le(THRESHOLD).astype(float)
        design, terms, _ = panel.design(sample, ["dry"], ["hill1"])
        fit = panel.fe_fit(design, "anom", terms)
        losses.extend(
            dict(
                measure=name,
                **r,
                n_obs=len(sample),
                n_countries=fit["n_clusters"],
                n_dry=int(sample.dry.sum()),
            )
            for r in panel.coef_rows(fit)
        )
        for iso, g in annual.groupby("iso3"):
            e = events.loc[events.iso3.eq(iso), "year"]
            index = g.set_index("year")["index"]
            dry = g[name].le(THRESHOLD)
            out.append(
                dict(
                    iso3=iso,
                    measure=name,
                    n_dry=int(dry.sum()),
                    n_events=len(e),
                    switched_vs_matched_dec=int((dry != g.december12.le(THRESHOLD)).sum()),
                    resistance=faces.resistance(index, e),
                    recovery=faces.recovery(index, e),
                    vulnerability_dry=faces.vulnerability(index, g.loc[dry, "year"]),
                    mean_dry_anom=g.loc[dry, "anom"].mean(),
                )
            )
    pd.DataFrame(out).to_csv(RESULT / "season_faces.csv", index=False)
    pd.DataFrame(losses).to_csv(RESULT / "season_loss_panel.csv", index=False)
    # Restrict slopes to the exact same yield-anomaly years for all three climate measures.
    anomalies = pd.read_parquet(ROOT / "outputs/crop_anomalies.parquet")
    items = pd.read_parquet(
        ROOT / "data/clean/crops.parquet", columns=["item_code", "item"]
    ).drop_duplicates()
    merged = anomalies.merge(items, on="item", validate="m:1")
    crop_links = {
        code: list(pair[1]) for code, pair in ITEMS.items() if all(c in CROPS for c in pair[1])
    }
    cap = common.copy()
    for prefix, members in [("millet", ["MILL", "PMIL"])]:
        g = common[common.mapspam_crop.isin(members)].copy()
        # Only use merged millet when all locally cultivated member classes have matched calendars.
        allowed = pd.read_parquet(ROOT / "data/clean/mapspam_area.parquet")
        required = (
            allowed.loc[allowed.mapspam_crop.isin(members) & allowed.area_ha.gt(0)]
            .groupby("iso3")
            .mapspam_crop.nunique()
        )
        g = g[g.groupby("iso3").mapspam_crop.transform("nunique").eq(g.iso3.map(required))]
        agg = (
            g.assign(**{m: g[m] * g.area_ha for m in MEASURES})
            .groupby(["iso3", "year"])[MEASURES + ["area_ha"]]
            .sum()
        )
        agg[MEASURES] = agg[MEASURES].div(agg.area_ha, axis=0)
        cap = pd.concat([cap, agg.reset_index().assign(mapspam_crop=prefix)], ignore_index=True)
    merged["mapspam_crop"] = merged.item_code.map(
        {k: v[0] if len(v) == 1 else "millet" for k, v in crop_links.items()}
    )
    merged = merged.merge(cap[KEY + ["year", *MEASURES]], on=KEY + ["year"], validate="m:1").dropna(
        subset=["yield_anom", *MEASURES]
    )
    slopes = []
    for (iso, item), g in merged.groupby(["iso3", "item"]):
        if len(g) < response.MIN_YEARS:
            continue
        slopes.append(
            dict(
                iso3=iso,
                item=item,
                n_years=len(g),
                value_share=g.value_share.iloc[0],
                **{m: response.resp_beta(g.yield_anom, g[m]) for m in MEASURES},
            )
        )
    slopes = pd.DataFrame(slopes)
    slopes.to_csv(RESULT / "season_crop_slopes.csv", index=False)
    diversity = [
        dict(iso3=i, n_crops=len(g), **{m: response.resp_div(g[m], g.value_share) for m in MEASURES})
        for i, g in slopes.groupby("iso3")
    ]
    pd.DataFrame(diversity).to_csv(RESULT / "season_response_diversity.csv", index=False)


def main():
    common, annual = matched(climate_table())
    comparisons(common, annual)


if __name__ == "__main__":
    main()
