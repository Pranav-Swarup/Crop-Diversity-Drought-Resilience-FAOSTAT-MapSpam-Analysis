"""CRU TS v4.10 monthly maximum temperature -> cropland-weighted country-year heat exposure.

  tmx_warm_w      deg C. Per 0.5 degree cell: mean Tmax of the cell's 3 climatologically warmest calendar
                  months (ranked on the 1993-2023 monthly means; the 3 months need not be consecutive and are
                  taken inside the calendar year, the same year convention as December SPEI-12). Then the
                  cropland-weighted (MapSPAM) mean over the country's cells.
  tmx_warm_trend  deg C. LOWESS trend (frac = 0.5) of tmx_warm_w over 1993-2023, per country.
  heat_z_w        (tmx_warm_w - tmx_warm_trend) / SD of that residual (ddof = 1), per country.
                  Detrended, so a hot year is hot relative to the temperature expected for that year; crop
                  yields are detrended the same way.
  stn_share_w     cropland-weighted share of warm-month cells with at least one contributing station
                  (CRU `stn` > 0). Where no station is in range CRU relaxes to the 1961-1990 climatology,
                  so the anomaly there is zero by construction.

Hot year: heat_z_w >= +1.0. Heat event: mirror of the drought event rule (src.metrics.events.find_events on
the sign-flipped series): first year of a run of hot years, 3 years before and after inside the window, the
3 years before hot-free. Severity: moderate >= 1.0, severe >= 1.5, extreme >= 2.0.

This is monthly data: it measures hot warm seasons, not daily extreme degree days.

Outputs: data/clean/heat.parquet       iso3, year, tmx_warm_w, tmx_warm_trend, heat_z_w, stn_share_w
         data/clean/heat_events.csv    iso3, year, heat_z_w, severity

Run: python -m src.climate.download_cru, then python -m src.climate.heat
"""
import sys
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr

from ..common import paths
from ..common.contract import YEAR_MAX, YEAR_MIN, summary, validate
from ..metrics.detrend import lowess_trend
from ..metrics.events import PAD, find_events
from . import cropland
from .aggregate import MIN_COVERAGE, coverage_fractions, weighted_mean
from .download_cru import nc_files

HOT_THR = 1.0
N_WARM = 3
YEARS = np.arange(YEAR_MIN, YEAR_MAX + 1)


def load_cru(var="tmx"):
    """CRU variable for the window as (time, lat north->south, lon west->east)."""
    parts = [xr.open_dataset(f)[var] for f in nc_files()]
    da = xr.concat(parts, dim="time").sel(time=slice(f"{YEAR_MIN}-01-01", f"{YEAR_MAX}-12-31"))
    da = da.sortby("lat", ascending=False).sortby("lon")
    assert da.shape == (12 * len(YEARS), 360, 720), da.shape
    assert np.isclose(float(da.lat[0]), 89.75) and np.isclose(float(da.lon[0]), -179.75), "CRU grid is not the SPEI grid"
    assert list(da.time.dt.month.values[:12]) == list(range(1, 13))
    return da


def warm_months(monthly):
    """monthly: (years, 12, cells). Boolean (12, cells): the N_WARM warmest calendar months of each cell by
    window mean. Cells with no data are all False."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)         # ocean cells are all-NaN
        clim = np.nanmean(monthly, axis=0)                      # (12, cells)
    ok = np.isfinite(clim).all(axis=0)
    order = np.argsort(np.where(np.isfinite(clim), clim, -np.inf), axis=0)[-N_WARM:]
    mask = np.zeros(clim.shape, dtype=bool)
    np.put_along_axis(mask, order, True, axis=0)
    mask[:, ~ok] = False
    return mask


def warm_season_mean(monthly, mask):
    """Mean over the masked months, per year and cell: (years, cells). NaN where the cell has no warm months."""
    n = mask.sum(axis=0)
    total = np.where(mask[None], monthly, 0.0).sum(axis=1)
    return np.where(n > 0, total / np.where(n > 0, n, 1), np.nan)


def heat_z(years, tmx):
    """LOWESS trend of a country series and the standardised residual. Returns (trend, z)."""
    trend = lowess_trend(tmx, years)
    resid = np.asarray(tmx, dtype=float) - trend
    sd = np.nanstd(resid, ddof=1)
    return trend, resid / sd if np.isfinite(sd) and sd > 0 else np.full(len(resid), np.nan)


def heat_events(heat):
    """Heat events of a DataFrame with iso3, year, heat_z_w, via find_events on the negated series."""
    flipped = heat.dropna(subset=["heat_z_w"])[["iso3", "year"]].copy()
    flipped["spei12_w"] = -heat.loc[flipped.index, "heat_z_w"]
    ev = find_events(flipped, thr=-HOT_THR)
    ev["spei12_w"] = -ev["spei12_w"]
    return ev.rename(columns={"spei12_w": "heat_z_w"})


def main():
    if not cropland.OUT.exists():
        print("cropland raster missing: run python -m src.climate.cropland first")
        sys.exit(1)
    crop_flat = np.load(cropland.OUT).ravel()               # (360, 720) north -> south
    tmx_da = load_cru("tmx")
    assert tmx_da.attrs.get("units") == "degrees Celsius", tmx_da.attrs
    tmx = tmx_da.values.reshape(len(YEARS), 12, -1).astype(float)
    stn = load_cru("stn").values.reshape(len(YEARS), 12, -1)

    mask = warm_months(tmx)
    warm = warm_season_mean(tmx, mask)                      # (years, cells)
    has_stn = warm_season_mean((stn > 0).astype(float), mask)

    template = tmx_da.isel(time=0).rio.write_crs(4326).rio.set_spatial_dims(x_dim="lon", y_dim="lat")
    cover = coverage_fractions(gpd.read_file(paths.COUNTRIES_GEOJSON), template)

    rows, low_cov = [], []
    for iso3, (cells, frac) in sorted(cover.items()):
        w = crop_flat[cells] * frac if len(cells) else np.array([])
        if len(cells) == 0 or w.sum() <= 0:
            continue
        t, cov = weighted_mean(warm[:, cells], w)
        if not np.isfinite(t).all():
            continue
        if np.nanmin(cov) < MIN_COVERAGE:
            low_cov.append((iso3, round(float(np.nanmin(cov)), 2)))
        s, _ = weighted_mean(has_stn[:, cells], w)
        trend, z = heat_z(YEARS, t)
        rows.append(pd.DataFrame({"iso3": iso3, "year": YEARS, "tmx_warm_w": t, "tmx_warm_trend": trend,
                                  "heat_z_w": z, "stn_share_w": s}))
    out = pd.concat(rows, ignore_index=True).astype({"year": int})
    out = out[out["heat_z_w"].notna()].reset_index(drop=True)

    validate(out, "heat")
    assert out["tmx_warm_w"].between(-20, 60).all(), "warm-season Tmax outside a plausible range"
    assert out["heat_z_w"].abs().max() < 6
    assert (out.groupby("iso3")["year"].count() == len(YEARS)).all(), "country with an incomplete heat series"
    sd = out.groupby("iso3")["heat_z_w"].std(ddof=1)
    assert np.allclose(sd, 1.0, atol=0.05), "heat_z_w is not unit-variance per country"

    ev = heat_events(out)
    validate(ev, "heat_events")
    z = out.set_index(["iso3", "year"])["heat_z_w"]
    for r in ev.itertuples():
        prior = z.reindex([(r.iso3, y) for y in range(r.year - PAD, r.year)])
        assert prior.notna().all() and (prior < HOT_THR).all(), f"{r.iso3} {r.year}: hot year in prior {PAD} years"

    paths.HEAT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(paths.HEAT, index=False)
    ev.to_csv(paths.HEAT_EVENTS, index=False)

    print(summary(out, "heat"))
    print(summary(ev, "heat_events"))
    hot = out["heat_z_w"] >= HOT_THR
    print(f"hot years (heat_z_w >= {HOT_THR}): {int(hot.sum())} ({hot.mean():.1%} of country-years); "
          f"events: {len(ev)} in {ev['iso3'].nunique()} countries; {ev['severity'].value_counts().to_dict()}")
    slope = out.groupby("iso3").apply(lambda g: np.polyfit(g["year"], g["tmx_warm_trend"], 1)[0] * 10, include_groups=False)
    print(f"warm-season Tmax trend, deg C per decade: median {slope.median():+.2f}, range {slope.min():+.2f} to {slope.max():+.2f}")
    stn_mean = out.groupby("iso3")["stn_share_w"].mean()
    print(f"station support (mean stn_share_w): median {stn_mean.median():.2f}; countries below 0.5: "
          f"{sorted(stn_mean[stn_mean < 0.5].index)}")
    print(f"countries with < {MIN_COVERAGE:.0%} valid cropland coverage in some year: {low_cov}")
    if paths.CLIMATE.exists():
        m = out.merge(pd.read_parquet(paths.CLIMATE)[["iso3", "year", "spei12_w"]], on=["iso3", "year"])
        m = m.dropna(subset=["spei12_w"])
        dry = m["spei12_w"] <= -1.0
        both = dry & (m["heat_z_w"] >= HOT_THR)
        print(f"correlation heat_z_w vs spei12_w (pooled): {m['heat_z_w'].corr(m['spei12_w']):+.2f}; "
              f"compound hot-dry country-years: {int(both.sum())} ({both.sum() / dry.sum():.0%} of dry years)")


if __name__ == "__main__":
    main()
