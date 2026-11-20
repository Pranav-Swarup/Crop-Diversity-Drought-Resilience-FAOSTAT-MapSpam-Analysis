"""Independently check country-crop VCI against weekly rasters on the ERA5 study's actual crop footprint."""

import re
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import netCDF4
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from .heat import bounds, fixed_satellite_basket
from .seasons import KEY, footprints
from .storage import RAW, RESULT, fetch

ARCHIVE = "https://www.star.nesdis.noaa.gov/data/pub0018/VHPdata4users/data/Blended_VH_16km/VH/"


def overlap(src, dst, latitude):
    """Fraction of each target cell overlapped by source cells, on an equal-angle geographic grid."""

    def edges(x):
        step = np.median(np.diff(x))
        return np.r_[x[0] - step / 2, (x[1:] + x[:-1]) / 2, x[-1] + step / 2]

    a, b = edges(np.asarray(src, dtype=float)), edges(np.asarray(dst, dtype=float))
    lower = np.maximum(b[:-1, None], a[None, :-1])
    upper = np.minimum(b[1:, None], a[None, 1:])

    def measure(x):
        return np.sin(np.deg2rad(x)) if latitude else x

    return csr_matrix(
        np.where(upper > lower, measure(upper) - measure(lower), 0) / np.diff(measure(b))[:, None]
    )


def unpack(v, qa):
    """NOAA's published valid_range is in physical units; packed VCI runs from 0 to 10000."""
    v.set_auto_maskandscale(False)
    qa.set_auto_maskandscale(False)
    raw = v[:]
    q = qa[:].astype(np.uint8)
    value = raw.astype(float) * float(v.scale_factor) + float(v.add_offset)
    good = (
        (raw != v._FillValue)
        & (value >= 0)
        & (value <= 100.00001)
        & ((q & 1) == 0)
        & ((q & 16) == 0)
        & ((q & (4 | 8)) != 0)
    )
    return np.where(good, value, np.nan)


def vegetation_grids():
    folder = RAW / "noaa_grid"
    listing = fetch(ARCHIVE, folder / "index.html").read_text()
    items = {}
    for name, year, week in re.findall(
        r'href="(VHP\.G16\.C07\.\w+\.P(\d{4})(\d{3})\.VH\.nc)"', listing
    ):
        y, w = int(year), int(week)
        if 2019 <= y <= 2023:
            if (y, w) in items:
                raise ValueError(f"Ambiguous gridded VCI for {y}/{w}")
            items[(y, w)] = name
    expected = {(y, w) for y in range(2019, 2024) for w in range(1, 53)}
    if set(items) != expected:
        raise ValueError(
            f"Unexpected NOAA inventory. Missing {sorted(expected - set(items))}, extra {sorted(set(items) - expected)}"
        )
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(lambda n: fetch(ARCHIVE + n, folder / n), items.values()))
    f = footprints()
    cells = np.sort(f.cell.unique())
    values = {}
    wy = wx = None
    for (y, w), name in sorted(items.items()):
        with netCDF4.Dataset(folder / name) as d:
            first = date(y, 1, 1) + timedelta(days=(w - 1) * 7)
            if str(d.time_coverage_start)[:8] != first.strftime("%Y%m%d") or str(d.time_coverage_end)[
                :8
            ] != (first + timedelta(days=6)).strftime("%Y%m%d"):
                raise ValueError(f"Unexpected NOAA composite dates in {name}")
            lon, lat = d["longitude"][:], d["latitude"][:][::-1]
            if wy is None:
                wy = overlap(lat, np.arange(-89.75, 90, 0.5), True)
                wx = overlap(lon, np.arange(-179.75, 180, 0.5), False)
                grid = (lat.copy(), lon.copy())
            elif not (np.array_equal(grid[0], lat) and np.array_equal(grid[1], lon)):
                raise ValueError("NOAA grid changed")
            v = unpack(d["VCI"], d["QA"])[::-1]
            num = (wx @ (wy @ np.nan_to_num(v)).T).T
            den = (wx @ (wy @ np.isfinite(v).astype(float)).T).T
            # Coastal or partly masked cells need at least half their area measured; report retained hectares.
            avg = np.divide(num, den, out=np.full_like(num, np.nan), where=den >= 0.5)
            values[y, w] = avg[::-1].ravel()[cells].astype("float32")
    return cells, values


def run():
    cells, weekly = vegetation_grids()
    f = footprints().groupby(KEY + ["plant", "harvest", "cell"], as_index=False).ha.sum()
    pairs = f[KEY].drop_duplicates().sort_values(KEY).reset_index(drop=True)
    f = f.merge(pairs.reset_index(names="gid"), on=KEY, validate="m:1")
    cell = np.searchsorted(cells, f.cell)
    gid = f.gid.to_numpy()
    ha = f.ha.to_numpy()
    denom = np.bincount(gid, weights=ha)

    def year_grid(year):
        n = (date(year + 1, 1, 1) - date(year, 1, 1)).days
        blank = np.full(len(cells), np.nan, dtype=np.float32)
        return np.stack([weekly.get((year, i // 7 + 1), blank) for i in range(n)])

    rows = []
    for year in range(2020, 2024):
        d = np.concatenate([year_grid(year - 1), year_grid(year)])
        sums = np.zeros((len(d) + 1, len(cells)), dtype=np.float64)
        count = np.zeros((len(d) + 1, len(cells)), dtype=np.int16)
        np.cumsum(np.nan_to_num(d), axis=0, out=sums[1:])
        np.cumsum(np.isfinite(d), axis=0, out=count[1:])
        a, b = bounds(f.plant, f.harvest, year)
        obs = count[b, cell] - count[a, cell]
        good = obs >= 0.9 * (b - a)
        mean = np.divide(sums[b, cell] - sums[a, cell], obs, where=obs > 0, out=np.zeros(len(f)))
        valid = np.bincount(gid[good], weights=ha[good], minlength=len(pairs))
        value = np.bincount(gid[good], weights=(ha * mean)[good], minlength=len(pairs))
        share = valid / denom
        rows.append(
            pairs.assign(
                year=year,
                grid_vci=np.divide(
                    value, valid, out=np.full(len(pairs), np.nan), where=(valid > 0) & (share >= 0.9)
                ),
                calendar_ha=denom,
                observed_ha=valid,
                observed_share=share,
            )
        )
    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(RESULT / "vci_shared_grid_2020_2023.parquet", index=False)
    print(
        f"{len(weekly)} real weekly grids, {out.grid_vci.notna().sum()} eligible country-crop-years",
        flush=True,
    )


def comparison():
    """Compare like seasons on a fixed basket present in the full-history satellite analysis."""
    keys = [*KEY, "year"]
    admin = pd.read_parquet(RESULT / "vci_seasonal.parquet")
    area = pd.read_parquet(RESULT / "season_crop_year.parquet", columns=keys + ["complete", "area_ha"])
    grid = pd.read_parquet(RESULT / "vci_shared_grid_2020_2023.parquet")
    history = area.merge(admin, on=keys, validate="1:1")
    pairs = fixed_satellite_basket(history)[keys + ["area_ha", "vci"]].merge(
        grid, on=keys, validate="1:1"
    )
    fixed = pairs.groupby(KEY)[["grid_vci", "vci"]].transform("count").eq(4).all(axis=1)
    pairs = pairs.loc[fixed].sort_values(keys).reset_index(drop=True)
    if pairs.empty:
        raise ValueError("No complete four-year matched VCI series")
    pairs.to_parquet(RESULT / "vci_footprints_paired.parquet", index=False)
    w = pairs.assign(admin_sum=pairs.area_ha * pairs.vci, grid_sum=pairs.area_ha * pairs.grid_vci)
    national = w.groupby(["iso3", "year"], as_index=False).agg(
        area_ha=("area_ha", "sum"),
        crops=("mapspam_crop", "size"),
        admin_sum=("admin_sum", "sum"),
        grid_sum=("grid_sum", "sum"),
    )
    national["vci"] = national.pop("admin_sum") / national.area_ha
    national["grid_vci"] = national.pop("grid_sum") / national.area_ha
    national.to_csv(RESULT / "figure_03b_vegetation.csv", index=False)
    stats = []
    for level, d in [("country-crop-year", pairs), ("country-year", national)]:
        delta = d.grid_vci - d.vci
        stats.append(
            dict(
                level=level,
                n=len(d),
                n_countries=d.iso3.nunique(),
                year_start=int(d.year.min()),
                year_end=int(d.year.max()),
                pearson=d.vci.corr(d.grid_vci),
                spearman=d.vci.corr(d.grid_vci, method="spearman"),
                mean_difference=delta.mean(),
                mean_absolute_difference=delta.abs().mean(),
            )
        )
    pd.DataFrame(stats).to_csv(RESULT / "vci_footprint_agreement.csv", index=False)


if __name__ == "__main__":
    run()
