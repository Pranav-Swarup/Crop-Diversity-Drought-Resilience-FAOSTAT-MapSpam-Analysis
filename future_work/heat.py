"""Growing-season ERA5 exposure, independently observed NOAA greenness and crop output. Author: Pmjarvis."""

import calendar
from datetime import date, timedelta

import numpy as np
import pandas as pd

from src.metrics import panel
from src.metrics.detrend import lowess_trend

from .era5 import DAY_DIR, source_signature
from .seasons import KEY, MIN_COVERAGE, YEARS, footprints
from .storage import RESULT, ROOT

DEG = ["edd28", "edd30", "edd32"]


def ordinal(day, year):
    """Actual zero-based day of year for a month/day in the static 365-day GGCMI calendar."""
    day = np.asarray(day)
    return day.astype(int) - 1 + (calendar.isleap(year) & (day >= 60))


def bounds(plant, harvest, year):
    """Offsets into the consecutive previous and current calendar years; end is exclusive."""
    previous = 365 + int(calendar.isleap(year - 1))
    wraps = np.asarray(plant) > np.asarray(harvest)
    start = np.where(wraps, ordinal(plant, year - 1), previous + ordinal(plant, year))
    return start, previous + ordinal(harvest, year) + 1


def era_footprints(frame):
    """Split 0.5° crop cells across overlapping 1.5° ERA5 cells, conserving physical area."""
    cols = ["iso3", "mapspam_crop", "plant", "harvest"]
    src = frame.groupby(cols + ["cell"], as_index=False, observed=True).ha.sum()
    cell = src.cell.to_numpy()
    lat = 89.75 - 0.5 * (cell // 720)
    lon = -179.75 + 0.5 * (cell % 720)
    lo, hi = lat - 0.25, lat + 0.25
    west, east = lon - 0.25, lon + 0.25
    first_y = np.clip(np.floor((lo + 90 + 0.75) / 1.5).astype(int), 0, 120)
    last_y = np.clip(np.floor((hi + 90 + 0.75 - 1e-10) / 1.5).astype(int), 0, 120)
    first_x = np.floor((west + 0.75) / 1.5).astype(int)
    last_x = np.floor((east + 0.75 - 1e-10) / 1.5).astype(int)
    pieces = []
    total = np.zeros(len(src))
    for j, y in enumerate([first_y, last_y]):
        center = -90 + 1.5 * y
        top = np.minimum(hi, center + 0.75)
        bottom = np.maximum(lo, center - 0.75)
        fy = (np.sin(np.deg2rad(top)) - np.sin(np.deg2rad(bottom))) / (
            np.sin(np.deg2rad(hi)) - np.sin(np.deg2rad(lo))
        )
        if j:
            fy = np.where(first_y != last_y, fy, 0)
        for k, x in enumerate([first_x, last_x]):
            fx = (np.minimum(east, 1.5 * x + 0.75) - np.maximum(west, 1.5 * x - 0.75)) / 0.5
            if k:
                fx = np.where(first_x != last_x, fx, 0)
            f = np.maximum(fy * fx, 0)
            ok = f > 0
            total += f
            pieces.append(
                src.loc[ok, cols].assign(grid=y[ok] * 240 + x[ok] % 240, ha=(src.ha.to_numpy() * f)[ok])
            )
    np.testing.assert_allclose(total, 1, atol=1e-10)
    return pd.concat(pieces, ignore_index=True).groupby(cols + ["grid"], as_index=False).ha.sum()


def era_seasons(years=YEARS):
    years = np.asarray(years, dtype=int)
    if (
        years.ndim != 1
        or not len(years)
        or np.any(np.diff(years) != 1)
        or years[0] < YEARS[0]
        or years[-1] > YEARS[-1]
    ):
        raise ValueError("ERA5 harvest years must be consecutive and within the study period")
    grouped = era_footprints(footprints())
    pairs = grouped[KEY].drop_duplicates().sort_values(KEY).reset_index(drop=True)
    grouped = grouped.merge(pairs.reset_index(names="gid"), on=KEY, validate="m:1")
    cell = grouped.grid.to_numpy()
    gid = grouped.gid.to_numpy()
    ha = grouped.ha.to_numpy()
    denom = np.bincount(gid, weights=ha)
    all_years = []

    # Each year is read once. The prefix sum integrates plant-to-harvest exposure including leap days.
    def load_year(year):
        d = date(year, 1, 1)
        days = (date(year + 1, 1, 1) - d).days
        arr = np.empty((days, len(DEG), 121 * 240), dtype=np.float32)
        for i in range(days):
            with np.load(DAY_DIR / f"{d + timedelta(days=i)}.npz") as f:
                actual_day = d + timedelta(days=i)
                if "source" not in f or str(f["source"]) != source_signature(actual_day):
                    raise ValueError(f"Hourly sources have not been verified for {actual_day}")
                arr[i] = f["ddd"].reshape(len(DEG), -1)
        return arr

    prev = load_year(int(years[0]) - 1)
    for year in years:
        current = load_year(int(year))
        start, stop = bounds(grouped.plant, grouped.harvest, int(year))
        out = pairs.copy().assign(year=int(year))
        for j, label in enumerate(DEG):
            days = np.concatenate([prev[:, j], current[:, j]])
            prefix = np.empty((len(days) + 1, days.shape[1]), dtype=np.float64)
            prefix[0] = 0
            np.cumsum(days, axis=0, out=prefix[1:])
            exposure = prefix[stop, cell] - prefix[start, cell]
            out[label] = np.bincount(gid, weights=ha * exposure) / denom
        all_years.append(out)
        prev = current
    return pd.concat(all_years, ignore_index=True)


def vci_seasons():
    weekly = pd.read_parquet(RESULT / "vci_weekly.parquet")
    f = footprints().groupby(KEY + ["plant", "harvest"], as_index=False).ha.sum()
    pairs = f[KEY].drop_duplicates().sort_values(KEY).reset_index(drop=True)
    f = f.merge(pairs.reset_index(names="gid"), on=KEY, validate="m:1")
    cols = pd.MultiIndex.from_frame(pairs)
    table = weekly.pivot(index=["year", "week"], columns=KEY, values="vci").reindex(columns=cols)

    def daily_values(year):
        # Official NOAA week 52 ends on day 364 even in leap years. Extra year-end days are unobserved.
        days = 365 + int(calendar.isleap(year))
        idx = pd.MultiIndex.from_arrays(
            [np.repeat(year, days), np.arange(days) // 7 + 1], names=["year", "week"]
        )
        return table.reindex(idx).to_numpy()

    gid = f.gid.to_numpy()
    ha = f.ha.to_numpy()
    den = np.bincount(gid, weights=ha)
    all_years = []
    for year in YEARS:
        start, stop = bounds(f.plant, f.harvest, int(year))
        v = np.vstack([daily_values(int(year) - 1), daily_values(int(year))])
        prefix = np.vstack([np.zeros(len(pairs)), np.cumsum(np.nan_to_num(v), axis=0)])
        counts = np.vstack([np.zeros(len(pairs)), np.cumsum(np.isfinite(v), axis=0)])
        obs = counts[stop, gid] - counts[start, gid]
        mean = np.divide(
            prefix[stop, gid] - prefix[start, gid], obs, out=np.full(len(f), np.nan), where=obs > 0
        )
        ok = (obs > 0) & (obs / (stop - start) >= MIN_COVERAGE)
        observed = np.bincount(gid, weights=ha * ok, minlength=len(pairs))
        vci = np.bincount(gid[ok], weights=ha[ok] * mean[ok], minlength=len(pairs))
        coverage = observed / den
        value = np.divide(
            vci,
            observed,
            out=np.full(len(pairs), np.nan),
            where=(observed > 0) & (coverage >= MIN_COVERAGE),
        )
        all_years.append(pairs.assign(year=int(year), vci=value, vci_area_coverage=coverage))
    return pd.concat(all_years, ignore_index=True)


def fixed_satellite_basket(crop):
    """Keep calendar-complete crops with observed satellite coverage in at least one season."""
    observed = crop.groupby(KEY).vci.transform("count").gt(0)
    return crop[crop.complete & observed].copy()


def standardized_excess(series):
    values = series.to_numpy(dtype=float)
    if np.ptp(values) == 0:
        return np.full(len(values), np.nan)
    residual = values - lowess_trend(values)
    sd = residual.std(ddof=1)
    return residual / sd if sd > 0 else np.full(len(values), np.nan)


def national(crop):
    common = fixed_satellite_basket(crop)
    measures = [*DEG, "vci", "season"]
    weighted = common.assign(**{c: common[c] * common.area_ha for c in measures})
    nat = weighted.groupby(["iso3", "year"])[measures + ["area_ha"]].sum(min_count=1)
    nat[measures] = nat[measures].div(nat.area_ha, axis=0)
    counts = common.groupby(["iso3", "year"]).vci.agg(["count", "size"])
    observed_area = (
        common.area_ha.where(common.vci.notna(), 0).groupby([common.iso3, common.year]).sum()
    )
    nat["basket_crops"] = counts["size"]
    nat["observed_vci_crops"] = counts["count"]
    nat["vci_basket_area_share"] = observed_area / nat.area_ha
    nat["vci"] = nat.vci.where(counts["count"].eq(counts["size"]))
    nat = nat.reset_index().sort_values(["iso3", "year"])
    for label in DEG:
        nat[label + "_z"] = nat.groupby("iso3")[label].transform(standardized_excess)
    old = pd.read_parquet(ROOT / "outputs/metrics_national.parquet")[
        ["iso3", "year", "anom", "spei12_w"]
    ]
    cru = pd.read_parquet(ROOT / "data/clean/heat.parquet")[["iso3", "year", "heat_z_w"]]
    out = nat.merge(old, on=["iso3", "year"], validate="1:1").merge(
        cru, on=["iso3", "year"], validate="1:1"
    )
    full = pd.read_parquet(ROOT / "data/clean/mapspam_area.parquet").groupby("iso3").area_ha.sum()
    out["share_all_cropland"] = out.area_ha / out.iso3.map(full)
    return out


def presented_benchmark():
    national = pd.read_parquet(ROOT / "outputs/metrics_national.parquet")
    temp = pd.read_parquet(ROOT / "data/clean/heat.parquet")
    years = national.merge(
        temp[["iso3", "year", "heat_z_w"]], on=["iso3", "year"], validate="1:1"
    ).dropna(subset=["anom", "heat_z_w", "spei12_w"])
    dry = years.spei12_w.le(-1)
    hot = years.heat_z_w.ge(1)
    terms = {"dry_only": dry & ~hot, "hot_only": hot & ~dry, "compound": dry & hot}
    years = years.assign(**{k: v.astype(float) for k, v in terms.items()})
    fit = panel.fe_fit(years, "anom", list(terms))
    frozen = (
        pd.read_csv(ROOT / "outputs/stats/heat_panel.csv")
        .query('model=="year_types"')
        .set_index("term")
    )
    rows = []
    for row in panel.coef_rows(fit):
        term = row["term"]
        cols = ["beta", "se", "ci_lo", "ci_hi", "p"]
        np.testing.assert_allclose(
            [row[c] for c in cols], frozen.loc[term, cols].to_numpy(float), atol=1e-10
        )
        rows.append(
            dict(
                model="presented_full_sample",
                outcome="anom",
                **row,
                n_obs=len(years),
                n_countries=years.iso3.nunique(),
                n_exposed=int(terms[term].sum()),
                n_neither=int((~hot & ~dry).sum()),
            )
        )
    pd.DataFrame(rows).to_csv(RESULT / "heat_presented_benchmark.csv", index=False)


def heat_models(nat):
    specs = {
        "presented_CRU_Dec12": ("heat_z_w", "spei12_w"),
        "ERA5_30_Dec12": ("edd30_z", "spei12_w"),
        **{f"ERA5_{n:02d}_season": (f"edd{n}_z", "season") for n in [28, 30, 32]},
    }
    rows = []
    valid = nat.dropna(
        subset=["vci", "anom", "heat_z_w", "spei12_w", "season", *[k + "_z" for k in DEG]]
    )
    flags = valid[["iso3", "year"]].copy()
    for name, (heat, drought) in specs.items():
        d = valid.copy()
        hot = d[heat] >= 1
        dry = d[drought] <= -1
        for term, indicator in {
            "hot_only": hot & ~dry,
            "dry_only": dry & ~hot,
            "compound": hot & dry,
        }.items():
            d[term] = indicator.astype(float)
        flags[name] = np.select(
            [hot & dry, hot & ~dry, dry & ~hot], ["compound", "hot_only", "dry_only"], default="neither"
        )
        for outcome in ["anom", "vci"]:
            terms = ["dry_only", "hot_only", "compound"]
            fit = panel.fe_fit(d, outcome, terms)
            for row in panel.coef_rows(fit):
                rows.append(
                    dict(
                        model=name,
                        outcome=outcome,
                        **row,
                        n_obs=len(d),
                        n_countries=d.iso3.nunique(),
                        n_exposed=int(d[row["term"]].sum()),
                        n_neither=int((~hot & ~dry).sum()),
                    )
                )
    flags.to_csv(RESULT / "heat_year_types.csv", index=False)
    switches = flags.groupby(["presented_CRU_Dec12", "ERA5_30_season"]).size().unstack(fill_value=0)
    switches.to_csv(RESULT / "heat_year_type_transitions.csv")
    audit = nat[["iso3", "year", "basket_crops", "observed_vci_crops", "vci_basket_area_share"]].copy()
    audit["satellite_complete"] = nat.vci.notna()
    audit["temperature_standardized"] = nat[[k + "_z" for k in DEG]].notna().all(axis=1)
    audit["in_model"] = audit.set_index(["iso3", "year"]).index.isin(
        valid.set_index(["iso3", "year"]).index
    )
    audit.to_csv(RESULT / "heat_sample_coverage.csv", index=False)
    pd.DataFrame(rows).to_csv(RESULT / "heat_effects.csv", index=False)
    return valid


def main():
    e = era_seasons()
    v = vci_seasons()
    v.to_parquet(RESULT / "vci_seasonal.parquet", index=False)
    keys = KEY + ["year"]
    season = pd.read_parquet(RESULT / "season_crop_year.parquet")
    crop = season.merge(e, on=keys, validate="1:1").merge(v, on=keys, validate="1:1")
    crop.to_parquet(RESULT / "heat_crop_year.parquet", index=False)
    nat = national(crop)
    nat.to_csv(RESULT / "heat_country_year.csv", index=False)
    heat_models(nat)
    presented_benchmark()


if __name__ == "__main__":
    main()
