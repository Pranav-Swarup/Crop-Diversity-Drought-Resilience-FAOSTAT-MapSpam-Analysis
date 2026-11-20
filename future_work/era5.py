"""Hourly ERA5 to daily threshold excess on the WeatherBench 1.5-degree grid. Author: Pmjarvis."""

import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from functools import lru_cache

import numpy as np
import requests
from numcodecs.blosc import decompress
from scipy.sparse import csr_matrix

from .source_objects import checked_bytes, pinned
from .storage import CACHE, RESULT

WB = "https://storage.googleapis.com/weatherbench2/datasets/era5/1959-2023_01_10-1h-240x121_equiangular_with_poles_conservative.zarr"
AR = "https://storage.googleapis.com/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
LIMITS = np.array([28.0, 30.0, 32.0], dtype=np.float32)
WB_START = date(1959, 1, 1)
AR_START = date(1900, 1, 1)
FIRST = date(1992, 1, 1)
END = date(2024, 1, 1)
OVERLAP_DAY = date(2023, 1, 1)
DAY_DIR = CACHE / "era5_daily"
DAY_DIR.mkdir(exist_ok=True)
_connections = threading.local()


def remote(url):
    if not hasattr(_connections, "session"):
        _connections.session = requests.Session()
    for attempt in range(7):
        try:
            r = _connections.session.get(url, timeout=(12, 65))
            r.raise_for_status()
            return r.content
        except requests.RequestException:
            if attempt == 6:
                raise
            time.sleep(min(30, 2**attempt))


@lru_cache(maxsize=None)
def metadata(base, var):
    return json.loads(remote(f"{base}/{var}/.zarray"))


def chunk(base, var, name):
    m = metadata(base, var)
    url = f"{base}/{var}/{name}"
    archive = "weatherbench" if base == WB else "arco"
    if var == "2m_temperature":
        record = pinned()[archive, name]
        raw = checked_bytes(remote(url + "?generation=" + record["generation"]), record)
    else:
        raw = remote(url)
    if m["compressor"]:
        if m["compressor"]["id"] != "blosc":
            raise ValueError("Unexpected Zarr compressor")
        raw = decompress(raw)
    return np.frombuffer(raw, dtype=m["dtype"]).reshape(m["chunks"])


@lru_cache(maxsize=1)
def regrid_weights():
    """Spherical cell-boundary intersection, including half-width polar cells and cyclic longitude."""
    source_lat = np.arange(-90.0, 90.01, 0.25)
    dest_lat = np.arange(-90.0, 90.01, 1.5)

    def edges(a):
        return np.r_[-90.0, (a[1:] + a[:-1]) / 2, 90.0]

    sa, da = edges(source_lat), edges(dest_lat)
    lo = np.maximum(da[:-1, None], sa[None, :-1])
    hi = np.minimum(da[1:, None], sa[None, 1:])
    wy = np.where(hi > lo, np.sin(np.deg2rad(hi)) - np.sin(np.deg2rad(lo)), 0)
    wy /= wy.sum(axis=1, keepdims=True)
    source_lon = np.arange(0, 360, 0.25)
    dest_lon = np.arange(0, 360, 1.5)
    diff = (source_lon[None, :] - dest_lon[:, None] + 180) % 360 - 180
    wx = np.maximum(0, np.minimum(0.75, diff + 0.125) - np.maximum(-0.75, diff - 0.125)) / 1.5
    if not np.allclose(wx.sum(axis=1), 1):
        raise ValueError("Non-conservative longitude grid")
    return csr_matrix(wy), csr_matrix(wx)


def native_hour(day, hour):
    offset = (day - AR_START).days * 24 + hour
    t = chunk(AR, "2m_temperature", f"{offset}.0.0")[0]
    wy, wx = regrid_weights()
    coarse = (wx @ (wy @ t[::-1]).T).T
    return coarse.astype(np.float32)


def wb_hours(block):
    return chunk(WB, "2m_temperature", f"{block}.0.0").transpose(0, 2, 1)


def grid_check():
    for base, lat, lon, start, stop, n in [
        (WB, np.arange(-90, 90.01, 1.5), np.arange(0, 360, 1.5), WB_START, date(2023, 1, 10), 48),
        (AR, np.arange(90, -90.01, -0.25), np.arange(0, 360, 0.25), AR_START, END, 1),
    ]:
        if not (
            np.allclose(chunk(base, "latitude", "0"), lat)
            and np.allclose(chunk(base, "longitude", "0"), lon)
        ):
            raise ValueError("Unexpected ERA5 grid")
        if not np.array_equal(chunk(base, "time", "0")[:48], np.arange(48)):
            raise ValueError("Unexpected hourly time coordinate")
        att = json.loads(remote(f"{base}/time/.zattrs"))
        if not att["units"].startswith(f"hours since {start.isoformat()}"):
            raise ValueError("Unexpected ERA5 time origin")
        meta = metadata(base, "2m_temperature")
        if meta["chunks"][0] != n or meta["shape"][0] < (stop - start).days * 24:
            raise ValueError("Unexpected hourly coverage")
    # Independent overlap at the change of archive. Compare Kelvin at every grid point.
    day = OVERLAP_DAY
    block, within = divmod((day - WB_START).days, 2)
    old = wb_hours(block)
    errors = []
    for hour in [0, 12, 23]:
        errors.append(np.abs(native_hour(day, hour) - old[within * 24 + hour]))
    errors = np.stack(errors)
    print(
        f"ERA5 archive overlap mean |difference|={errors.mean():.6g} K, max={errors.max():.6g} K",
        flush=True,
    )
    if errors.mean() > 0.04 or errors.max() > 0.6:
        raise ValueError("ERA5 archives do not reproduce the same coarse-grid temperatures")
    receipt = {
        "date": day.isoformat(),
        "utc_hours": [0, 12, 23],
        "grid_shape": [121, 240],
        "weatherbench_object": f"{block}.0.0",
        "arco_objects": [f"{(day - AR_START).days * 24 + hour}.0.0" for hour in [0, 12, 23]],
        "mean_absolute_kelvin": float(errors.mean()),
        "max_absolute_kelvin": float(errors.max()),
    }
    (RESULT / "era5_archive_overlap.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return float(errors.mean()), float(errors.max())


def daily(hours):
    if (
        hours.shape != (24, 121, 240)
        or not np.isfinite(hours).all()
        or hours.min() < 150
        or hours.max() > 360
    ):
        raise ValueError("Incomplete or out-of-range hourly temperature")
    c = hours - 273.15
    return np.stack([np.maximum(c - th, 0).mean(axis=0, dtype=np.float64) for th in LIMITS]).astype(
        np.float32
    )


def source_signature(day):
    if day < date(2023, 1, 1):
        names = [("weatherbench", f"{(day - WB_START).days // 2}.0.0")]
    else:
        first = (day - AR_START).days * 24
        names = [("arco", f"{hour}.0.0") for hour in range(first, first + 24)]
    records = pinned()
    identity = [(a, k, records[a, k]["generation"], records[a, k]["md5_base64"]) for a, k in names]
    return hashlib.sha256(json.dumps([day.isoformat(), identity]).encode()).hexdigest()


def verified_day(day):
    path = DAY_DIR / f"{day}.npz"
    if not path.exists():
        return False
    with np.load(path) as saved:
        return "source" in saved and str(saved["source"]) == source_signature(day)


def save(day, degree):
    p = DAY_DIR / f"{day}.npz"
    if p.exists():
        with np.load(p) as previous:
            np.testing.assert_array_equal(
                previous["ddd"], degree, err_msg=f"Reconstruction differs on {day}"
            )
    tmp = p.with_suffix(".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, ddd=degree, source=source_signature(day))
    tmp.replace(p)


def make_wb(block):
    day = WB_START + timedelta(days=2 * block)
    needed = [
        j
        for j in range(2)
        if FIRST <= day + timedelta(days=j) < date(2023, 1, 1)
        and not verified_day(day + timedelta(days=j))
    ]
    if needed:
        temp = wb_hours(block)
        for j in needed:
            save(day + timedelta(days=j), daily(temp[24 * j : 24 * j + 24]))


def make_native(day):
    if verified_day(day):
        return
    temp = np.stack([native_hour(day, h) for h in range(24)])
    save(day, daily(temp))


def download(source="both", workers=12):
    if source in ("wb", "both"):
        a = (FIRST - WB_START).days // 2
        b = ((date(2023, 1, 1) - WB_START).days - 1) // 2
        with ThreadPoolExecutor(workers) as pool:
            for idx, _ in enumerate(pool.map(make_wb, range(a, b + 1))):
                if idx % 300 == 0:
                    print(f"WeatherBench blocks {idx + 1}/{b - a + 1}", flush=True)
    if source in ("native", "both"):
        start = date(2023, 1, 1)
        with ThreadPoolExecutor(workers) as pool:
            for i, _ in enumerate(
                pool.map(make_native, [start + timedelta(days=i) for i in range(365)])
            ):
                if i % 20 == 0:
                    print(f"Native ERA5 days {i + 1}/365", flush=True)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("source", choices=["check", "both", "wb", "native"])
    p.add_argument("--workers", type=int, default=12)
    a = p.parse_args()
    if a.source == "check":
        grid_check()
    else:
        download(a.source, a.workers)
