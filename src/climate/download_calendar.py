"""Download the GGCMI Phase 3 crop calendar (planting and maturity dates, 0.5 degree) for the seasonal-exposure extension.

One NetCDF file per crop and water regime (`rf` rainfed, `ir` irrigated), about 100 KB each, for 19 crops.

Source: Jagermeyr et al. 2021, GGCMI Phase 3 crop calendar.
        Zenodo record 5062513, DOI 10.5281/zenodo.5062513, licence CC-BY-4.0
        https://zenodo.org/records/5062513

Run: python -m src.climate.download_calendar
"""
from ..common import paths
from ..common.sources import download

CAL_DIR = paths.RAW / "calendar"
CAL_DOI = "10.5281/zenodo.5062513"
CAL_URL = "https://zenodo.org/records/5062513/files/{name}?download=1"
CROPS = ["mai", "ri1", "wwh", "swh", "soy", "bar", "rye", "mil", "sor", "pea", "bea", "sgb", "sgc", "cas", "sun", "rap", "nut", "cot", "pot"]
REGIMES = ["rf", "ir"]


def cal_name(crop, regime):
    return f"{crop}_{regime}_ggcmi_crop_calendar_phase3_v1.01.nc4"


def cal_file(crop, regime):
    return CAL_DIR / cal_name(crop, regime)


def main():
    for crop in CROPS:
        for regime in REGIMES:
            name = cal_name(crop, regime)
            download(f"GGCMI Phase 3 crop calendar {crop} {regime}", CAL_URL.format(name=name), CAL_DIR / name,
                     doi=CAL_DOI, note="Planting and maturity day of year, 0.5 degree (Jagermeyr et al. 2021), CC-BY-4.0")
    missing = [cal_name(c, r) for c in CROPS for r in REGIMES if not cal_file(c, r).exists()]
    assert not missing, f"calendar files missing: {missing}"
    small = [f.name for f in CAL_DIR.glob("*.nc4") if f.stat().st_size < 10_000]
    assert not small, f"calendar files look truncated: {small}"
    print(f"\ncalendar files: {len(CROPS) * len(REGIMES)} ({len(CROPS)} crops x {len(REGIMES)} regimes) in {CAL_DIR}")


if __name__ == "__main__":
    main()
