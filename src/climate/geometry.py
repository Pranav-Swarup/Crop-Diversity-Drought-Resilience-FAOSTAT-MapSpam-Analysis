"""Natural Earth admin-0 dissolved to our iso3 codes -> data/clean/countries.geojson.

First-iteration pipeline written from src/metrics' side; Prathmesh owns this module.
Geometry is simplified (0.02 degrees) to keep the committed file small; the climate
aggregation runs on a 0.5 degree grid, so this does not affect it.

Run: python -m src.climate.geometry
"""
import geopandas as gpd

from ..common import paths
from ..common.contract import validate_geojson
from .download import NE_DIR

# Natural Earth code -> our code. Kosovo (KOS) and Somaliland (SOL) have no ISO code in
# Natural Earth and are reported inside Serbia and Somalia by FAOSTAT.
REMAP = {"BEL": "BLX", "LUX": "BLX", "SRB": "SCG", "MNE": "SCG", "KOS": "SCG",
         "SDN": "SDX", "SSD": "SDX", "SOL": "SOM"}
SIMPLIFY_DEG = 0.02


def main():
    ne = gpd.read_file(f"zip://{NE_DIR / 'ne_10m_admin_0_countries.zip'}")
    assert ne.crs.to_epsg() == 4326
    code = ne["ISO_A3_EH"].where(ne["ISO_A3_EH"] != "-99", ne["ADM0_A3"])
    ne["iso3"] = code.replace(REMAP)
    out = ne[["iso3", "geometry"]].dissolve(by="iso3").reset_index()
    out["geometry"] = out.geometry.simplify(SIMPLIFY_DEG, preserve_topology=True).make_valid()
    out = out[~out.geometry.is_empty]

    paths.CLEAN.mkdir(parents=True, exist_ok=True)
    out.to_file(paths.COUNTRIES_GEOJSON, driver="GeoJSON", COORDINATE_PRECISION=3)
    iso = validate_geojson(paths.COUNTRIES_GEOJSON)
    assert {"BLX", "SCG", "SDX", "CHN", "TWN"} <= set(iso)
    assert not set(REMAP) & set(iso)
    size = paths.COUNTRIES_GEOJSON.stat().st_size / 1e6
    assert size < 50, f"countries.geojson is {size:.0f} MB"
    print(f"countries.geojson: {len(iso)} features, {size:.1f} MB")
    if paths.COUNTRIES.exists():
        import pandas as pd
        ours = set(pd.read_csv(paths.COUNTRIES)["iso3"])
        print(f"countries.csv entities without a geometry: {sorted(ours - set(iso))}")


if __name__ == "__main__":
    main()
