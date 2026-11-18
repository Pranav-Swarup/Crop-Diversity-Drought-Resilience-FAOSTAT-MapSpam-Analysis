"""Match DES districts (2022-23 codes) to geoBoundaries 2021 district polygons by name within state.

Writes data/clean/india/districts.geojson (property dist_code, matched districts only) and data/clean/india/district_match.csv
(every DES district with its match score). Nothing is guessed below MIN_SCORE: unmatched districts stay unmatched and are
left out of the district climate and analysis; they are listed.

Run: python -m src.india.geo
"""
import re
import unicodedata
from difflib import SequenceMatcher

import geopandas as gpd
import pandas as pd

from ..common import paths

GEO = paths.RAW / "india_geo"
OUT = paths.CLEAN_REAL / "india"
MIN_SCORE = 0.82


# Known renames since the 2021 polygons (DES 2022-23 name -> geoBoundaries name). Only unambiguous one-to-one renames; a district
# that was split or merged (e.g. Purba Bardhaman vs the undivided Barddhaman polygon) is NOT aliased and stays unmatched.
ALIASES = {
    "Bengaluru Urban": "Bangalore", "Bengaluru Rural": "Bangalore Rural",
    "Bangalore (Urban)": "Bangalore", "Bangalore (Rural)": "Bangalore Rural",
    "Y.S.R. Kadapa": "Kadapa(YSR)", "Dangs": "The Dangs", "Gurugram": "Gurgaon", "Belagavi": "Belgaum", "Vijayapura": "Bijapur",
    "Kalaburagi": "Gulbarga", "Narmadapuram": "Hoshangabad", "Ahilyanagar": "Ahmadnagar", "Chhatrapati Sambhajinagar": "Aurangabad",
    "Dharashiv": "Osmanabad", "Raigad": "Raigarh", "Balasore": "Baleshwar", "Sonepur": "Subarnapur", "Sri Muktsar Sahib": "Muktsar",
    "Prayagraj": "Allahabad", "Ayodhya": "Faizabad", "Amroha": "Jyotiba Phule Nagar", "Hathras": "Mahamaya Nagar",
    "Bhadohi": "Sant Ravidas Nagar (Bhadohi)", "Howrah": "Haora", "North 24 Parganas": "North Twenty Four Parganas",
    "South 24 Parganas": "South Twenty Four Parganas", "Sribhumi": "Karimganj", "Gangtok": "East District", "Mangan": "North District",
    "Namchi": "South District", "Gyalshing": "West District",
}


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\(.*?\)|[^a-z ]", " ", s)
    s = re.sub(r"\b(district|dist|zila|city)\b", " ", s)
    # common transliteration variants
    s = s.replace("w", "v").replace("ph", "f").replace("kh", "k").replace("gh", "g").replace("dh", "d").replace("bh", "b")
    s = re.sub(r"(.)\1", r"\1", s)               # double letters
    s = re.sub(r"[aeiouy]", "", s)               # vowels are the main source of spelling differences
    return re.sub(r"\s+", "", s)


def norm_state(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = s.replace("&", "and")
    return re.sub(r"[^a-z]", "", s).replace("jammuandkashmir", "jammukashmir")


def load_polygons():
    """geoBoundaries ADM2 with the state (`st`) each polygon lies in and a normalised name (`nm`)."""
    g2 = gpd.read_file(GEO / "geoBoundaries-IND-ADM2.geojson")
    g1 = gpd.read_file(GEO / "geoBoundaries-IND-ADM1.geojson")
    g1["st"] = g1.shapeName.map(norm_state)
    pts = g2.copy()
    pts["geometry"] = g2.geometry.representative_point()
    pts = gpd.sjoin(pts, g1[["st", "geometry"]], how="left", predicate="within")
    g2["st"] = pts["st"].reindex(g2.index).values
    # points on a state border / sea fall outside every state polygon: nearest state
    miss = g2.st.isna()
    if miss.any():
        near = gpd.sjoin_nearest(pts.loc[miss].drop(columns=["st", "index_right"], errors="ignore").to_crs("EPSG:7755"),
                                 g1[["st", "geometry"]].to_crs("EPSG:7755"), how="left")
        g2.loc[miss, "st"] = near.groupby(near.index).st.first()
    g2["nm"] = g2.shapeName.map(norm)
    assert g2.st.notna().all()
    return g2


def match_to_polygons(df, g2, key, state_col, name_col, aliases=ALIASES):
    """One-to-one greedy best-first match of rows of `df` (unique `key`) to polygons by name within state.
    Returns df[key] with poly_idx, score (NaN where nothing reaches MIN_SCORE)."""
    d = df[[key, state_col, name_col]].copy()
    d["st"] = d[state_col].map(norm_state)
    d["nm"] = d[name_col].map(lambda n: norm(aliases.get(n, n)))
    cand = []
    for _, r in d.iterrows():
        for i, p in g2[g2.st == r.st].iterrows():
            cand.append((SequenceMatcher(None, r.nm, p.nm).ratio(), r[key], i))
    rows, used, done = [], set(), set()
    for score, k, i in sorted(cand, reverse=True):
        if score < MIN_SCORE or k in done or i in used:
            continue
        rows.append((k, i, score))
        used.add(i)
        done.add(k)
    m = pd.DataFrame(rows, columns=[key, "poly_idx", "score"])
    out = df[[key]].merge(m, on=key, how="left")
    assert out[key].is_unique
    return out


def main():
    des = pd.read_csv(OUT / "districts.csv", dtype={"dist_code": str})
    g2 = load_polygons()
    mm = match_to_polygons(des, g2, "dist_code", "state_name", "dist_name")
    des["nm"] = des.dist_name.map(lambda n: norm(ALIASES.get(n, n)))
    m = mm.dropna(subset=["poly_idx"]).copy()
    m["poly_idx"] = m.poly_idx.astype(int)
    m["poly_name"] = g2.loc[m.poly_idx, "shapeName"].values
    m["poly_id"] = g2.loc[m.poly_idx, "shapeID"].values
    res = des.merge(m, on="dist_code", how="left")
    res["matched"] = res.poly_idx.notna()
    res.drop(columns=["nm", "poly_idx"]).to_csv(OUT / "district_match.csv", index=False)

    gm = g2.loc[m.poly_idx].copy()
    gm["dist_code"] = m.dist_code.values
    gm[["dist_code", "geometry"]].to_file(OUT / "districts.geojson", driver="GeoJSON")

    assert m.dist_code.is_unique and m.poly_idx.is_unique
    print(f"DES districts {len(des)}, polygons {len(g2)}; matched {len(m)} ({len(m) / len(des):.1%}); "
          f"exact name {(m.poly_name.map(norm) == m.dist_code.map(des.set_index('dist_code').nm)).sum()}")
    for col in ("stable", "core"):
        sub = res[res[col]]
        print(f"{col}: {int(sub.matched.sum())} of {len(sub)} matched")
    low = res[res.matched & (res.score < 0.92)][["dist_code", "dist_name", "poly_name", "score", "state_name"]].sort_values("score")
    print(f"matches with score < 0.92 (check by eye): {len(low)}\n", low.head(25).round(2).to_string(index=False))
    un = res[~res.matched & res.stable][["dist_code", "dist_name", "state_name"]]
    print(f"unmatched stable districts: {len(un)}\n", un.to_string(index=False))


if __name__ == "__main__":
    main()
