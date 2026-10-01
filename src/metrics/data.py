"""Country-level analysis table: metrics + region + window means of the controls."""
import numpy as np
import pandas as pd

from ..common import paths
from ..common.contract import load

FACES = ["stability", "resistance", "recovery", "vulnerability"]
PREDICTORS = ["hill1", "resp_div"]
CONTROLS = ["log_gdp_pc", "fert_kg_ha", "irrig_share"]


def analysis_table(metrics=None):
    """One row per country: contract metrics, name, region, log_gdp_pc, fert_kg_ha, irrig_share
    (controls are means over the window), share_imputed."""
    metrics = load("metrics") if metrics is None else metrics
    controls = load("controls").groupby("iso3")[["gdp_pc", "fert_kg_ha", "irrig_share"]].mean()
    controls["log_gdp_pc"] = np.log(controls["gdp_pc"])
    df = metrics.merge(load("countries")[["iso3", "name", "region"]], on="iso3", how="left") \
                .merge(controls[CONTROLS].reset_index(), on="iso3", how="left") \
                .merge(load("quality")[["iso3", "share_imputed"]], on="iso3", how="left")
    assert df["region"].notna().all(), "metrics has countries missing from countries.csv"
    return df


def source_line(n):
    """Caption source line for figures."""
    if paths.USE_DUMMY:
        return f"SYNTHETIC DUMMY DATA (src.common.make_dummy), not results. n = {n} countries."
    return (f"Source: FAOSTAT (QCL, QV), SPEIbase v2.10, MapSPAM 2020, World Bank WDI. "
            f"1993–2023, n = {n} countries.")
