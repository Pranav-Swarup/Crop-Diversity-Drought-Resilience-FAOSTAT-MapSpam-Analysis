"""Render findings and figure links directly from the analysis tables. Author: Pmjarvis."""

import json

import pandas as pd

from .storage import BASE, RESULT


def report():
    lines = [
        "# Post-presentation findings",
        "",
        "Pmjarvis conducted the follow-up analyses and verification. The figures refer back to the delivered presentation. The tables contain unrounded values, sample sizes and units. [Methods and sources](README.md) · [Presentation freeze](baseline.csv)",
        "",
        "## Growing-season drought",
        "",
    ]
    if (RESULT / "figure_01_drought.csv").exists():
        t = pd.read_csv(RESULT / "figure_01_drought.csv").set_index("measure")
        n = t.loc["season"]
        old = t.loc["spei12_w"]
        matched = t.loc["december12"]
        cy = pd.read_csv(RESULT / "season_national_year.csv")
        loss = int((cy.december12.le(-1) & cy.season.gt(-1)).sum())
        gain = int((cy.december12.gt(-1) & cy.season.le(-1)).sum())
        coverage = pd.read_csv(RESULT / "season_area_coverage.csv")
        lines.extend(
            [
                f"Across **{int(n.n_country_years):,} country-years in {int(n.n_countries)} countries**, the presented annual definition identifies **{int(old.n_drought)}** drought years. December exposure on the matched crop basket identifies **{int(matched.n_drought)}**. Using a fixed six months to harvest identifies **{int(t.loc['harvest6', 'n_drought'])}**. Using those crops’ planting-to-harvest windows identifies **{int(n.n_drought)}**. On the matched basket, {loss} years leave the drought group and {gain} enter it. The median country’s matched crops account for {100 * coverage.share_all_cropland.median():.1f}% of its mapped physical crop area. [Country-year exposures](results/season_national_year.csv) · [Coverage](results/season_area_coverage.csv)",
                "",
                f"At mean crop diversity, the estimated output difference in a drought year is **{old.output_difference_pct_trend:.2f}%** of trend under the presented exposure and **{n.output_difference_pct_trend:.2f}%** under the seasonal exposure (95% interval {n.lo_pct:.2f}% to {n.hi_pct:.2f}%). The output series is identical in these estimates. The seasonal measure changes which years count as dry. [Drought comparison figure](figures/01_calendar_drought.png) · [Figure data](results/figure_01_drought.csv) · [Panel estimates](results/season_loss_panel.csv)",
                "",
                "Each point in the paired figure compares estimates on the same crop or country. Moving the drought window changes crop yield slopes and the event years used for resistance and recovery. Countries with no usable event under either definition cannot contribute a paired event estimate. [Response figure](figures/02_calendar_responses.png) · [Figure data](results/figure_02_responses.csv) · [Country results](results/season_faces.csv)",
                "",
            ]
        )
    else:
        lines += ["Awaiting verified seasonal results.", ""]
    lines += ["## Daily heat and observed vegetation", ""]
    if (RESULT / "figure_03_heat.csv").exists():
        t = pd.read_csv(RESULT / "figure_03_heat.csv")
        pair = t[(t.model == "ERA5_30_season") & (t.term == "compound")].set_index("outcome")
        out = pair.loc["anom"]
        veg = pair.loc["vci"]
        ref = t[
            (t.model == "presented_CRU_Dec12") & (t.term == "compound") & (t.outcome == "anom")
        ].iloc[0]
        published = t[
            (t.model == "presented_full_sample") & (t.term == "compound") & (t.outcome == "anom")
        ].iloc[0]
        heat_only_change = t[
            (t.model == "ERA5_30_Dec12") & (t.term == "compound") & (t.outcome == "anom")
        ].iloc[0]
        flags = pd.read_csv(RESULT / "heat_year_types.csv")
        old_compound = flags.presented_CRU_Dec12.eq("compound")
        new_compound = flags.ERA5_30_season.eq("compound")
        shared = int((old_compound & new_compound).sum())
        lines += [
            f"The full presented model identifies {int(published.n_exposed)} compound years in {int(published.n_obs):,} country-years, with an estimated {published.beta_plot:.2f}% output difference. The common sample includes **{int(out.n_obs):,} country-years across {int(out.n_countries)} countries**. The presented CRU/December definitions identify {int(ref.n_exposed)} hot-and-dry years in that sample. Growing-season ERA5 degree-days above 30°C and seasonal drought identify **{int(out.n_exposed)}**.",
            "",
            f"On the same country-years, the presented definitions estimate a **{ref.beta_plot:.2f}%** compound-year output difference. Switching the heat measure alone identifies {int(heat_only_change.n_exposed)} compound years and gives **{heat_only_change.beta_plot:.2f}%**. With both seasonal measures, {shared} country-years qualify under both the original and new definitions, {int((old_compound & ~new_compound).sum())} leave the compound group and {int((~old_compound & new_compound).sum())} enter it. [Paired year classifications](results/heat_year_type_transitions.csv)",
            "",
            f"Compared with years meeting neither threshold, the estimated compound-year output difference is **{out.beta_plot:.2f}% of trend** (95% interval {out.ci_lo_plot:.2f} to {out.ci_hi_plot:.2f}). The corresponding satellite greenness difference is **{veg.beta_plot:.2f} VCI points** (95% interval {veg.ci_lo_plot:.2f} to {veg.ci_hi_plot:.2f}). These estimates compare the same observed country-years across models. NOAA’s 2010 crop mask and the 2020 heat footprint differ within countries. [Heat and greenness figure](figures/03_heat_and_vegetation.png) · [Figure data](results/figure_03_heat.csv) · [All heat thresholds](results/heat_effects.csv) · [Country-year measurements](results/heat_country_year.csv) · [Changed classifications](results/heat_year_type_transitions.csv) · [Satellite coverage and exclusions](results/heat_sample_coverage.csv)",
            "",
        ]
    else:
        lines += [
            "The weekly vegetation study is complete. The hourly heat estimates require verification of the remaining ERA5 measurements.",
            "",
        ]
    if (RESULT / "vci_footprint_agreement.csv").exists():
        v = pd.read_csv(RESULT / "vci_footprint_agreement.csv").set_index("level")
        n = v.loc["country-year"]
        c = v.loc["country-crop-year"]
        lines += [
            f"On the shared crop footprint, **{int(n.n)} country-years across {int(n.n_countries)} countries** can be compared directly in 2020–2023. The two seasonal greenness series have correlation **{n.pearson:.3f}** and differ by **{n.mean_absolute_difference:.2f} VCI points** on average in absolute value. Across {int(c.n):,} individual country-crop-years, the mean absolute difference is {c.mean_absolute_difference:.2f} points. The national average hides some crop-level disagreement. [Satellite footprint figure](figures/03b_vegetation_footprint.png) · [Figure data](results/figure_03b_vegetation.csv) · [Agreement statistics](results/vci_footprint_agreement.csv)",
            "",
        ]
    lines += ["## Power for extreme droughts", ""]
    if (RESULT / "figure_04_extremes.csv").exists():
        t = pd.read_csv(RESULT / "figure_04_extremes.csv").set_index("model")
        original = t.loc["presented_1993_2023"]
        short = t.loc["common_1993_2023"]
        long = t.loc["common_1961_2023"]
        allc = t.loc["eligible_1961_2023"]
        lines += [
            f"The historical reader reproduces the presented **{int(original.n_extreme)} extreme country-years in {int(original.n_exposed)} countries**, the model coefficients and confidence limits, and the studentized country-permutation p-value ({original.p_perm_t:.3f}). On the **same {int(short.n_countries)} countries**, adding 1961–1992 raises the number of extreme observations from **{int(short.n_extreme)} to {int(long.n_extreme)}**. The approximate 80%-power detectable interaction falls from {short.mde_pp:.2f} to {long.mde_pp:.2f} percentage points of output trend per standard deviation of diversity.",
            "",
            f"The full 1961–2023 eligibility rules admit **{int(allc.n_countries)} countries**, producing **{int(allc.n_extreme)}** extreme observations. The diversity interaction is **{allc.effect_pp:+.2f} percentage points** per standard deviation (95% interval {allc.lo_pp:.2f} to {allc.hi_pp:.2f}, studentized permutation p = {allc.p_perm_t:.3f}). A positive interaction means the estimated extreme-year output difference is less negative, or more positive, at higher diversity. The common-country comparison holds membership fixed. The tighter long-window eligibility rules explain how the full historical cohort can have fewer extreme years than the presented 107-country cohort. [Extreme-drought figure](figures/04_extreme_drought_power.png) · [Figure data](results/figure_04_extremes.csv) · [Full estimates](results/history_severity.csv) · [Eligibility](results/history_eligibility.csv)",
            "",
        ]
        if (RESULT / "history_sensitivity.csv").exists():
            lines += [
                "Additional estimates hold the diversity measurement period and approximate smoothing span fixed on the same countries. [Window sensitivity](results/history_sensitivity.csv)",
                "",
            ]
    lines += ["## Official Indian district data", ""]
    status = RESULT / "district_status.json"
    if status.exists():
        info = json.loads(status.read_text())
        lines += [
            info["summary"]
            + " [Official source checks](results/district_status.json) · [Presented district benchmarks](results/district_presented_benchmarks.csv) · [Official release inputs](DISTRICT.md)",
            "",
        ]
        if info.get("status") == "complete":
            lines += [
                "[District comparisons](results/district_comparison.csv) · [District figure](figures/05_district_comparison.png)",
                "",
            ]
        elif info.get("status") == "archived_complete":
            fits = pd.read_csv(RESULT / "district_comparison.csv").query('term == "spei_z"').set_index("model")
            lines += [
                "The interaction below measures how much the output response to a one-unit increase in SPEI (wetter conditions) changes per standard deviation of prior crop diversity. Values are percentage points of output trend. A drier change of the same size reverses the sign.",
                "",
                "| Sample | Districts | District-years | Interaction | 95% interval |",
                "|---|---:|---:|---:|---:|",
            ]
            labels = {"core:core": "Presented DES, 1998–2022", "ext5:core": "Presented five crops, 1993–2015",
                      "archive": "ICRISAT, 1966–2017", "archive_irrigation_sample": "Same period, irrigation sample",
                      "archive_crop_irrigation": "Same sample, adjusted for crop irrigation", "overlap": "ICRISAT, 1998–2017",
                      "overlap_irrigation_sample": "Later period, irrigation sample",
                      "overlap_crop_irrigation": "Later sample, adjusted for crop irrigation"}
            for model, label in labels.items():
                row = fits.loc[model]
                lines.append(f"| {label} | {int(row.n_districts)} | {int(row.n_obs):,} | {100*row.beta:+.2f} | {100*row.ci_lo:+.2f} to {100*row.ci_hi:+.2f} |")
            agr = pd.read_csv(RESULT / "district_quantity_agreement.csv")
            irr = agr.loc[agr.measure.eq("irrigated area (ha)") & agr.crop.eq("All comparable crops")].iloc[0]
            prices = pd.read_csv(RESULT / "district_price_sensitivity.csv").query('term == "spei_z"').set_index("model")
            latest = prices.loc["district_2015"]
            intl = prices.loc["presented_fao"]
            published = pd.read_csv(RESULT / "district_manual_sensitivity.csv").query('term == "spei_z"').set_index("model").loc["as_printed_crop_irrigation"]
            cov = pd.read_csv(RESULT / "district_coverage.csv").query('eligible and window == "archive"')
            farm = pd.read_csv(RESULT / "district_farm_price_agreement.csv")
            lines += [
                "",
                "The long and later windows give opposite point estimates. Every interval includes zero. These samples do not establish whether more diverse districts consistently lose less output in drought. The new rows also change the crop basket, district geography and time window relative to the presentation. [Model comparison figure](figures/05_district_comparison.png) · [All estimates and sample sizes](results/district_comparison.csv)",
                "",
                f"The long-window crop basket covers a median **{100*cov.balanced_area_share.median():.2f}%** of reported hectares within the 19 crops. Using the values that conflict with the publisher’s manual expands the adjusted model to {int(published.n_districts)} districts and gives an interaction of {100*published.beta:+.2f} points (95% interval {100*published.ci_lo:+.2f} to {100*published.ci_hi:+.2f}). [Coverage and exclusions](results/district_coverage.csv) · [Manual sensitivity](results/district_manual_sensitivity.csv)",
                "",
                f"Holding {int(latest.n_districts)} districts, {int(latest.n_obs):,} district-years, crop quantities and the weather fixed, the estimated interaction is **{100*latest.beta:+.2f} points** at 2015 district farm prices and **{100*intl.beta:+.2f}** at the presented FAOSTAT prices. This isolates price weighting from the other sample differences. [Price comparison figure](figures/05c_district_price_comparison.png) · [Four price specifications](results/district_price_sensitivity.csv)",
                "",
                f"All **{info['source_cells_checked']:,} numerical entries** in the archived official crop export match the independently deposited World Bank copy. In the separate comparison of apportioned and unapportioned farm-price series, {int(farm.identical.sum()):,} of {int(farm.joint_prices.sum()):,} joint positive prices agree exactly. [Crop-file audit](results/district_source_reconciliation.csv) · [Price audit](results/district_farm_price_agreement.csv)",
                "",
                f"Of {info['modern_aligned_districts']} aligned historical and modern districts, {int(irr.districts)} contribute positive irrigation pairs from both agricultural publications. **{int(irr.within_tolerance):,} of {int(irr.positive_pairs):,} ({100*irr.within_tolerance/irr.positive_pairs:.1f}%)** agree within 10 hectares or 1% of the government comparator, whichever is larger. The median absolute difference is {irr.median_absolute_difference:g} ha and the largest is {irr.max_absolute_difference:,.0f} ha. Jaisalmer chickpea in 2013–14 is the largest discrepancy. No historical and modern comparison polygon is identical. [Quantity agreement figure](figures/05b_district_quantity_agreement.png) · [Values by crop and source](results/district_quantity_agreement.csv) · [Boundary sensitivity](results/district_boundary_sensitivity.csv) · [Methods, sources and limitations](DISTRICT.md)",
                "",
            ]
    else:
        lines += [
            "The last study requires the latest official apportioned release. [District inputs and baseline comparators](README.md#official-indian-district-data)",
            "",
        ]
    lines += [
        "## Verification",
        "",
        "The baseline manifest verifies the original figures, statistics, code, data, slides and explorer. The extension’s tests reconcile source identities, calendar bounds, exposure and country coverage, model estimates and plotted values. [Freeze manifest](baseline.csv) · [Source checksums](source_files.csv.gz) · [Follow-up output checksums](outputs.csv) · [Documentation inventory](DOCUMENTATION.md)",
        "",
    ]
    (BASE / "FINDINGS.md").write_text("\n".join(lines))


if __name__ == "__main__":
    report()
