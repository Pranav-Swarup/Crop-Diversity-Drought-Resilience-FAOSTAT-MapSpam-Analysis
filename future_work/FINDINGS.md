# Post-presentation findings

Pmjarvis conducted the follow-up analyses and verification. The figures refer back to the delivered presentation. The tables contain unrounded values, sample sizes and units. [Methods and sources](README.md) · [Presentation freeze](baseline.csv)

## Growing-season drought

Across **3,317 country-years in 107 countries**, the presented annual definition identifies **626** drought years. December exposure on the matched crop basket identifies **622**. Using a fixed six months to harvest identifies **525**. Using those crops’ planting-to-harvest windows identifies **524**. On the matched basket, 225 years leave the drought group and 127 enter it. The median country’s matched crops account for 73.5% of its mapped physical crop area. [Country-year exposures](results/season_national_year.csv) · [Coverage](results/season_area_coverage.csv)

At mean crop diversity, the estimated output difference in a drought year is **-4.41%** of trend under the presented exposure and **-5.48%** under the seasonal exposure (95% interval -6.78% to -4.18%). The output series is identical in these estimates. The seasonal measure changes which years count as dry. [Drought comparison figure](figures/01_calendar_drought.png) · [Figure data](results/figure_01_drought.csv) · [Panel estimates](results/season_loss_panel.csv)

Each point in the paired figure compares estimates on the same crop or country. Moving the drought window changes crop yield slopes and the event years used for resistance and recovery. Countries with no usable event under either definition cannot contribute a paired event estimate. [Response figure](figures/02_calendar_responses.png) · [Figure data](results/figure_02_responses.csv) · [Country results](results/season_faces.csv)

## Daily heat and observed vegetation

The full presented model identifies 181 compound years in 3,317 country-years, with an estimated -5.60% output difference. The common sample includes **2,771 country-years across 99 countries**. The presented CRU/December definitions identify 150 hot-and-dry years in that sample. Growing-season ERA5 degree-days above 30°C and seasonal drought identify **194**.

On the same country-years, the presented definitions estimate a **-5.68%** compound-year output difference. Switching the heat measure alone identifies 187 compound years and gives **-8.07%**. With both seasonal measures, 84 country-years qualify under both the original and new definitions, 66 leave the compound group and 110 enter it. [Paired year classifications](results/heat_year_type_transitions.csv)

Compared with years meeting neither threshold, the estimated compound-year output difference is **-8.48% of trend** (95% interval -10.88 to -6.07). The corresponding satellite greenness difference is **-2.64 VCI points** (95% interval -4.13 to -1.14). These estimates compare the same observed country-years across models. NOAA’s 2010 crop mask and the 2020 heat footprint differ within countries. [Heat and greenness figure](figures/03_heat_and_vegetation.png) · [Figure data](results/figure_03_heat.csv) · [All heat thresholds](results/heat_effects.csv) · [Country-year measurements](results/heat_country_year.csv) · [Changed classifications](results/heat_year_type_transitions.csv) · [Satellite coverage and exclusions](results/heat_sample_coverage.csv)

On the shared crop footprint, **404 country-years across 101 countries** can be compared directly in 2020–2023. The two seasonal greenness series have correlation **0.964** and differ by **2.33 VCI points** on average in absolute value. Across 3,940 individual country-crop-years, the mean absolute difference is 4.13 points. The national average hides some crop-level disagreement. [Satellite footprint figure](figures/03b_vegetation_footprint.png) · [Figure data](results/figure_03b_vegetation.csv) · [Agreement statistics](results/vci_footprint_agreement.csv)

## Power for extreme droughts

The historical reader reproduces the presented **58 extreme country-years in 41 countries**, the model coefficients and confidence limits, and the studentized country-permutation p-value (0.165). On the **same 75 countries**, adding 1961–1992 raises the number of extreme observations from **30 to 49**. The approximate 80%-power detectable interaction falls from 7.50 to 4.89 percentage points of output trend per standard deviation of diversity.

The full 1961–2023 eligibility rules admit **78 countries**, producing **50** extreme observations. The diversity interaction is **+5.19 percentage points** per standard deviation (95% interval 1.91 to 8.48, studentized permutation p = 0.017). A positive interaction means the estimated extreme-year output difference is less negative, or more positive, at higher diversity. The common-country comparison holds membership fixed. The tighter long-window eligibility rules explain how the full historical cohort can have fewer extreme years than the presented 107-country cohort. [Extreme-drought figure](figures/04_extreme_drought_power.png) · [Figure data](results/figure_04_extremes.csv) · [Full estimates](results/history_severity.csv) · [Eligibility](results/history_eligibility.csv)

Additional estimates hold the diversity measurement period and approximate smoothing span fixed on the same countries. [Window sensitivity](results/history_sensitivity.csv)

## Official Indian district data

Archived ICRISAT/TCI data cover 311 parent districts. The 1966–2017 analysis values production at observed 2015 farm-harvest prices and admits 176 districts. The live portal’s latest apportioned edition could not be independently established. [Official source checks](results/district_status.json) · [Presented district benchmarks](results/district_presented_benchmarks.csv) · [Official release inputs](DISTRICT.md)

The interaction below measures how much the output response to a one-unit increase in SPEI (wetter conditions) changes per standard deviation of prior crop diversity. Values are percentage points of output trend. A drier change of the same size reverses the sign.

| Sample | Districts | District-years | Interaction | 95% interval |
|---|---:|---:|---:|---:|
| Presented DES, 1998–2022 | 308 | 6,160 | +0.34 | -0.69 to +1.37 |
| Presented five crops, 1993–2015 | 127 | 2,286 | +0.85 | -0.99 to +2.69 |
| ICRISAT, 1966–2017 | 176 | 8,272 | -0.67 | -2.20 to +0.86 |
| Same period, irrigation sample | 149 | 4,667 | -1.03 | -2.78 to +0.72 |
| Same sample, adjusted for crop irrigation | 149 | 4,667 | -1.20 | -2.85 to +0.45 |
| ICRISAT, 1998–2017 | 193 | 2,895 | +1.46 | -0.08 to +3.00 |
| Later period, irrigation sample | 144 | 1,461 | +1.26 | -0.27 to +2.80 |
| Later sample, adjusted for crop irrigation | 144 | 1,461 | +0.51 | -1.47 to +2.49 |

The long and later windows give opposite point estimates. Every interval includes zero. These samples do not establish whether more diverse districts consistently lose less output in drought. The new rows also change the crop basket, district geography and time window relative to the presentation. [Model comparison figure](figures/05_district_comparison.png) · [All estimates and sample sizes](results/district_comparison.csv)

The long-window crop basket covers a median **56.35%** of reported hectares within the 19 crops. Using the values that conflict with the publisher’s manual expands the adjusted model to 162 districts and gives an interaction of -0.85 points (95% interval -2.32 to +0.61). [Coverage and exclusions](results/district_coverage.csv) · [Manual sensitivity](results/district_manual_sensitivity.csv)

Holding 110 districts, 5,170 district-years, crop quantities and the weather fixed, the estimated interaction is **-1.33 points** at 2015 district farm prices and **-1.51** at the presented FAOSTAT prices. This isolates price weighting from the other sample differences. [Price comparison figure](figures/05c_district_price_comparison.png) · [Four price specifications](results/district_price_sensitivity.csv)

All **1,210,950 numerical entries** in the archived official crop export match the independently deposited World Bank copy. In the separate comparison of apportioned and unapportioned farm-price series, 55,558 of 55,629 joint positive prices agree exactly. [Crop-file audit](results/district_source_reconciliation.csv) · [Price audit](results/district_farm_price_agreement.csv)

The government land-use CSV agrees with the portal datastore across **7,865,000 fields in 605,000 rows** when joined by record ID. This verifies the downloaded observations, including blanks. The comparison with ICRISAT below addresses whether the two agricultural publications actually give the same hectares. [Datastore queries and page digests](district_land_use_pages.csv)

Of 27 aligned historical and modern districts, 22 contribute positive irrigation pairs from both agricultural publications. **2,130 of 2,252 (94.6%)** agree within 10 hectares or 1% of the government comparator, whichever is larger. The median absolute difference is 2 ha and the largest is 141,976 ha. Jaisalmer chickpea in 2013–14 is the largest discrepancy. No historical and modern comparison polygon is identical. [Quantity agreement figure](figures/05b_district_quantity_agreement.png) · [Values by crop and source](results/district_quantity_agreement.csv) · [Boundary sensitivity](results/district_boundary_sensitivity.csv) · [Methods, sources and limitations](DISTRICT.md)

## Verification

The working-tree check verifies the presented figures, statistics, code, data, delivered slide PDF and explorer. The historical snapshot preserves all original slide versions. The extension’s tests reconcile source identities, calendar bounds, exposure and country coverage, model estimates and plotted values. [Freeze manifest](baseline.csv) · [Source checksums](source_files.csv.gz) · [Follow-up output checksums](outputs.csv) · [Documentation inventory](DOCUMENTATION.md)
