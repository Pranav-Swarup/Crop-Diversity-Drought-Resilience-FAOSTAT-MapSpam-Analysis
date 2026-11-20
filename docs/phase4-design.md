# Phase 4 design: does crop diversity cause drought resilience? Evidence from Indian districts

## Status after the presented Phase 4 study

This document preserves the design used for the presented DES and five-crop results. Pmjarvis’s follow-up estimates the same type of diversity interaction on the ICRISAT/TCI 1966 parent geography, using historical farm prices and prior-year crop irrigation. It also compares agricultural quantities on geographically aligned units and tests four price schedules on the same observations. [Follow-up findings](../future_work/FINDINGS.md#official-indian-district-data) · [District sources and boundary rules](../future_work/DISTRICT.md)


This is the original Phase 4 proposal. The district pipeline shipped in November 2026. It uses the DES core sample and a separate five-crop ICRISAT mirror sample. The methods note records the adopted sources, periods, boundary matching and remaining limitations. [Implemented methods](../src/india/README.md) · [District results](../outputs/stats/india_panel.csv)

## 1. Why this phase

Phases 1–3 compare about 107 countries. Every key result is cross-sectional: diverse countries also tend to be richer, larger, more irrigated and better at reporting, and `hill1` barely changes within a country, so the panel only identifies `spei × hill1` against a time-invariant `hill1`. Three weaknesses follow:

1. **Correlation, not cause.** The diversity effect cannot be separated from what goes with diversity.
2. **Scale.** National totals average out regional failures, so the unit is not where farming decisions or droughts happen.
3. **Power.** The extreme-drought test in phase 3 (`severity.py`) rests on 58 extreme country-years (14 in the most diverse tercile); the extreme × `hill1` term is +0.042 with `p_perm_t` = 0.16.

Phase 4 goes deep instead of wide. Indian districts have long crop records, and their crop mixes **change over time** (irrigation expansion, the shift from coarse cereals to rice and wheat, pulses policy). Comparing a district with itself before and after its mix changed is the comparison the national data cannot make.

**Research question.** Within Indian districts, does a more diverse crop mix in a given year reduce the production loss from drought in that year, once district and state-year effects and irrigation are held fixed?

## 2. Goals, non-goals, anti-goals

**Goals**
- A district-year panel for India in which crop diversity varies over time, so both `hill1` and `spei × hill1` are identified within districts.
- The four faces of resilience per district, computed with the existing `src/metrics/` functions, and a comparison of district-level and national-level results.
- A severity test (moderate / severe / extreme) with enough extreme district-years to have power.
- An explicit test of irrigation as the main confounder.

**Non-goals**
- A global causal design (natural experiments across countries).
- The global trade-failure simulation (dropped; see section 8).
- Policy advice. Results are evidence, not recommendations.

**Anti-goals** (failure even if the code runs)
- Any gap-filled, interpolated or substituted district data (hard rule 1 of `CLAUDE.md`).
- Changing the national analysis to match the district one.
- Claiming the India result holds globally. It is one country; it shows a mechanism, not generality.

## 3. Approaches considered

| Approach | What it is | Trade-off | Decision |
|---|---|---|---|
| A. Breadth: India districts + 2050 projections + global trade cascade | Three new strands at once | Carries the phase 1–3 correlational weakness forward into projections and simulations; heavy assumptions in the trade model | Not chosen |
| B. **Depth: India district panel with time-varying diversity** | One identification-focused analysis; projections only as a small closing scenario | One country only; data needs a manual, registered download | **Chosen** |
| C. Global gridded yields (GDHY) | Same idea on a global grid | GDHY ends in 2016 (cannot cover the window); grid yields are partly modelled | Not chosen |

## 4. Data

### 4.1 ICRISAT District Level Database (DLD)

Source: ICRISAT–TCI District Level Database for Indian Agriculture, `http://data.icrisat.org/dld/`. Registration required, so it is downloaded **by hand by a team member**. The site could not be reached when this was written; the module and year details below are **unverified** and must be confirmed from the downloaded files.

Put the files in `data/raw/icrisat/` (gitignored), unchanged.

| Priority | Module | Why |
|---|---|---|
| Essential | Crop area, production, yield; all crops, states, years. **Apportioned** (1966 boundaries) if offered; unapportioned too if available, as a cross-check | Outcome and diversity. Fixed boundaries keep series continuous (same idea as `BLX`/`SCG`/`SDX`) |
| Essential | Irrigated area by crop | Irrigation is the main confounder |
| Essential | District boundaries matching the crop data | District-level SPEI. If ICRISAT has none, **stop and decide as a team**; no silent substitute |
| Essential | Documentation: definitions, units, codes, missing-value codes, apportioning method, terms of use, citation | Assertions, `SOURCES.md` |
| Strongly wanted | Farm harvest prices (district × crop) | Value weighting (the project never sums tonnes) |
| Useful | Land use, fertiliser, HYV area | Controls |
| Useful | Monthly rainfall | Sanity check of SPEI only, never a replacement |

After download, report: exact year range per file, number of states and districts, apportioned or not, file format, download date, terms of use.

**Download rule.** `CLAUDE.md` requires every download to be a script. Because of the login, the script `src/india/register_icrisat.py` instead checks the expected files exist, records checksums and download date, and writes the source entry to `data/raw/SOURCES.md`. **[team]** accept this exception.

### 4.2 Climate and cropland (existing, gridded)

`data/raw/spei/` (SPEIbase v2.10) and `data/raw/mapspam/` exist only on the machine that ran `src.climate` (Prathmesh). District-level drought is computed there, with the same method as `spei12_w` but on district polygons.

### 4.3 Window **[team]**

SPEIbase ends in December 2023; ICRISAT is expected to end earlier. Proposal:
- **Core window:** 1993 to the last ICRISAT year, for comparability with the national analysis.
- **Extended window:** from the first ICRISAT year (possibly 1966), for power. Reported separately, never blended with the core.

## 5. Definitions (proposals)

- **Unit and keys:** district on fixed (apportioned) boundaries; keys `dist_code` (str), `year` (int); `state_code` kept.
- **Yield:** `prod_t / area_ha`, computed by us.
- **Value:** `prod_t × price`, with one constant price per crop (median district farm harvest price over a base period, deflated to a fixed year), mirroring `value_const`. If prices are missing or too sparse, fall back to area weighting for the faces and report it. **[team]** base period.
- **Detrending, anomalies, index:** unchanged (`src.metrics.detrend`, LOWESS `frac=0.5`).
- **Diversity:**
  - `hill1` = window-mean area shares, for the faces (same as national).
  - `hill1_tv` = Hill-1 of area shares over the previous 5 years (`t−5..t−1`), lagged so the current year's planting response to drought does not enter. This is the identifying variable. Variant: 3-year window.
- **Drought:** `spei12_w` at district level (December SPEI-12, MapSPAM cropland-weighted), same thresholds and event rule as `CLAUDE.md`. Variant: kharif-season SPEI (June–September), as a robustness check only.
- **Irrigation:** `irrig_share` = irrigated area / gross cropped area, lagged one year.

## 6. Analyses

1. **District panel (core).**
   `anom_dt = β·spei_dt + γ·spei_dt × z(hill1_tv,d,t−1) + δ·z(hill1_tv,d,t−1) + controls + α_d + λ_state×year + ε_dt`.
   Standard errors clustered by district; spatially correlated errors (Conley) as robustness. State × year effects absorb monsoon-wide shocks and state policy.
   Variants:
   - add `spei × irrig_share` and `spei × fertiliser` (does diversity survive irrigation?)
   - dry/wet split
   - rain-fed-dominant districts only
   - split-sample `hill1_tv`, as in `split_resp_div`
   - leave-one-state-out
2. **Severity bins** at district level, with the same bins and inference rules as `src/metrics/severity.py` (cite `p_perm_t` for the extreme bin).
3. **Four faces per district** and cross-district Spearman against `hill1`, `resp_div` and `exp_corr`, compared with the national tables.
4. **Validation.** District drought years against India's documented drought years (2002, 2009, 2014–15; sources to be cited by the validation owner), reusing the `src/validation/` approach.
5. **Closing scenario (small, only if 1 holds).** Apply the fitted district relationship to projected 2050 SPEI from CMIP6 (ISIMIP3b, bias-adjusted). Computing SPEI from CMIP6 needs precipitation and PET variables, a separate download. Labelled a scenario, not a forecast. **[team]** scenarios and models.

## 7. Outputs (proposed contract, `src/india/`)

| File | Columns |
|---|---|
| `data/clean/india/districts.csv` | dist_code, dist_name, state_code, state_name, boundary_base |
| `data/clean/india/crops.parquet` | dist_code, year, crop, area_ha, prod_t, yield_t_ha, irr_area_ha, price, value_const, is_missing |
| `data/clean/india/climate.parquet` | dist_code, year, spei12_w, spei_kharif_w |
| `data/clean/india/districts.geojson` | property `dist_code` |
| `outputs/india/metrics.parquet` | dist_code, hill1, stability, resistance, recovery, vulnerability, vuln_drought, n_events, resp_div, phi_sync, mean_crop_cv |
| `outputs/india/panel.parquet` | dist_code, year, anom, spei12_w, hill1_tv, irrig_share |
| `outputs/stats/india_*.csv` | panel, severity, Spearman, national-vs-district comparison |

All pipelines end with assertions and a summary, as `CLAUDE.md` requires. No dummy data until the real schema is known.

## 8. What was dropped and why

- **Global trade-failure simulation:** results would mostly reflect the simulation's assumptions, and phase 3 found supply diversity does not change the effect of a country's own drought (`supply_panel.csv`, p = 0.69).
- **Global causal design:** no clean, sourced list of national crop-mix shocks exists in the project.
- **Large 2050 projection strand:** kept only as the small closing scenario above.

## 9. Milestones

0. **Review phase 2 and 3 code** by the module owners (it is unreviewed and phase 4 reuses `src/metrics/`).
1. ICRISAT registration and download; report coverage (section 4.1).
2. Team decisions marked **[team]**: window, download-script exception, price base period, module ownership.
3. District boundaries and district SPEI (on the gridded-data machine).
4. Clean district crop table with assertions.
5. Core panel and variants; severity; faces.
6. Validation.
7. Closing scenario (optional).

## 10. Open questions

- **[team]** Owner of the new `src/india/` module.
- Does ICRISAT's coverage reach far enough past 1993 for a usable core window?
- Are farm harvest prices complete enough for value weighting?
- Does ICRISAT provide boundaries, and on which base year?
- How should district splits after 1966 be handled if only unapportioned data is available?
