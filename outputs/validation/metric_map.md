# Metric map: what each metric operationalises

Definitions are copied from `CLAUDE.md`. "Concept source" lists the papers that `CLAUDE.md` or the code docstrings in `src/metrics/` name for the metric. "Paper in our Review 1" lists only papers from our own literature review, matched by what the review says about them. The Review 1 matches are a first pass and must be checked against the papers before this goes on a slide (see the notes below the table).

| Metric | Definition (from CLAUDE.md) | Concept source | Paper in our Review 1 it relates to |
|---|---|---|---|
| `hill1` | exp(Shannon entropy) of mean harvested-area shares over the window (effective number of crops) | Hill number of order 1 (effective number of species). CLAUDE.md names no paper; Hill (1973) is the usual reference, to verify | Hufnagel et al. 2020 [3] (how crop diversification is defined); Ramankutty et al. 2018 [1] (supply narrowing to a few staple crops) |
| `stability` | 1 / CV of detrended residuals of national value | Urruty et al. 2016 (stability face) | Tilman et al. 2006 [2] (variability of total output); Mahaut et al. 2021 [8] (national yield stability) |
| `resistance` | median over events of I_t / mean(I_{t-3..t-1}) (Lloret et al. 2011) | Lloret et al. 2011 (formula); Urruty et al. 2016 (face) | Urruty et al. 2016 [4]; Renard et al. 2023 [9] (buffering of drought and heat impacts) |
| `recovery` | median over events of mean(I_{t+1..t+3}) / I_t | Urruty et al. 2016 (face); CLAUDE.md cites no paper for this formula | Urruty et al. 2016 [4] |
| `vulnerability` | share of years with I_t < 0.9 (loss > 10% below trend) | Urruty et al. 2016 (face) | Urruty et al. 2016 [4] |
| `vuln_drought` | same as `vulnerability`, restricted to drought years | Urruty et al. 2016 (face), drought years only | Urruty et al. 2016 [4]; Renard et al. 2023 [9] |
| `resp_beta` (per crop) | OLS slope of crop yield anomaly on `spei12_w`; crops with >= 20 non-imputed years and >= 1% mean value share | Response diversity concept (Ross et al. 2023) | Tilman et al. 2006 [2] (species respond differently to stress); Renard et al. 2023 [9] |
| `resp_div` | value-weighted SD of `resp_beta` across a country's crops | Response diversity (Ross et al. 2023) | Tilman et al. 2006 [2] (insurance effect); Renard et al. 2023 [9] |
| `resp_divergence` | Ross et al. 2023 divergence on the same `resp_beta` slopes (CLAUDE.md says to check the paper for the exact formula) | Ross, Petchey, Sasaki & Armitage 2023, section 3.1 (Methods Ecol. Evol. 14:1150-1167, as given in `response.py`) | Concept only (Tilman et al. 2006 [2]); no paper in Review 1 uses this measure |
| `phi_sync` | Var(total) / (sum of sd_i)^2, on detrended per-crop value residuals (Loreau & de Mazancourt 2008; Thibaut & Connolly 2013) | Loreau & de Mazancourt 2008; Thibaut & Connolly 2013 | Mahaut et al. 2021 [8] (asynchrony between crops); Tilman et al. 2006 [2] (compensatory dynamics) |
| `mean_crop_cv` | sum of w_i * CV_i, with w_i = mean value share of crop i (same decomposition as `phi_sync`) | Loreau & de Mazancourt 2008; Thibaut & Connolly 2013 | Mahaut et al. 2021 [8] (stability of individual crops) |
| CV_total (identity, not a metric column) | sqrt(`phi_sync`) * `mean_crop_cv`; asserted to hold numerically | Loreau & de Mazancourt 2008; Thibaut & Connolly 2013 | Mahaut et al. 2021 [8] (crop-level stability vs asynchrony) |

Notation: I_t = y_t / trend_t is the national value index (sum of `value_const` across crops, LOWESS-detrended, frac = 0.5, 1993-2023).
A drought event is the first year of a run of years with `spei12_w` <= -1.0, with 3 drought-free years before it and 3 years after it inside the window.

## To verify before use

- **Review 1 matches:** check each against the paper itself. `resp_divergence` and `hill1` have no paper in Review 1 that uses the measure, so "concept only" or a loose match is the honest entry.
- **Urruty et al. 2016:** Review 1 cites it for separating stability, robustness, vulnerability and resilience. CLAUDE.md uses its four faces as stability, resistance, recovery and vulnerability. Confirm in the paper that these are its four faces before writing "operationalises".
- **Hill (1973):** given from memory as the usual reference for Hill numbers. Look it up before citing.
- **Papers outside Review 1** (Lloret 2011, Loreau & de Mazancourt 2008, Thibaut & Connolly 2013, Ross et al. 2023) are named in CLAUDE.md or the code only. Check their details before adding them to a reference list.
