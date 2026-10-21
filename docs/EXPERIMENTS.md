# Experiment log

Code: `src/hcw_ml/{features,models,evaluate}.py`, runner `scripts/run_experiment_01.py`.
Outputs: `reports/experiments/01_cv_results.csv` (every fold of every parameter set), `01_best_params.csv`,
`01_load_curves.csv`; figures `reports/figures/01_param_effects.png`, `01_load_curves.png` (figures are git-ignored,
regenerate with the script, about 3 minutes).

## Experiment 01 – three candidate models, first parameter sweep

**Question.** Does hospital load predict deaths per admission, and which model class fits best?

**Target (B).** deaths in weeks +1..+2 / admissions in weeks −1..0, per unit-week. Models predict the rate and are fit
with weight = admissions (equivalent to a Poisson count model with a log-admissions offset).

**Data used.** 5,984 unit-weeks with admissions: ES-NUTS3 (5,335, 49 provinces), BE (504, 3 regions), BR-Rio (145, 1 city). Excluded:
ES-comarca (overlaps the Catalan provinces, so it would double-count), IT and HU (no hospital admissions, deferred to the
stock target), TR (no hospital data).

**Features.** `rel_load` (census / the unit's own p95), `rel_icu_load`, `icu_share`, `load_growth` (week-on-week change of
`rel_load`), `panel`, and `era` (before 2021-03-01, to 2021-12-19, after: a coarse stand-in for vaccination and variant).
Past deaths are deliberately not used.

**Models** (`models.py`).
| name | what | tuned parameters |
|---|---|---|
| `glm_noload` | Poisson GLM, panel + era only (reference) | alpha {1e-4, 1e-2} |
| `glm` | ridge Poisson GLM, cubic spline of `rel_load` + panel + era | alpha {1e-5..1e-2}, spline knots {4, 5, 7} |
| `hgb` | HistGradientBoosting, Poisson loss, all features | learning rate, depth {2,3,5}, iterations {100,300}, min leaf {50,200} |
| `rf` | RandomForest, Poisson criterion, all features | min leaf {20,50,150}, max features {0.5,1}, depth {6,None} |

**Validation** (`evaluate.py`). Two schemes, both reported:
- *temporal*: rolling origin, train ≤ 2020-12-28 / 2021-06-28 / 2021-12-27, test the next 26 weeks. The p95 reference
  behind `rel_load` is refit on training weeks only.
- *unit*: 5-fold GroupKFold holding out whole provinces/regions (tests transfer to new units).

Metric: weight-averaged Poisson deviance (lower is better); also weighted MAE of the rate and bias = predicted/observed
deaths − 1. Parameters are chosen on mean deviance over folds.

### Results (mean over folds)

| scheme | model | best deviance | MAE (rate) | bias | best params |
|---|---|---|---|---|---|
| temporal | **hgb** | **0.0702** | 0.089 | +0.22 | lr 0.1, depth 2, 100 iter, leaf 50 |
| temporal | glm_noload | 0.0725 | 0.095 | +0.22 | alpha 1e-4 |
| temporal | glm | 0.0763 | 0.098 | +0.23 | alpha 1e-4, 7 knots |
| temporal | rf | 0.0769 | 0.099 | +0.29 | leaf 20, mf 0.5, depth 6 |
| unit | **hgb** | **0.0765** | 0.097 | −0.03 | lr 0.1, depth 5, 300 iter, leaf 50 |
| unit | rf | 0.0766 | 0.095 | −0.01 | leaf 20, mf 1.0, depth None |
| unit | glm | 0.0940 | 0.105 | −0.04 | alpha 1e-4, 4 knots |
| unit | glm_noload | 0.0947 | 0.106 | −0.05 | alpha 1e-4 |

Spread across all parameter sets (min–max deviance): temporal glm 0.076–0.081, hgb 0.070–0.086, rf 0.077–0.105; unit
glm 0.094–0.100, hgb 0.077–0.094, rf 0.077–0.097.

### How the parameters behave (`01_param_effects.png`)
- **GLM:** stronger ridge penalty hurts (alpha 0.01 is worst); knots barely matter. The load spline adds almost nothing
  over the no-load reference (unit CV 0.0940 vs 0.0947; temporal it is *worse*, 0.0763 vs 0.0725).
- **HGB:** insensitive to learning rate and iterations once depth ≥ 2; large leaves (200) hurt; depth 3 is a noisy dip
  in the temporal scheme (few, small folds). Needs min leaf ≈ 50 or lower.
- **RF:** the most parameter-sensitive. Large leaves (150) smooth away the structure (deviance 0.093–0.105); small
  leaves (20) are best. Unrestricted depth helps in unit CV, hurts in temporal CV (overfits periods).

### What the fits show
1. **Tree models beat the GLM only in unit CV**, and the gap is large (0.077 vs 0.094). But `glm` has the load spline and
   `glm_noload` does not, and they are almost equal, so the gain does not come from a smooth load effect in the GLM.
2. **The partial-dependence curves (`01_load_curves.png`) explain it.** HGB/RF predict a *very high* fatality at
   relative load below about 0.2 (up to 3 deaths per admission at load 0.02 for HGB), then a flat line from about 0.3
   to 1.2. There is **no rise of fatality with load**. The low-load spike is most likely a lag artefact: deaths in weeks
   +1..+2 still come from admissions during the previous wave, but the denominator (admissions in the quiet weeks) is
   tiny. That is an effect of the target definition, not of the healthcare system.
3. **Temporal CV is dominated by drift.** All models over-predict by +20 to +30% in the first and last folds because
   fatality fell over time (vaccination, Omicron) and `era` cannot extrapolate to a period it was not trained on. This
   scheme says little about the load effect.
4. **Fold variance is large** (unit-fold deviance 0.052–0.134), mostly from a few small or odd units. Differences below
   about 0.005 should not be read as real.

### Problem found and fixed during the experiment
The first run gave 41 and 67 training rows for the first two temporal folds. `add_relative_load(train_end=...)` merged
on a MultiIndex with NaN `census_schema` keys, which silently matched nothing, so only Rio had a load value. Fixed
in `features.py` (explicit "none" label) with a regression test in `tests/test_features.py`; the results above are from the
rerun. Rio's new-schema weeks still have no reference in early folds, because there is no training data in that
schema; they are dropped from those folds.

### Conclusions and caveats
- **Best model on this data so far: `hgb`** (depth 5, 300 iterations, lr 0.1, leaf 50 for unit transfer; a shallow depth 2
  for temporal extrapolation). It is not yet clear that it is better *for the purpose*: the extra accuracy comes from the
  low-load region, and the ABM multiplier needs the high-load region.
- For the ABM multiplier, the data currently show a multiplier of about **1.0 above relative load 0.3**: no overload effect
  detectable with these predictors. This matches the mixed prior results in the report (rate ratios 0.7–1.6 by country).
- Unresolved: the fit is evaluated on deviance over all loads, which rewards getting the quiet weeks right.

### Next experiments (proposed)
1. **Fix the target's lag artefact:** drop weeks with very low admissions (e.g. fewer than 20 in the 2-week window) or use a
   longer admission window; re-run. This is the first thing to try before any further tuning.
2. **Ablation** of `rel_load`, `rel_icu_load`, `icu_share`, `load_growth`, `panel` and `era` in the tree models to see what carries the
   gain, and monotone constraints on `rel_load` for HGB (it supports them).
3. **Evaluate on the high-load region only** (e.g. rel_load > 0.5) and per panel.
4. Add IT and HU through the stock target B′ and check agreement of the load curves.
5. Compare against a Poisson GLM with unit fixed effects (the report's design) to reproduce its rate ratios.

---

# Experiment 02 – temporal models (distributed lag, drift, lag-feature boosting)

Code: `src/hcw_ml/temporal.py`, runner `scripts/run_experiment_02.py`, tests `tests/test_temporal.py`.
Outputs: `reports/experiments/02_cv_results.csv` (every fold), `02_best_params.csv`, `02_multiplier.csv`; figures
`reports/figures/02_param_effects.png`, `02_delay_and_multiplier.png`.

**Why.** Experiment 01 used a fixed two-week death window, which produced a low-load artefact and ignored the
admission-to-death delay and the drift over time. Here the model works on **weekly death counts** and learns the delay.

**Models** (all Poisson on weekly deaths, same panel and same folds as exp. 01: ES-NUTS3, BE, Rio; 5,912 unit-weeks
with complete 8-week lag histories):
| name | what | tuned |
|---|---|---|
| `dlag_noload` | E[deaths_t] = panel × era × Σ h_k·adm_{t−k} (learned delay h, no load) | K ∈ {4,6,8} |
| `dlag` | adds a load multiplier m(L) (cubic spline of relative load, per lag), era effects | K, spline knots {4,6}, ridge α {0.1,1,10} |
| `dlag_time_noload` | as `dlag_noload` but a per-panel smooth time trend instead of era dummies | trend df {4,6,10} |
| `dlag_time` | `dlag` with the per-panel smooth time trend (absorbs vaccination/variant drift) | trend df, α {1,10} |
| `hgb_lag` | HistGradientBoosting (Poisson) on death rate per admission, features: lag shares of admissions, lagged load | lr, depth, iterations, leaf |

h = softmax over lags 0..K, so it is a proper delay distribution. The likelihood has an analytic gradient (checked against
finite differences in `tests/test_temporal.py`); a simulation test recovers a known delay and a known m(L). A background
term of 0.1 admission-equivalents covers deaths not traced to admissions. Metric: mean Poisson deviance per unit-week on
counts (lower is better; not comparable to exp. 01's rate deviance), plus the same on high-load weeks (relative load > 0.5)
and bias = predicted/observed deaths − 1.

## Results (mean over folds, best parameters)
| scheme | model | deviance | high-load deviance | bias | best params |
|---|---|---|---|---|---|
| temporal | **dlag_time_noload** | **8.09** | 8.43 | +0.05 | K 6, trend df 4 |
| temporal | dlag_time | 8.43 | 9.99 | +0.06 | trend df 4, α 1 |
| temporal | hgb_lag | 8.54 | 9.92 | +0.06 | lr 0.1, depth 3, 300 iter |
| temporal | dlag_noload | 9.94 | 18.51 | −0.11 | K 4 |
| temporal | dlag | 10.24 | 17.57 | −0.13 | K 4, 4 knots, α 10 |
| unit | dlag_time | 5.30 | 9.80 | +0.02 | trend df 10, α 10 |
| unit | dlag | 6.83 | 13.72 | −0.03 | K 8, 6 knots, α 1 |
| unit | hgb_lag | 8.86 | 20.71 | −0.03 | lr 0.05, depth 2, 100 iter |
| unit | dlag_time_noload | 10.40 | 25.38 | +0.21 | trend df 10 |
| unit | dlag_noload | 29.73 | 83.91 | +0.55 | K 4 |

17 of 432 fits (7 `dlag`, 10 `dlag_time`) hit the iteration cap (500) without converging; they are in the CSV with
`converged = False` and should be treated as unreliable.

## What the fits show
1. **The delay is short.** Fitted h_k peaks at lag 1 week (about 40–48% of deaths), lag 0 about 25%, and almost all deaths
   occur within 4–6 weeks of admission. Two different model variants agree. A static two-week window (exp. 01) was therefore a
   reasonable but crude choice, and the low-load spike in exp. 01 does not appear here.
2. **Modelling the delay matters most.** Adding the learned delay and drift control cuts deviance a great deal
   relative to the era-dummy no-load model (temporal 9.9 → 8.1, unit 29.7 → 10.4).
3. **The load multiplier depends on how drift is handled** (`02_delay_and_multiplier.png`):
   - with era dummies (`dlag`): m rises to about **1.6 at relative load 1.0** (90% unit-bootstrap band 1.26–1.62), and the
     shape holds when weeks with few admissions are dropped (min 20 or 50 admissions: 1.65–1.8 at load 1.2);
   - with a per-panel smooth time trend (`dlag_time`): m is **≈1.0 up to load 0.8 and 1.05–1.09 at load 1.0–1.2**, with a 90% band
     of 0.98–1.21 that includes 1.
   The first estimate is most likely confounded with calendar time: high-load weeks fall in the earlier, deadlier waves, and three
   era steps cannot absorb that. Once the trend is flexible, the load effect largely disappears.
4. **Load does not improve out-of-sample fit once drift is controlled.** In temporal CV the no-load trend model is best (8.09 vs
   8.43 with load). In unit CV the gap in favour of `dlag_time` (5.30 vs 10.40) is driven by one fold: the no-load model scores 29.81
   in fold 3 and 5.9–7.3 in the others, against 3.5–7.5 with load. In the other four folds the two are within about 0.5 of each other,
   so the headline gap should not be read as evidence for a load effect.
5. **Boosting with lag features is competitive but not better** than the structured model, and is not interpretable as a multiplier.
6. **Parameters** (`02_param_effects.png`): K = 4–8 changes little once the trend is controlled; a smoother trend (low df) is better
   out of time, a flexible one (df 10) is better across units; stronger ridge (α 10) helps the load spline when the trend is flexible.
   Boosting prefers shallow trees with leaf ≥ 50.

## Limits
- A flexible per-panel trend can absorb a real load effect when load and time move together. Identification of the load effect
  rests on provinces whose waves are staggered; Belgium (3 regions) and Rio (1 city) add little. The honest summary is a range, not a point.
- Count deviance is large (about 5–10 per unit-week) because deaths are overdispersed; a negative-binomial likelihood would be more suitable.
- Test weeks use observed admissions in the test period (the model conditions on admissions), so this is not a forecast of admissions.
- Time-trend extrapolation in temporal CV is constant beyond the last training week.
- Relative load uses the unit's own p95 as a capacity proxy, so an effect "at capacity" cannot be separated from an effect of load level.

## Conclusion for the ABM
Using the best-supported model (`dlag_time`), the data support a multiplier on fatality of **about 1.0 for relative load ≤ 0.8 and at most
about 1.2 at relative load 1.0–1.2** (upper end of the 90% band). The 1.6 from the era-dummy model is a plausible upper bound but is
probably inflated by time confounding. For PanSIM, I'd use m(L) = `dlag_time` curve from `02_multiplier.csv` as the central case
and the `dlag` curve as a high-sensitivity case when exploring how fatality responds to reduced HCW availability.

## Next experiments (proposed)
1. Negative-binomial likelihood and unit random effects; raise the iteration cap or re-parametrise to remove the 17 non-converged fits.
2. Allow m(L) to differ by panel (ES vs BE vs Rio) and test for heterogeneity.
3. Add IT and HU through a census-based (stock) version of the lag model.
4. Rolling-origin folds with more and shorter test windows, so that temporal CV is not dominated by three folds.

---

# Experiment 03 – negative-binomial lag model and per-panel multipliers

Code: `DistLag(family="nb")` in `src/hcw_ml/temporal.py` (Var = μ + μ²/r, r fitted, analytic gradient tested), runner `scripts/run_experiment_03.py`.
Outputs: `reports/experiments/03_cv_results.csv`, `03_best_params.csv`, `03_multiplier_by_panel.csv`; figure `reports/figures/03_multiplier_by_panel.png`.
Same data and folds as exp. 02. Multiplier m(L)/m(0.25) at load 1.0, 90% bootstrap band (units for ES, unit × 26-week blocks for BE and Rio, 40 resamples).

| model | pooled | BE (3 regions) | Rio (1 city) | ES (49 provinces) |
|---|---|---|---|---|
| NB, per-panel time trend (df 4) | 1.39 [1.24, 1.58] | 1.54 [1.28, 1.79] | 1.03 [0.95, 1.27] | 1.12 [0.87, 1.42] |
| NB, era dummies | 1.23 [1.09, 1.41] | 1.63 [1.50, 1.79] | 1.30 [0.98, 1.37] | 1.16 [0.95, 1.45] |

Fitted dispersion r is about 3 for ES and the pooled data (strongly overdispersed), so the Poisson bands in exp. 02 were too narrow. The delay peaks at 1 week in every panel.

## Findings
1. **Model fit.** NB does not improve count deviance (it optimises a different likelihood): temporal CV Poisson 8.39 vs NB 8.84, unit CV 5.31 vs 7.57. Held-out NB log score is best without load in temporal CV (−3.51 vs −3.64) and about equal across load/no-load in unit CV (−3.06 vs −3.10). Load still does not improve out-of-sample prediction.
2. **The pooled multiplier is higher than in exp. 02.** NB with a trend gives 1.39 at load 1.0, against 1.0–1.1 for Poisson `dlag_time`. Part of this is the chosen trend flexibility: unit CV picked time_df 4 for NB (Poisson picked 10), and a stiffer trend absorbs less. Do not read 1.39 as a firmer estimate than the exp. 02 range.
3. **Heterogeneity.** Belgium shows a clear increase (1.5–1.6, bands above 1 under both specifications), ES is weak and consistent with 1 (about 1.1–1.2), Rio is about 1 with a trend. The pooled value is driven by BE, which has only 3 regions and 489 unit-weeks, so it is the least independent evidence.
4. **Limits.** 13 of 120 CV fits did not converge. No unit random effects were fitted (the per-panel trends and panel fixed effects stand in). Per-panel bands for BE and Rio rely on block resampling of very short series.

## Conclusion for the ABM
Sensitivity range for m(1.0): low 1.0, central about 1.2 (NB pooled, era and trend average), high about 1.6 (Belgium / exp. 02 era model). Report per-country curves, not an average; Belgium and Spain differ.

---

# Experiment 04 – stock version for Italy and Hungary

Code: `make_weekly_dataset(exposure="census", panels=...)` in `src/hcw_ml/temporal.py`, runner `scripts/run_experiment_04.py`.
Outputs: `reports/experiments/04_cv_results.csv`, `04_multiplier_stock.csv`; figure `reports/figures/04_multiplier_stock.png`.

IT and HU have no hospital admissions, so the exposure is the weekly hospital census (deaths per patient-week, target B′) instead of admissions.
Same negative-binomial lag model (K = 4), fitted per panel. ES, BE and Rio are re-fitted with the same stock exposure as a cross-check against
exp. 03. Bootstrap: units for IT and ES, unit × 26-week blocks for HU, BE, Rio (40 resamples). m(L)/m(0.25) at load 1.0, 90% band:

| panel | per-panel time trend | era dummies | exp. 03 (admissions) for comparison |
|---|---|---|---|
| **IT** (21 regions, 5,195 unit-weeks) | 1.87 [1.63, 2.11] | 1.71 [1.59, 1.87] | – |
| **HU** (national, 132 weeks) | 1.05 [0.62, 1.23] | 1.11 [0.71, 1.35] | – |
| ES (49 provinces) | 1.30 [1.12, 1.57] | 1.14 [0.97, 1.36] | 1.12 / 1.16 |
| BE (3 regions) | 1.46 [1.26, 1.69] | 1.78 [1.46, 1.96] | 1.54 / 1.63 |
| Rio (1 city) | 1.34 [0.70, 6.68] | 1.50 [0.85, 2.68] | 1.03 / 1.30 |

## Findings
1. **Cross-check.** For the countries with both versions, the census-based fits give about the same picture as the admissions-based ones (ES 1.1–1.3, BE 1.5–1.8). The stock version is therefore usable where admissions are missing. Rio's band is too wide to say anything.
2. **Italy: the largest and best-supported effect.** m(1.0) is 1.7–1.9 with narrow bands, under both ways of handling time. It is also the first panel where adding load improves held-out fit: temporal CV deviance 9.77–9.89 with load vs 10.57 without (time trend), 11.26 vs 12.53 (era); unit CV 8.97–9.03 vs 9.98 (trend), 10.10 vs 10.80 (era). The held-out log score barely moves (−3.80 vs −3.83), so the gain is modest.
3. **Hungary: no usable estimate.** m(1.0) is 1.05–1.11 with bands 0.6–1.35, which include 1. Models with load do worse out of sample than without (temporal deviance 249–315 vs 80–188), a sign of overfitting 132 weekly values of one national series, where load and calendar time are almost the same variable. Treat Hungary as "no evidence either way".
4. **Caution on the stock exposure.** The census is both the exposure and the source of the load measure, so part of the Italian effect may come from the census at high load containing more severe patients or longer stays (deaths per patient-week rise even if the per-patient risk does not). The admissions-based result for ES and BE does not have this coupling.
5. The delay is shorter than in the admissions version (about 60–85% of deaths in the same week), as expected, because the census already counts patients who are in hospital.

## Conclusion for the ABM
Per-country m(1.0): IT about 1.7–1.9, BE 1.5–1.8, ES 1.1–1.3, Rio unclear, HU unclear (about 1.0–1.1, band includes 1). Italy and Belgium are the evidence for a real effect; Spain is weak, Hungary is uninformative. For PanSIM (Hungarian calibration) the Hungarian data cannot supply the multiplier, so the multiplier has to come from other countries as a scenario (Spain low, Italy/Belgium high), not from Hungary's own curve.
