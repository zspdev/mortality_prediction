# HCW_ML – Project plan

**Goal:** estimate how hospital COVID-19 mortality rises with hospital load (a proxy for strained healthcare capacity), across countries (BE, BR-Rio, ES, HU, IT, TR). The output is a **load multiplier on in-hospital fatality** used to tune the ABM (PanSIM) that simulates virus spread, so the ABM can show how the death rate changes as healthcare-worker (HCW) availability changes.

**Key limitation:** the data contain no HCW information. HCW availability must enter the ABM through an explicit assumption (e.g. staff shortfall lowers effective capacity, which raises relative load). That link comes from literature or calibration, not from this data.

## 0. What the data is (and isn't)

- `data/raw/covid_hospital_icu_deaths_long.csv.gz`: ~720k rows, long format (country, geo unit, date, indicator, source). Indicators: `deaths_new/cum`, `hospital_census_covid`, `icu_census_covid`, `ward_census_covid`, `hospital_admissions_new`, `icu_admissions_new`, `ventilated_census_covid`, plus Brazil-specific variants.
- `data/raw/..._series_catalogue.csv`: 60 series with coverage and flags.
- `docs/COVID_HOSPITAL_LOAD_MORTALITY_REPORT.md`: data provenance and a first associational analysis.
- **Missing:** `covid_population_denominators.csv` (mentioned in the report, not supplied), ICU bed capacity, age, vaccination, variant, case counts.
- Coverage is very uneven: ES (~56% of rows) and IT dominate; HU and TR are single national series; TR has no hospital data after July 2020; BR definitions change in 2021-Q3.

## 0b. The ABM interface (PanSIM, `/Users/zsofiaprager/Documents/PhD/PanSIM/pansim`)

Read from the code and configs; no other data is available.

- **Hospital states:** progression has `I5_h` and `I6_h` (hospitalised, work-state `S`), then `R_h`, `D1` (COVID death) and `D2` (non-COVID death). From `I3`/`I4`, an agent goes to `I5_h` (~0.30%) or `I6_h` (~0.12%) in the example matrix (illness0_3); from `I6_h`, `D1` has probability ~0.21 and `R_h` ~0.79. Matrices are per age band and pre-condition.
- **Only existing death lever:** `--diseaseProgressionDeathScaling` (per strain, default `1.0,1.03,1.3,0.6,0.6`), applied in `simulationOrchestration.cpp` when the agent is in state index 7 (`I6_h`). It is static, so there is **no load-dependent mortality** today.
- **HCWs:** agents working at location type 12 (hospital) or 14 (doctor's office), flagged by `flagHealthcareWorkers()`; 10,136 in the Hungarian example. HCW absence arises only through infection and quarantine. Hospital location `capacity` is a movement limit, not a bed count.
- **Consequence for this project:** the ML output m(L) should scale the `I6_h → D1` transition. Because it is a ratio to baseline fatality, a mismatch between PanSIM's baseline CFR and the observed one cancels. m(L) must be computed from relative load, which in PanSIM is hospital census (`I5_h` + `I6_h`) relative to a reference level, and relative load in turn depends on how many HCWs are present.
- **Required ABM change (outside this repo):** replace the static per-strain death scaling with a time-varying factor m(L_t), where L_t is the daily hospital census over a reference (e.g. the p95 or a capacity that scales with the share of HCWs available). The tuning lookup table can be exported as a CSV.

## 1. Target definition (decided)

"Mortality rate" is not a column. **Primary target: in-hospital case fatality proxy (B)** = deaths in weeks +1..+2 per hospital admission in weeks −1..0. It maps onto a per-patient death probability/hazard in the ABM, and it avoids double-counting the epidemic dynamics the ABM already simulates.

| Target | Definition | Role |
|---|---|---|
| **B. Deaths per admission** | deaths wk+1..+2 / admissions wk−1..0 | **primary**; model as deaths with a log-admissions offset (Poisson/NB) |
| B′. Deaths per hospital patient-week | deaths wk+1 / census | stock version for HU and others without admissions |
| A. Weekly deaths per 100k | deaths / population | not used: dominated by incidence; needs population file |
| C. Weekly death count | Poisson/NB | baseline only |

**Framing: explanatory, not forecasting.** Estimate CFR as a function of relative load, controlling for unit and period. Deliverable: the partial-dependence curve CFR(load) and the multiplier m(L) = CFR(L) / CFR(baseline load), with uncertainty, per country and pooled. Do not use past deaths or raw census as features; they teach persistence rather than load effects.

## 2. Project layout

```
data/raw/          immutable inputs          src/hcw_ml/data.py      loaders, pivots
data/processed/    weekly unit panel (parquet) src/hcw_ml/features.py  lags, rolling, load-vs-p95   (todo)
notebooks/         jupytext .py notebooks    src/hcw_ml/models.py    baselines + ML                (todo)
reports/figures/   outputs                   src/hcw_ml/evaluate.py  CV splits, metrics            (todo)
docs/              provenance report         tests/
```

Environment: `uv sync`; run notebooks with `uv run jupyter lab`.

## 3. Phase 1 – EDA (`notebooks/01_eda.py`, started)

1. **Coverage:** units × indicators × date range per country; missingness map; flagged values (1,488) and negative daily deaths from cumulative corrections.
2. **Data quality:** reporting-day artefacts (HU), zero-filling (BE deaths), definition break (BR 2021-Q3), revisions (ES), duplicate or inconsistent series between `primary` and `country_series` roles (e.g. summed units vs national).
3. **Target behaviour:** weekly deaths per country/unit, overdispersion, zero inflation at small units, wave structure.
4. **Relationships:** lead–lag cross-correlation of census/admissions vs deaths per unit (expect deaths to lag admissions by ~1–3 weeks); scatter of load vs deaths per admission by country; the sign differs by country in the report, so check heterogeneity explicitly.
5. **Distribution shift:** compare periods (pre-vaccine, Delta, Omicron) and countries.
6. **Decision output:** which countries/units/periods are usable, and the final target definition.

## 4. Phase 2 – Preprocessing and feature panel

- Aggregate to a **weekly unit panel** (ISO weeks), keep one canonical series per indicator (resolve primary vs secondary), handle flags/negatives explicitly.
- Features: **relative load** (census and ICU census relative to the unit's own p95, or to bed capacity once available; p95 computed on training data only), ICU share of hospital census, week-on-week growth, period terms (pre-vaccine, Delta, Omicron), country/unit identifiers. Past deaths are excluded.
- Units with admissions (ES, BE, Rio; IT ICU entries from 2020-12) use target B; HU (and any unit without admissions) uses B′.
- Missing-data policy per indicator (HU/TR lack some; Brazil has two schemas) – no silent imputation across the BR break.

## 5. Phase 3 – Validation design (do this before modelling)

- **Temporal splits** (expanding window / rolling origin) for model selection; never random K-fold. Also **unit-grouped CV**, since neighbouring units in one wave are not independent.
- **Leave-one-country-out** to answer the "across countries" question – can a model transfer to an unseen country?
- Metrics: MAE/RMSE on rates, Poisson deviance on counts, calibration of prediction intervals; report per country, not only pooled.
- Baselines: constant CFR per unit and period (no load effect), then deaths ~ lagged admissions with load added. The load effect must beat the no-load baseline out of sample.

## 6. Phase 4 – Modelling

1. Baselines above.
2. Interpretable: Poisson/NegBin GLM with unit and period fixed effects (matches the report's design, so results are comparable).
3. ML: gradient boosting (LightGBM/HistGradientBoosting, Poisson loss), random forest; optionally a mixed-effects or hierarchical model for small units.
4. Tuning with the temporal CV only; compare pooled vs per-country models.
5. Interpretation: partial dependence of CFR on relative load (the main output), SHAP, permutation importance.
6. **ABM hand-off:** export m(L) per country and pooled (table + monotone fit), with CIs, as a CSV lookup (relative load → multiplier on `I6_h → D1`). Report where the sign differs by country; do not average it away. Hungary is the natural primary calibration target, since PanSIM's example is Hungarian and `korona_hun.xlsx` is already in both repos, but HU has only a national series with census, so use the stock version B′ there.

## 7. Phase 5 – Evaluation and reporting

Error by country/wave/period, failure analysis around wave peaks, uncertainty intervals, limitations (no capacity, age, vaccination, variant; deaths in all settings vs in-hospital). Outputs to `reports/`.

## 8. Risks and open questions

- **No HCW data:** staffing effects are inferred only via a load/capacity proxy plus an assumed link in the ABM.
- **Mixed prior evidence:** the report finds rate ratios 1.5–1.8 (Catalonia, Belgium pre mid-2021), 0.68 (Spain provinces) and flat (Rio). Catalonia's plateau may reflect wave phase and case mix rather than overload.
- **Deaths in all settings:** only Rio SRAG is in-hospital, so CFR here is a proxy.
- **Confounders missing:** age, vaccination, variant, case counts; early-2020 admissions under-counted.
- **Small effective sample of independent units** for BE (3 regions), HU, TR, Rio → cross-country generalisation will be weakly identified.
- **Open:** whether to restrict to countries with admissions; how the ABM defines the load reference (proposed in 8b: effective capacity reduced by the infected-HCW share).
- **ABM limitation:** PanSIM has no load-dependent mortality yet, so the multiplier needs a code change.
- **No extra data exists:** population, capacity, age and vaccination files will not be supplied. Population denominators are not needed for target B. Relative load uses each unit's own p95 as the capacity proxy.

## 8b. HCW infection handling (decided: no HCW data exists)

No HCW infection or absence series is available, so none is fitted. The ML side learns only m(L), the fatality multiplier as a function of relative load L = census / unit p95 census. HCWs enter on the ABM side:

1. Each week PanSIM counts infected (and isolating) HCWs, `H_inf(t)`, from its own epidemic state.
2. Effective capacity: `C_eff(t) = C0 * (1 - H_inf(t) / H_total)`, where `C0` is the baseline capacity, set so the unit's p95 census corresponds to L = 1.
3. Relative load: `L(t) = census_ABM(t) / C_eff(t)`. HCW infections raise L and so move fatality along m(L).
4. Apply `m(L)` as a multiplier on `I6_h -> D1`, using the one-week delay found in experiment 02.

Caveats: the HCW-to-capacity link (step 2, assuming capacity scales linearly with the share of HCWs working) is an assumption, not an estimate. The ML data cannot validate it. Run it as a sensitivity analysis (e.g. capacity elasticity 0.5, 1.0, 1.5). Also, m(L) is confounded with calendar time (experiment 02), so use the flat per-panel-trend curve and the `dlag` curve as the low and high cases.

## 9. Progress and next steps

- Done: weekly panel and targets (`features.py`), models and CV (`models.py`, `evaluate.py`), experiment 01 (three models, first
  parameter sweep). See `docs/EXPERIMENTS.md`. Headline: tree models fit best, but no rise of fatality with load is visible
  above relative load 0.3; the tree gain comes from a low-load artefact of the target.

- Done: experiment 02 (temporal models: distributed lag with learned delay, drift control, lag-feature boosting). Delay peaks at
  1 week; the load multiplier is about 1.6 at full load with era dummies but about 1.0 (band 0.98-1.21) with a per-panel smooth
  trend, so the load effect is largely confounded with calendar time. See `docs/EXPERIMENTS.md`.
- Environment note: the repo's `.venv` was evicted by macOS/iCloud; use `UV_PROJECT_ENVIRONMENT=~/.venvs/hcw_ml uv sync --frozen`
  and run Python from there.

- Done: experiment 03 (negative-binomial lag model, per-panel multipliers). m(1.0): pooled 1.2–1.4, BE 1.5–1.6, ES about 1.1–1.2, Rio about 1.0–1.3. See `docs/EXPERIMENTS.md`.

- Done: experiment 04 (stock version for IT and HU). m(1.0): IT about 1.7–1.9 (load improves held-out fit modestly), HU about 1.05–1.1 with band including 1 (no usable estimate). See `docs/EXPERIMENTS.md`.

**Next:**

1. Unit random effects; check whether the Italian effect survives a census-independent load measure (e.g. ICU share).
2. Work with available data only (no population or capacity files); capacity proxy = unit p95 census.
3. Define the multiplier on `I6_h → D1` and agree how the ABM computes relative load.
4. Build the weekly panel (`features.py`) and the no-load baseline.
