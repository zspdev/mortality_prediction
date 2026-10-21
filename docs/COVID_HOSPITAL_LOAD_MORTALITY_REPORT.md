# COVID-19 hospitalisation, ICU and deaths export, and a first look at mortality under hospital load

## 1. Italy: what was collected
- **Source**: Dipartimento della Protezione Civile open data (GitHub `pcm-dpc/COVID-19`), file `dpc-covid19-ita-regioni.csv`: 37,401 rows, 21 regions and autonomous provinces, daily, 2020-02-24 to 2025-01-08. Also downloaded: the province file (cases only) and the national series.
- **Used series**: COVID-19 patients in hospital (`totale_ospedalizzati`), in ordinary wards (`ricoverati_con_sintomi`), in ICU (`terapia_intensiva`), daily ICU entries (`ingressi_terapia_intensiva`, available from 2020-12-03 only) and cumulative deaths (`deceduti`). Daily deaths are the difference of the cumulative series; 46 region-days are negative (source corrections) and are kept with a flag. Total deaths in the file: 198,683.
- **Not obtained: bed capacity.** Intensive-care bed counts by region come from the Ministry of Health open-data file "Posti letto per Regione e disciplina" (dati.salute.gov.it). That server fails the TLS handshake from here and its archive mirror is refused by the proxy. Please download the 2020 file ("Posti letto per Regione e disciplina 2020"; discipline "Terapia intensiva") into Downloads and I will join it. Capacity is the denominator the overload question needs; this analysis uses a proxy instead (section 3).

## 2. The export (all countries)
`covid_hospital_icu_deaths_long.csv.gz`: 719,766 rows, one row per country, geography, date, indicator and source (schema in the file header; definitions in `definition_note`). `covid_hospital_icu_deaths_series_catalogue.csv` lists the 60 series with units, dates, completeness and flagged values; `covid_population_denominators.csv` holds the populations.

| Country | Hospital census | ICU census | New admissions | COVID deaths | Resolution and window |
|---|---|---|---|---|---|
| Belgium | yes | yes | yes (hospital, ICU) | yes | 11 provinces for hospital data, 3 regions for deaths; 2020-03 to 2023-06 |
| Spain | yes | yes | yes | yes | 49–52 provinces (2020-01 to 2022-06; deaths and admissions to 2022-03); Catalonia 43 comarques (to 2022-07) |
| Italy | yes | yes | ICU entries from 2020-12 | yes | 21 regions, 2020-02 to 2025-01 |
| Hungary | yes (dashboard, to 2022-12) | ventilated patients only | no | yes | national, reporting days |
| Türkiye | no | ICU and ventilated, March–July 2020 only | no | yes (daily, 2020-01 to 2024-08) | national |
| Brazil (Rio) | yes, hospitals' self-report 2020–22 | yes | SRAG COVID admissions and ICU entries | SIM COVID deaths; SRAG hospital deaths | city and state; definition change in 2021-Q3 |

Hungary: hospital census, ventilated patients and deaths all come from the NNGYK dashboard export (`korona_hun.xlsx`) and are COVID-related counts as reported there; deaths are the dashboard's own daily series (48,495 to 2022-12-28; ECDC's national series is kept only as a cross-check, 47,938 to 2022-10-26). The ICU proxy is ventilated patients. Türkiye has no usable hospital series after July 2020. The Rio hospital census is the sum over hospitals that reported on the day (number reporting is exported as `n_hospitals_reporting`); its fields change meaning in 2021-Q3. Sciensano's death file omits days without deaths, so zeros were filled within the span for weekly sums.

## 3. Mortality and hospital load: first results
**Question**: does the death rate per hospital patient rise when hospitals carry a heavier COVID-19 load?

**Design (exploratory, associational)**: weekly unit panels; load = the unit's COVID-19 hospital census per 100,000 or as a share of the unit's own 95th percentile (a proxy for capacity, because capacities were not available); outcome = deaths in weeks +1 and +2 per hospital admission in weeks −1 and 0 (flow version), or deaths in week +1 per hospital patient-week (stock version); Poisson regression with unit fixed effects, calendar-quarter fixed effects and a five-level load variable; cluster-robust errors by unit. Results are rate ratios of the highest load quintile against the lowest (`overload_mortality_specifications.csv`; by quintile in `overload_mortality_flow_bins.csv`).

**Results, deaths per admission, highest vs lowest load quintile, relative load, weeks to 2021-06-30**
- Catalonia (43 comarques, 1,326 unit-weeks): 1.64 (95% CI 1.30–2.05); the rate ratio is 1.48, 1.70, 1.78 and 1.64 across quintiles 2–5, so it rises between the lowest and the second quintile and then flattens. Raw deaths per 100 admissions go from 11 to 21.
- Belgium (3 regions, 201 unit-weeks): 1.46 (1.09–1.96), with a maximum of 1.85 in the fourth quintile. Raw deaths per 100 admissions go from 20 to 37.
- Spain, 49 provinces (ISCIII): 0.68 (0.55–0.85), a decline with load.
- Rio de Janeiro city (SRAG in-hospital deaths): 0.95 (0.80–1.12), flat.
- Over all periods the Catalonia estimate is 1.49 (1.28–1.74), Belgium 0.99 (0.73–1.35), Spain 0.67 (0.50–0.91), Rio 0.99 (0.85–1.14).

**Stock version (deaths per hospital patient-week, per 100k, all periods)**: Italy 0.57 (0.34–0.93), Spain 0.79 (0.74–0.84), Belgium 1.04 (0.77–1.42), Rio 1.04 (0.72–1.50), Hungary (NNGYK dashboard deaths and hospital census, single national series, 106 weeks) 1.93 (1.11–3.36), 2.76 (1.48–5.14) to June 2021.

**Reading**: the sign of the association differs by country, so the data do not show a general rise in mortality with load. Two series (Catalonia, Belgium before mid-2021) show 1.5–1.8 times more deaths per admission in weeks of high load than in inter-wave weeks, and Hungary's national series points the same way. Catalonia's plateau from the second quintile onward means this cannot be called an overload gradient: the lowest quintile is the inter-wave trough, so the contrast may reflect wave phase and case mix as much as strain.

## 4. Why this is not yet a test of overload
1. **No capacity denominator.** Load relative to the unit's own 95th percentile is not occupancy of actual beds, and ICU bed numbers rose during the waves.
2. **Deaths in all settings.** Except for the Rio SRAG series, deaths include care homes and homes, which follow community incidence, so deaths per admission is not an in-hospital fatality rate.
3. **Admission ascertainment.** Spring 2020 admissions were under-counted where testing was limited, which inflates deaths per admission in early weeks.
4. **Case mix and variant.** Age, vaccination status and variant are not in these series; quarter fixed effects only control for the period on average.
5. **Reporting.** ISCIII data are by date of event and revised later; the Rio hospital census depends on which hospitals reported.
6. **Few independent units** for Belgium (3 regions), Hungary (1) and Rio (1).

## 5. What would make it a real test
- ICU bed capacity by region and week: Italian Ministry file (needs your download), Belgian Sciensano ICU capacity, Catalan hospital bed data, CNES intensive-care beds for Rio (already in the Brazil database).
- In-hospital deaths and discharges by admission date, with age bands (Brazil SRAG has them; Sciensano publishes age-specific admissions and deaths).
- A model with admission cohorts followed to outcome, adjusted for age and calendar-time effects, and a capacity-based load variable; then a load multiplier on hospital fatality for PanSIM.
