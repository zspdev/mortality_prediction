"""Weekly unit panel with the case-fatality targets and relative-load features (PLAN.md §1, §4).

One row per (panel, unit, ISO week starting Monday). Flows (admissions, deaths) are weekly sums, stocks
(census) are weekly means over the days that were reported.

Targets
- ``cfr_flow``  = deaths in weeks +1..+2 / admissions in weeks -1..0   (needs admissions)
- ``cfr_stock`` = deaths in week +1 / census in week 0                 (deaths per patient-week)
"""
from pathlib import Path

import numpy as np
import pandas as pd

from hcw_ml.data import load_long

# Panels: (country, geo_level) -> how deaths / admissions / census are taken. Each panel is an
# independent set of units; ES NUTS3 and ES comarca overlap in Catalonia, so never pool them.
BE_REGION = lambda code: code[:3]  # BE10 -> BE1, BE21..BE25 -> BE2, BE31..BE35 -> BE3


def _daily(df: pd.DataFrame, indicator: str, level: str, roles=("primary",), col: str | None = None) -> pd.Series:
    d = df[(df.indicator == indicator) & (df.geo_level == level) & (df.series_role.isin(roles))]
    s = d.groupby(["geo_code", "date"], observed=True)["value"].sum(min_count=1)
    return s.rename(col or indicator)


def _weekly(daily: pd.Series, kind: str) -> pd.Series:
    """Daily series indexed by (unit, date) -> weekly (unit, week). Flows: sum, zero-filled inside the
    series span (Sciensano omits zero-death days). Stocks: mean over reported days."""
    out = []
    for unit, s in daily.groupby(level=0):
        s = s.droplevel(0)
        if kind == "flow":
            s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D")).fillna(0.0)
            w = s.resample("W-MON", label="left", closed="left").agg(["sum", "size"])
            w = w["sum"][w["size"] == 7]  # complete weeks only
        else:
            w = s.resample("W-MON", label="left", closed="left").mean()
        w.index = pd.MultiIndex.from_product([[unit], w.index], names=["unit", "week"])
        out.append(w)
    return pd.concat(out).rename(daily.name)


def _assemble(parts: dict[str, pd.Series], panel: str, country: str, level: str) -> pd.DataFrame:
    p = pd.concat(parts.values(), axis=1)
    p.columns = list(parts)
    p = p.reset_index()
    p.insert(0, "panel", panel)
    p.insert(1, "country", country)
    p.insert(2, "geo_level", level)
    return p


def _panel_simple(df, country, level, deaths, adm, census, icu=None, panel=None, roles=("primary",)):
    parts = {}
    parts["deaths"] = _weekly(_daily(df[df.country_iso2 == country], deaths, level, roles), "flow")
    if adm:
        parts["admissions"] = _weekly(_daily(df[df.country_iso2 == country], adm, level, roles), "flow")
    if census:
        parts["census"] = _weekly(_daily(df[df.country_iso2 == country], census, level, roles), "stock")
    if icu:
        parts["icu_census"] = _weekly(_daily(df[df.country_iso2 == country], icu, level, roles), "stock")
    return _assemble(parts, panel or f"{country}-{level}", country, level)


def _panel_be(df) -> pd.DataFrame:
    """Hospital data are by province (NUTS2) but deaths only by region (NUTS1): aggregate the former."""
    be = df[df.country_iso2 == "BE"]
    parts = {"deaths": _weekly(_daily(be, "deaths_new", "NUTS1"), "flow")}
    for name, ind, kind in [("admissions", "hospital_admissions_new", "flow"),
                            ("census", "hospital_census_covid", "stock"),
                            ("icu_census", "icu_census_covid", "stock")]:
        d = _daily(be, ind, "NUTS2").reset_index()
        d["geo_code"] = d["geo_code"].map(BE_REGION)
        d = d.groupby(["geo_code", "date"])[ind].sum()
        parts[name] = _weekly(d, kind)
    return _assemble(parts, "BE-NUTS1", "BE", "NUTS1")


def _panel_rio(df) -> pd.DataFrame:
    """Rio de Janeiro city: SRAG admissions and in-hospital deaths. Census is ward + ICU of the hospitals
    that reported; its definition changes in 2021-Q3, so ``census_schema`` records which side a week is on
    and relative load must be computed within a schema."""
    br = df[(df.country_iso2 == "BR") & (df.geo_level == "municipality")]
    parts = {
        "deaths": _weekly(_daily(br, "deaths_new_in_hospital_srag", "municipality"), "flow"),
        "admissions": _weekly(_daily(br, "hospital_admissions_new", "municipality"), "flow"),
    }
    census = []
    for schema, (ward, icu) in {"old": ("ward_census_covid_confirmed", "icu_census_covid_confirmed"),
                                "new": ("ward_census_covid_new_schema", "icu_census_covid_new_schema")}.items():
        w = _daily(br, ward, "municipality")
        i = _daily(br, icu, "municipality")
        tot = pd.concat([w, i], axis=1).sum(axis=1, min_count=2).rename("census")
        tot = _weekly(tot, "stock").dropna().to_frame()
        tot["census_schema"] = schema
        census.append(tot)
    census = pd.concat(census)
    # keep each week's schema only up to / from its natural period (old to 2021-09-12, new from it)
    cut = pd.Timestamp("2021-09-13")
    wk = census.index.get_level_values("week")
    census = census[((census.census_schema == "old") & (wk < cut)) | ((census.census_schema == "new") & (wk >= cut))]
    p = _assemble(parts, "BR-Rio", "BR", "municipality").merge(
        census.reset_index(), on=["unit", "week"], how="left")
    return p


def build_panel(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Weekly panel for ES (NUTS3 and comarca), BE, IT, HU, BR-Rio. Türkiye is excluded: no hospital
    series after July 2020."""
    df = load_long() if df is None else df
    panels = [
        _panel_simple(df, "ES", "NUTS3", "deaths_new", "hospital_admissions_new", "hospital_census_covid",
                      "icu_census_covid", panel="ES-NUTS3"),
        _panel_simple(df, "ES", "comarca", "deaths_new", "hospital_admissions_new", "hospital_census_covid",
                      "icu_census_covid", panel="ES-comarca"),
        _panel_be(df),
        # Italy: only ICU entries from 2020-12, so no hospital admissions -> stock target only.
        _panel_simple(df, "IT", "NUTS2", "deaths_new", None, "hospital_census_covid", "icu_census_covid"),
        # Hungary: national, reporting days; 2022 weekly values cover the days since the last report.
        _panel_simple(df, "HU", "country", "deaths_new", None, "hospital_census_covid"),
        _panel_rio(df),
    ]
    p = pd.concat(panels, ignore_index=True)
    p = p[p.groupby(["panel", "unit"])["deaths"].transform("count") > 0]  # units without any death series
    p = p.sort_values(["panel", "unit", "week"]).reset_index(drop=True)
    return add_targets(p)


def add_targets(p: pd.DataFrame) -> pd.DataFrame:
    """Add lead/lag columns and the two fatality targets. Shifts run on a complete weekly grid per unit, so
    gaps give NaN rather than silently pulling in a later week."""
    out = []
    ids = ["panel", "country", "geo_level", "unit"]
    for _, g in p.groupby(["panel", "unit"], sort=False):
        meta = g.iloc[0][ids]
        g = g.set_index("week").sort_index().drop(columns=ids)
        g = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="W-MON"))
        g.index.name = "week"
        for c in ids:
            g[c] = meta[c]
        if "admissions" in g:
            g["adm_lag01"] = g["admissions"] + g["admissions"].shift(1)
        out.append(g.reset_index())
    p = pd.concat(out, ignore_index=True)
    # Italy's cumulative-count corrections give negative weekly deaths: treat as missing, not as zero.
    p.loc[p["deaths"] < 0, "deaths"] = np.nan
    p["deaths_lead1"] = p.groupby(["panel", "unit"])["deaths"].shift(-1)
    p["deaths_lead12"] = p["deaths_lead1"] + p.groupby(["panel", "unit"])["deaths"].shift(-2)
    if "adm_lag01" in p:
        ok = p["adm_lag01"] > 0
        p["cfr_flow"] = np.where(ok, p["deaths_lead12"] / p["adm_lag01"].where(ok), np.nan)
    ok = p["census"] > 0
    p["cfr_stock"] = np.where(ok, p["deaths_lead1"] / p["census"].where(ok), np.nan)
    return p


def add_relative_load(p: pd.DataFrame, train_end=None, q: float = 0.95) -> pd.DataFrame:
    """Census / the unit's own q-quantile census (capacity proxy, since no bed counts exist).

    The quantile is computed per (panel, unit[, census_schema]) on weeks <= ``train_end`` only (all weeks if
    None), so it can be fit inside a training fold without leakage. ICU load is added if icu_census exists."""
    p = p.copy()
    # NaN group keys do not survive a merge on a MultiIndex, so give "no schema" an explicit label.
    p["_schema"] = p["census_schema"].astype("object").fillna("none") if "census_schema" in p else "none"
    keys = ["panel", "unit", "_schema"]
    for col, new in [("census", "rel_load"), ("icu_census", "rel_icu_load")]:
        if col not in p:
            continue
        fit = p if train_end is None else p[p["week"] <= pd.Timestamp(train_end)]
        ref = fit.groupby(keys)[col].quantile(q).rename("_ref")
        p = p.merge(ref, left_on=keys, right_index=True, how="left")
        p[new] = p[col] / p["_ref"].where(p["_ref"] > 0)
        p = p.drop(columns="_ref")
    return p.drop(columns="_schema")


def save_panel(path: Path | None = None) -> Path:
    """Build the weekly panel with relative load (full-sample p95) and write it to data/processed."""
    from hcw_ml.data import PROCESSED

    path = PROCESSED / "weekly_panel.parquet" if path is None else path
    path.parent.mkdir(parents=True, exist_ok=True)
    add_relative_load(build_panel()).to_parquet(path, index=False)
    return path
