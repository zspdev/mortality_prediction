import numpy as np
import pandas as pd

from hcw_ml.features import add_relative_load, add_targets


def _panel(**cols):
    n = len(next(iter(cols.values())))
    return pd.DataFrame({"panel": "P", "country": "XX", "geo_level": "L", "unit": "u",
                         "week": pd.date_range("2021-01-04", periods=n, freq="W-MON"), **cols})


def test_cfr_flow_uses_leading_deaths_and_lagging_admissions():
    p = add_targets(_panel(deaths=[0, 0, 10, 20, 0], admissions=[10, 10, 0, 0, 0], census=[1, 1, 1, 1, 1]))
    # week index 1: deaths in weeks +1,+2 = 10+20, admissions in weeks -1,0 = 10+10
    assert p.loc[1, "cfr_flow"] == 30 / 20
    assert np.isnan(p.loc[4, "cfr_flow"])  # no admissions -> undefined, not inf


def test_gap_does_not_pull_in_later_week():
    p = _panel(deaths=[1, 2, 3, 4], admissions=[1, 1, 1, 1], census=[1, 1, 1, 1]).drop(index=1)
    out = add_targets(p)
    assert len(out) == 4 and np.isnan(out.loc[0, "deaths_lead1"])


def test_negative_deaths_become_missing():
    p = add_targets(_panel(deaths=[1, -3, 2], census=[1, 1, 1]))
    assert p["deaths"].isna().tolist() == [False, True, False]


def test_relative_load_quantile_fit_on_train_only():
    p = _panel(census=[1, 2, 3, 4, 100])
    full = add_relative_load(p)["rel_load"].iloc[0]
    train = add_relative_load(p, train_end="2021-01-25")["rel_load"].iloc[0]  # first 4 weeks only
    assert train > full


def test_relative_load_train_end_keeps_units_without_schema():
    p = _panel(census=[1, 2, 3, 4, 5]).assign(census_schema=[np.nan, np.nan, "new", "new", "new"])
    out = add_relative_load(p, train_end="2021-02-01")
    assert out["rel_load"].notna().all() or out["rel_load"].notna().sum() >= 2
    q = _panel(census=[1, 2, 3, 4, 5]).assign(census_schema=np.nan)
    assert add_relative_load(q, train_end="2021-02-01")["rel_load"].notna().all()
