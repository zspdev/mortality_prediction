import pandas as pd

from hcw_ml.data import to_wide


def test_to_wide_pivots_indicators():
    df = pd.DataFrame({
        "country_iso2": ["IT"] * 2, "geo_level": ["NUTS2"] * 2, "geo_code": ["ITC1"] * 2,
        "date": pd.to_datetime(["2020-03-01"] * 2),
        "indicator": ["deaths_new", "icu_census_covid"], "value": [1.0, 5.0],
        "series_role": ["primary"] * 2,
    })
    wide = to_wide(df)
    assert set(wide.columns) == {"deaths_new", "icu_census_covid"}
