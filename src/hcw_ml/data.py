"""Loading and reshaping the long-format COVID hospital/ICU/deaths export."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

LONG_FILE = RAW / "covid_hospital_icu_deaths_long.csv.gz"
CATALOGUE_FILE = RAW / "covid_hospital_icu_deaths_series_catalogue.csv"

UNIT_KEYS = ["country_iso2", "geo_level", "geo_code"]


def load_long(path: Path = LONG_FILE) -> pd.DataFrame:
    """One row per country/geography/date/indicator/source."""
    df = pd.read_csv(
        path,
        parse_dates=["date"],
        dtype={"geo_code": "string", "geo_name": "string", "value_flag": "string"},
    )
    for col in ["country_iso2", "geo_level", "indicator", "source", "series_role", "time_res"]:
        df[col] = df[col].astype("category")
    return df


def load_catalogue(path: Path = CATALOGUE_FILE) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["start", "end"])


def to_wide(df: pd.DataFrame, geo_level: str | None = None, role: str = "primary") -> pd.DataFrame:
    """Pivot indicators to columns, indexed by (country, geo_level, geo_code, date)."""
    d = df[df["series_role"] == role]
    if geo_level is not None:
        d = d[d["geo_level"] == geo_level]
    return (
        d.pivot_table(index=UNIT_KEYS + ["date"], columns="indicator", values="value", observed=True)
        .sort_index()
    )
