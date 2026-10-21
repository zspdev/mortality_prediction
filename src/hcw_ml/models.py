"""Candidate models for case fatality as a function of hospital load (PLAN.md §6).

All models predict the *rate* (deaths per admission, target B) and are fit with ``sample_weight`` = admissions.
For a Poisson model this is equivalent to a count model with a log(admissions) offset, and it lets every
estimator share one interface. Predicted deaths = rate * admissions.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer

# Era is a coarse stand-in for vaccination / variant (no such data exist): pre-vaccine, vaccine + Alpha/Delta,
# Omicron. It is a calendar function, so it is known at prediction time.
ERA_BREAKS = [pd.Timestamp("2021-03-01"), pd.Timestamp("2021-12-20")]

LOAD = ["rel_load"]
EXTRA = ["rel_icu_load", "icu_share", "load_growth"]
CAT = ["panel", "era"]


def make_dataset(panel: pd.DataFrame) -> pd.DataFrame:
    """Rows usable for target B: panels with admissions, positive exposure, defined outcome."""
    d = panel[panel["admissions"].notna() & (panel["panel"] != "ES-comarca")].copy()
    d = d[(d["adm_lag01"] > 0) & d["deaths_lead12"].notna() & d["rel_load"].notna()].copy()
    d["era"] = sum((d["week"] >= b).astype(int) for b in ERA_BREAKS).astype(str)
    d["rate"] = d["deaths_lead12"] / d["adm_lag01"]
    d["exposure"] = d["adm_lag01"]
    return d.reset_index(drop=True)


def add_derived(d: pd.DataFrame) -> pd.DataFrame:
    """Week-on-week load growth (past only) and ICU share of the hospital census."""
    d = d.sort_values(["panel", "unit", "week"]).copy()
    d["load_growth"] = d.groupby(["panel", "unit"])["rel_load"].diff()
    d["icu_share"] = d["icu_census"] / d["census"].where(d["census"] > 0)
    return d


def _spline(df, n_knots):
    return SplineTransformer(n_knots=n_knots, degree=3, extrapolation="constant", include_bias=False)


def glm(alpha=1e-3, n_knots=5, use_load=True):
    """Ridge-penalised Poisson GLM: spline(rel_load) + panel and era effects. ``use_load=False`` is the
    no-load reference the load effect has to beat."""
    parts = [("cat", OneHotEncoder(handle_unknown="ignore"), CAT)]
    if use_load:
        parts.append(("load", _spline(None, n_knots), LOAD))
    return Pipeline([("prep", ColumnTransformer(parts)),
                     ("m", PoissonRegressor(alpha=alpha, max_iter=1000))])


def _tree_prep():
    num = LOAD + EXTRA
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
        ("num", SimpleImputer(strategy="median", add_indicator=True), num),
    ])


def hgb(learning_rate=0.05, max_depth=3, max_iter=200, min_samples_leaf=50, l2_regularization=1.0):
    return Pipeline([("prep", _tree_prep()),
                     ("m", HistGradientBoostingRegressor(loss="poisson", learning_rate=learning_rate,
                                                         max_depth=max_depth, max_iter=max_iter,
                                                         min_samples_leaf=min_samples_leaf,
                                                         l2_regularization=l2_regularization, random_state=0))])


def rf(min_samples_leaf=50, max_features=0.5, max_depth=None, n_estimators=200):
    return Pipeline([("prep", _tree_prep()),
                     ("m", RandomForestRegressor(criterion="poisson", n_estimators=n_estimators,
                                                 min_samples_leaf=min_samples_leaf, max_features=max_features,
                                                 max_depth=max_depth, n_jobs=-1, random_state=0))])


# name -> (builder, parameter grid)
MODELS = {
    "glm_noload": (lambda **k: glm(use_load=False, **k), {"alpha": [1e-4, 1e-2]}),
    "glm": (glm, {"alpha": [1e-5, 1e-4, 1e-3, 1e-2], "n_knots": [4, 5, 7]}),
    "hgb": (hgb, {"learning_rate": [0.03, 0.1], "max_depth": [2, 3, 5], "max_iter": [100, 300],
                  "min_samples_leaf": [50, 200]}),
    "rf": (rf, {"min_samples_leaf": [20, 50, 150], "max_features": [0.5, 1.0], "max_depth": [6, None]}),
}
