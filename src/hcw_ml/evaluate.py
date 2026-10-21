"""Cross-validation splits and metrics for the fatality models."""
import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import mean_poisson_deviance
from sklearn.model_selection import GroupKFold

from hcw_ml.features import add_relative_load
from hcw_ml.models import add_derived

# Rolling-origin: train on weeks <= cutoff, test on the next 26 weeks.
CUTOFFS = ["2020-12-28", "2021-06-28", "2021-12-27"]
HORIZON_WEEKS = 26


def temporal_folds(panel: pd.DataFrame, build, cutoffs=CUTOFFS):
    """Yield (name, train, test). The relative-load reference (p95) is refit on training weeks only."""
    for c in cutoffs:
        d = build(add_relative_load(panel, train_end=c))
        c = pd.Timestamp(c)
        yield f"t<={c.date()}", d[d["week"] <= c], d[(d["week"] > c) & (d["week"] <= c + pd.Timedelta(weeks=HORIZON_WEEKS))]


def grouped_folds(d: pd.DataFrame, n_splits=5):
    """Hold out whole units (new provinces / regions) over all periods. Tests spatial transfer."""
    groups = d["panel"].astype(str) + ":" + d["unit"].astype(str)
    for i, (tr, te) in enumerate(GroupKFold(n_splits).split(d, groups=groups)):
        yield f"unit-fold{i}", d.iloc[tr], d.iloc[te]


def scores(model, test: pd.DataFrame) -> dict:
    pred = np.clip(model.predict(test), 1e-6, None)
    y, w = test["rate"].values, test["exposure"].values
    dev = mean_poisson_deviance(y, pred, sample_weight=w)
    return {"deviance": dev,
            "mae_rate_w": float(np.average(np.abs(y - pred), weights=w)),
            "bias": float(np.sum(pred * w) / np.sum(y * w) - 1),  # predicted/observed deaths - 1
            "n": len(test)}


def grid(params: dict):
    keys = list(params)
    for vals in itertools.product(*params.values()):
        yield dict(zip(keys, vals))
