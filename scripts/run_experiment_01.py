"""Experiment 01: three candidate models (+ no-load reference) for deaths per admission vs hospital load.

Writes per-fold results and the best parameters to reports/experiments/01_*.csv and figures to
reports/figures/. Run: ``uv run python scripts/run_experiment_01.py``.
"""
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hcw_ml.evaluate import grid, grouped_folds, scores, temporal_folds
from hcw_ml.features import add_relative_load, build_panel
from hcw_ml.models import MODELS, add_derived, make_dataset

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "experiments"
FIG = ROOT / "reports" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

FEATURES = ["panel", "era", "rel_load", "rel_icu_load", "icu_share", "load_growth"]


def build(panel):
    return add_derived(make_dataset(panel))


def fit(builder, params, train):
    m = builder(**params)
    m.fit(train[FEATURES], train["rate"], m__sample_weight=train["exposure"])
    return m


class Wrapped:
    """Lets scores() call model.predict(test_frame)."""
    def __init__(self, m):
        self.m = m

    def predict(self, d):
        return self.m.predict(d[FEATURES])


def main():
    panel = build_panel()
    full = build(add_relative_load(panel))
    schemes = {
        "temporal": list(temporal_folds(panel, build)),
        "unit": list(grouped_folds(full)),
    }
    print({k: [(n, len(a), len(b)) for n, a, b in v] for k, v in schemes.items()})

    rows = []
    for name, (builder, params) in MODELS.items():
        for p in grid(params):
            for scheme, folds in schemes.items():
                for fold, tr, te in folds:
                    if len(te) == 0:
                        continue
                    sc = scores(Wrapped(fit(builder, p, tr)), te)
                    rows.append({"model": name, "scheme": scheme, "fold": fold, "params": json.dumps(p), **sc})
        print("done", name, flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "01_cv_results.csv", index=False)

    # mean over folds per (model, params, scheme); pick best on deviance per scheme
    agg = (res.groupby(["model", "params", "scheme"])[["deviance", "mae_rate_w", "bias"]].mean().reset_index())
    best = agg.loc[agg.groupby(["model", "scheme"])["deviance"].idxmin()].sort_values(["scheme", "deviance"])
    best.to_csv(OUT / "01_best_params.csv", index=False)
    print(best.to_string(index=False))

    plot_param_effects(agg)
    plot_load_curves(full, best)


def plot_param_effects(agg):
    """Deviance against each parameter (others at their best): how the parameters move the fit."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, model in zip(axes.ravel(), ["glm", "hgb", "rf"] * 2):
        pass
    for r, scheme in enumerate(["temporal", "unit"]):
        for c, model in enumerate(["glm", "hgb", "rf"]):
            a = agg[(agg.model == model) & (agg.scheme == scheme)].copy()
            P = pd.json_normalize(a["params"].map(json.loads)).set_index(a.index)
            ax = axes[r, c]
            for col in P.columns:
                g = a.assign(v=P[col].astype(str)).groupby("v")["deviance"].min()
                order = sorted(g.index, key=lambda s: float(s) if s not in ("None",) else 1e9)
                ax.plot([f"{col}={v}" for v in order], g[order].values, "o-", label=col)
            ref = agg[(agg.model == "glm_noload") & (agg.scheme == scheme)]["deviance"].min()
            ax.axhline(ref, color="k", ls="--", lw=1, label="no-load GLM")
            ax.set_title(f"{model} - {scheme} CV"); ax.set_ylabel("mean Poisson deviance")
            ax.tick_params(axis="x", rotation=90, labelsize=6)
    axes[0, 0].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(FIG / "01_param_effects.png", dpi=130); plt.close(fig)


def plot_load_curves(full, best):
    """Fit the best (unit-CV) params on all data; partial dependence of fatality on relative load, per panel."""
    grid_load = np.linspace(0.02, 1.2, 40)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    curves = []
    for ax, model in zip(axes, ["glm", "hgb", "rf"]):
        row = best[(best.model == model) & (best.scheme == "unit")].iloc[0]
        m = fit(MODELS[model][0], json.loads(row["params"]), full)
        for panel_name, g in full.groupby("panel"):
            ys = []
            for L in grid_load:
                x = g.copy(); x["rel_load"] = L
                ys.append(np.average(m.predict(x[FEATURES]), weights=g["exposure"]))
            ax.plot(grid_load, ys, label=panel_name)
            curves += [{"model": model, "panel": panel_name, "rel_load": L, "cfr": y} for L, y in zip(grid_load, ys)]
        ax.set_title(f"{model}: partial dependence of deaths/admission on relative load"); ax.set_xlabel("relative load")
    axes[0].set_ylabel("predicted deaths per admission"); axes[0].legend()
    fig.tight_layout(); fig.savefig(FIG / "01_load_curves.png", dpi=130); plt.close(fig)
    pd.DataFrame(curves).to_csv(OUT / "01_load_curves.csv", index=False)


if __name__ == "__main__":
    main()
