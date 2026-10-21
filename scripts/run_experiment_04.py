"""Experiment 04: stock version (deaths per patient-week) for IT and HU, with ES / BE / Rio as a cross-check.

Exposure is the hospital census instead of admissions (PLAN.md target B'). Outputs reports/experiments/04_*.csv and
reports/figures/04_multiplier_stock.png. Run:
``UV_PROJECT_ENVIRONMENT=~/.venvs/hcw_ml uv run python scripts/run_experiment_04.py``
"""
import warnings
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hcw_ml.evaluate import grouped_folds, temporal_folds
from hcw_ml.features import add_relative_load, build_panel
from hcw_ml.temporal import DistLag, count_scores, make_weekly_dataset

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
OUT, FIG = ROOT / "reports" / "experiments", ROOT / "reports" / "figures"
PANELS = ["IT-NUTS2", "HU-country", "ES-NUTS3", "BE-NUTS1", "BR-Rio"]
K = 4
SPECS = {  # name -> kwargs
    "noload_time": {"n_knots": 0, "time_df": 4},
    "load_time": {"n_knots": 4, "time_df": 4, "alpha": 1.0},
    "load_time_a10": {"n_knots": 4, "time_df": 4, "alpha": 10.0},
    "noload_era": {"n_knots": 0},
    "load_era": {"n_knots": 4, "alpha": 1.0},
}
N_BOOT = 40
LOADS = np.linspace(0.05, 1.2, 40)


def clusters(d, panel):
    if panel in ("IT-NUTS2", "ES-NUTS3"):
        return (d["unit"].astype(str)).to_numpy()
    blk = ((d["week"] - d["week"].min()).dt.days // 182).astype(str)
    return (d["unit"].astype(str) + "|" + blk).to_numpy()


def main():
    panel = build_panel()
    rows, mults = [], []
    for pn in PANELS:
        build = partial(make_weekly_dataset, kmax=K, exposure="census", panels=[pn])
        full = build(add_relative_load(panel))
        print(pn, "rows", len(full), "units", full["unit"].nunique(), flush=True)
        folds = [("temporal", f) for f in temporal_folds(panel, build)]
        if full["unit"].nunique() >= 5:
            folds += [("unit", f) for f in grouped_folds(full)]
        for name, spec in SPECS.items():
            for scheme, (fold, tr, te) in folds:
                if len(te) and len(tr):
                    m = DistLag(K=K, family="nb", **spec).fit(tr)
                    rows.append({"panel": pn, "model": name, "scheme": scheme, "fold": fold,
                                 "converged": m.converged_, "r": m.r_, **count_scores(m, te)})
        # multiplier on all data, block bootstrap band
        rng = np.random.default_rng(0)
        cl = pd.Series(clusters(full, pn)); ids = cl.unique(); idx = {c: np.flatnonzero(cl.values == c) for c in ids}
        for name in ("load_time", "load_era"):
            kw = {"K": K, "family": "nb", **SPECS[name]}
            m = DistLag(**kw).fit(full)
            boots = [DistLag(**kw).fit(full.iloc[np.concatenate([idx[c] for c in rng.choice(ids, len(ids))])]).multiplier(LOADS)
                     for _ in range(N_BOOT)]
            lo, hi = np.percentile(boots, [5, 95], axis=0); mu = m.multiplier(LOADS)
            at = lambda a: float(np.interp(1.0, LOADS, a))
            print(f"  {name}: m(1.0)={at(mu):.2f} [{at(lo):.2f},{at(hi):.2f}] r={m.r_:.1f} conv={m.converged_} "
                  f"delay={np.round(m.delay_, 2)}", flush=True)
            mults += [{"panel": pn, "model": name, "rel_load": L, "multiplier": a, "lo90": b, "hi90": c, "n": len(full),
                       "r": m.r_, "converged": m.converged_} for L, a, b, c in zip(LOADS, mu, lo, hi)]
    res = pd.DataFrame(rows); res.to_csv(OUT / "04_cv_results.csv", index=False)
    pd.DataFrame(mults).to_csv(OUT / "04_multiplier_stock.csv", index=False)
    agg = res.groupby(["panel", "scheme", "model"])[["deviance", "nb_logscore", "bias"]].mean().round(3)
    print(agg.to_string())
    non = int((~res["converged"].astype(bool)).sum()); print("non-converged:", non, "of", len(res))
    mt = pd.DataFrame(mults)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, name in zip(axes, ["load_time", "load_era"]):
        for pn, d in mt[mt.model == name].groupby("panel"):
            ax.plot(d.rel_load, d.multiplier, label=pn, lw=2.5 if pn in ("IT-NUTS2", "HU-country") else 1)
        ax.axhline(1, color="k", lw=0.5); ax.set_ylim(0, 4); ax.set_title(f"{name}: m(L)/m(0.25), stock exposure"); ax.set_xlabel("relative load")
    axes[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(FIG / "04_multiplier_stock.png", dpi=130)


if __name__ == "__main__":
    main()
