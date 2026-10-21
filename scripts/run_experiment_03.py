"""Experiment 03: negative-binomial distributed-lag model and per-panel load multipliers.

Outputs reports/experiments/03_*.csv and reports/figures/03_multiplier_by_panel.png. Run:
``UV_PROJECT_ENVIRONMENT=~/.venvs/hcw_ml uv run python scripts/run_experiment_03.py``
"""
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hcw_ml.evaluate import grid, grouped_folds, temporal_folds
from hcw_ml.features import add_relative_load, build_panel
from hcw_ml.temporal import DistLag, count_scores, make_weekly_dataset

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
OUT, FIG = ROOT / "reports" / "experiments", ROOT / "reports" / "figures"

MODELS = {
    "nb_time_noload": ({"n_knots": 0, "K": 6, "family": "nb"}, {"time_df": [4, 6, 10]}),
    "nb_time": ({"K": 6, "n_knots": 4, "family": "nb"}, {"time_df": [4, 6, 10], "alpha": [1.0, 10.0]}),
    "pois_time": ({"K": 6, "n_knots": 4}, {"time_df": [4, 6, 10], "alpha": [1.0, 10.0]}),
}
N_BOOT = 40
LOADS = np.linspace(0.05, 1.2, 40)


def cv(panel, full):
    schemes = {"temporal": list(temporal_folds(panel, make_weekly_dataset)), "unit": list(grouped_folds(full))}
    rows = []
    for name, (fixed, params) in MODELS.items():
        for p in grid(params):
            kw = {**fixed, **p}
            for scheme, folds in schemes.items():
                for fold, tr, te in folds:
                    if len(te):
                        m = DistLag(**kw).fit(tr)
                        rows.append({"model": name, "scheme": scheme, "fold": fold, "params": json.dumps(kw),
                                     "converged": m.converged_, "r": m.r_, **count_scores(m, te)})
        print("done", name, flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "03_cv_results.csv", index=False)
    agg = res.groupby(["model", "params", "scheme"])[["deviance", "deviance_hi", "bias", "nb_logscore"]].mean().reset_index()
    best = agg.loc[agg.groupby(["model", "scheme"])["deviance"].idxmin()].sort_values(["scheme", "deviance"])
    best.to_csv(OUT / "03_best_params.csv", index=False)
    print(best.to_string(index=False))
    print("non-converged:", int((~res["converged"].astype(bool)).sum()), "of", len(res))
    return best


def clusters(d):
    """Resampling clusters: units for ES; unit x 26-week block for single-unit / few-unit panels (BE, Rio)."""
    blk = ((d["week"] - d["week"].min()).dt.days // 182).astype(str)
    return np.where(d["panel"].str.startswith("ES"), d["panel"] + "|" + d["unit"].astype(str),
                    d["panel"] + "|" + d["unit"].astype(str) + "|" + blk)


def by_panel(full, kw):
    """Fit the NB model pooled and per panel; bootstrap bands over clusters. Returns long table."""
    rng = np.random.default_rng(0)
    out = []
    groups = {"pooled": full, **{p: g for p, g in full.groupby("panel")}}
    for name, d in groups.items():
        d = d.reset_index(drop=True)
        try:
            m = DistLag(**kw).fit(d)
        except Exception as e:
            print("fail", name, e); continue
        cl = pd.Series(clusters(d)); ids = cl.unique()
        idx = {c: np.flatnonzero(cl.values == c) for c in ids}
        boots = []
        for _ in range(N_BOOT):
            pick = rng.choice(ids, len(ids))
            boots.append(DistLag(**kw).fit(d.iloc[np.concatenate([idx[c] for c in pick])]).multiplier(LOADS))
        lo, hi = np.percentile(boots, [5, 95], axis=0)
        mult = m.multiplier(LOADS)
        print(f"{name}: n={len(d)} units={d['unit'].nunique()} r={m.r_:.2f} conv={m.converged_} "
              f"m(1.0)={np.interp(1.0, LOADS, mult):.2f} [{np.interp(1.0, LOADS, lo):.2f},{np.interp(1.0, LOADS, hi):.2f}]"
              f" delay_peak={int(np.argmax(m.delay_))}", flush=True)
        out += [{"group": name, "rel_load": L, "multiplier": a, "lo90": b, "hi90": c, "n": len(d), "r": m.r_,
                 "converged": m.converged_} for L, a, b, c in zip(LOADS, mult, lo, hi)]
    return pd.DataFrame(out)


def main():
    panel = build_panel()
    full = make_weekly_dataset(add_relative_load(panel))
    best = cv(panel, full)
    kw = json.loads(best[(best.model == "nb_time") & (best.scheme == "unit")].iloc[0]["params"])
    kw_era = {"K": 6, "n_knots": 4, "alpha": 1.0, "family": "nb"}
    tabs = []
    for label, k in [("nb_time", kw), ("nb_era", kw_era)]:
        print("==", label, k)
        t = by_panel(full, k); t.insert(0, "model", label); tabs.append(t)
    tab = pd.concat(tabs); tab.to_csv(OUT / "03_multiplier_by_panel.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, label in zip(axes, ["nb_time", "nb_era"]):
        for g, d in tab[tab.model == label].groupby("group"):
            ax.plot(d.rel_load, d.multiplier, label=g, lw=2.5 if g == "pooled" else 1.2)
            if g == "pooled":
                ax.fill_between(d.rel_load, d.lo90, d.hi90, alpha=0.15)
        ax.axhline(1, color="k", lw=0.5); ax.set_title(f"{label}: m(L)/m(0.25) by panel"); ax.set_xlabel("relative load"); ax.set_ylim(0, 4)
    axes[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(FIG / "03_multiplier_by_panel.png", dpi=130)


if __name__ == "__main__":
    main()
