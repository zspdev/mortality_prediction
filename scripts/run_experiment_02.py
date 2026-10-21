"""Experiment 02: temporal models (distributed-lag Poisson, lag-feature boosting) for weekly deaths.

Outputs reports/experiments/02_*.csv and reports/figures/02_*.png. Run:
``UV_PROJECT_ENVIRONMENT=~/.venvs/hcw_ml uv run python scripts/run_experiment_02.py``
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
from hcw_ml.temporal import DistLag, HGBLag, count_scores, make_weekly_dataset

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
OUT, FIG = ROOT / "reports" / "experiments", ROOT / "reports" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

# name -> (class, kwargs fixed, grid)
MODELS = {
    "dlag_noload": (DistLag, {"n_knots": 0}, {"K": [4, 6, 8]}),
    "dlag": (DistLag, {}, {"K": [4, 6, 8], "n_knots": [4, 6], "alpha": [0.1, 1.0, 10.0]}),
    "dlag_time_noload": (DistLag, {"n_knots": 0, "K": 6}, {"time_df": [4, 6, 10]}),
    "dlag_time": (DistLag, {"K": 6, "n_knots": 4}, {"time_df": [4, 6, 10], "alpha": [1.0, 10.0]}),
    "hgb_lag": (HGBLag, {"K": 6}, {"learning_rate": [0.05, 0.1], "max_depth": [2, 3, 5],
                                    "max_iter": [100, 300], "min_samples_leaf": [50, 200]}),
}
N_BOOT = 30


def main():
    panel = build_panel()
    full = make_weekly_dataset(add_relative_load(panel))
    print(len(full), full.groupby("panel").size().to_dict())
    schemes = {"temporal": list(temporal_folds(panel, make_weekly_dataset)), "unit": list(grouped_folds(full))}
    print({k: [(n, len(a), len(b)) for n, a, b in v] for k, v in schemes.items()}, flush=True)

    rows = []
    for name, (cls, fixed, params) in MODELS.items():
        for p in grid(params):
            kw = {**fixed, **p}
            for scheme, folds in schemes.items():
                for fold, tr, te in folds:
                    if len(te) == 0:
                        continue
                    m = cls(**kw).fit(tr)
                    rows.append({"model": name, "scheme": scheme, "fold": fold, "params": json.dumps(kw),
                                 "converged": getattr(m, "converged_", True), **count_scores(m, te)})
        print("done", name, flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "02_cv_results.csv", index=False)

    agg = res.groupby(["model", "params", "scheme"])[["deviance", "deviance_hi", "bias", "bias_hi"]].mean().reset_index()
    best = agg.loc[agg.groupby(["model", "scheme"])["deviance"].idxmin()].sort_values(["scheme", "deviance"])
    best.to_csv(OUT / "02_best_params.csv", index=False)
    print(best.to_string(index=False))
    print("non-converged fits:", int((~res["converged"].astype(bool)).sum()), "of", len(res))

    plot_param_effects(agg)
    final_dlag(panel, full, best)


def plot_param_effects(agg):
    models = ["dlag", "dlag_time", "hgb_lag"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for r, scheme in enumerate(["temporal", "unit"]):
        for c, model in enumerate(models):
            a = agg[(agg.model == model) & (agg.scheme == scheme)].copy()
            P = pd.json_normalize(a["params"].map(json.loads)).set_index(a.index)
            ax = axes[r, c]
            for col in [x for x in P.columns if P[x].nunique() > 1]:
                g = a.assign(v=P[col].astype(str)).groupby("v")["deviance"].min()
                order = sorted(g.index, key=float)
                ax.plot([f"{col}={v}" for v in order], g[order].values, "o-")
            for ref_model, ls in [("dlag_noload", "--"), ("dlag_time_noload", ":")]:
                ref = agg[(agg.model == ref_model) & (agg.scheme == scheme)]["deviance"].min()
                ax.axhline(ref, color="k", ls=ls, lw=1, label=ref_model)
            ax.set_title(f"{model} - {scheme} CV"); ax.set_ylabel("mean count deviance")
            ax.tick_params(axis="x", rotation=90, labelsize=6)
    axes[0, 0].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(FIG / "02_param_effects.png", dpi=130); plt.close(fig)


def final_dlag(panel, full, best):
    """Fit the best DistLag variants on all data: delay h, multiplier m(L) with unit-bootstrap bands, and the
    minimum-admissions sensitivity check."""
    loads = np.linspace(0.05, 1.2, 40)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    units = full.groupby(["panel", "unit"]).ngroup()
    rng = np.random.default_rng(0)
    out = []
    for name, color in [("dlag", "C0"), ("dlag_time", "C1")]:
        kw = json.loads(best[(best.model == name) & (best.scheme == "unit")].iloc[0]["params"])
        m = DistLag(**kw).fit(full)
        axes[0].plot(range(m.K + 1), m.delay_, "o-", color=color, label=f"{name} K={m.K}")
        boots = []
        ids = np.unique(units)
        for _ in range(N_BOOT):
            pick = rng.choice(ids, len(ids))
            d = pd.concat([full[units == u] for u in pick], ignore_index=True)
            boots.append(DistLag(**kw).fit(d).multiplier(loads))
        lo, hi = np.percentile(boots, [5, 95], axis=0)
        mult = m.multiplier(loads)
        axes[1].plot(loads, mult, color=color, label=name); axes[1].fill_between(loads, lo, hi, color=color, alpha=0.2)
        out += [{"model": name, "rel_load": L, "multiplier": a, "lo90": b, "hi90": c}
                for L, a, b, c in zip(loads, mult, lo, hi)]
        # sensitivity: drop weeks with few admissions in the lag window
        for min_adm in (0, 20, 50):
            d = make_weekly_dataset(add_relative_load(panel), min_adm=min_adm)
            mm = DistLag(**kw).fit(d)
            if name == "dlag":
                axes[2].plot(loads, mm.multiplier(loads), label=f"min admissions {min_adm} (n={len(d)})")
    axes[0].set_title("fitted admission-to-death delay h_k (weeks)"); axes[0].set_xlabel("lag k"); axes[0].legend()
    axes[1].axhline(1, color="k", lw=0.5); axes[1].set_title("load multiplier m(L)/m(0.25), 90% unit-bootstrap band")
    axes[1].set_xlabel("relative load"); axes[1].legend()
    axes[2].axhline(1, color="k", lw=0.5); axes[2].set_title("dlag: sensitivity to minimum admissions")
    axes[2].set_xlabel("relative load"); axes[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "02_delay_and_multiplier.png", dpi=130); plt.close(fig)
    pd.DataFrame(out).to_csv(OUT / "02_multiplier.csv", index=False)


if __name__ == "__main__":
    main()
