import numpy as np
import pandas as pd
from scipy.optimize import check_grad

from hcw_ml.temporal import DistLag, HGBLag, make_weekly_dataset


def _synthetic(n_units=20, weeks=120, K=6, seed=0, m=lambda L: 1 + 0.5 * L):
    rng = np.random.default_rng(seed)
    true_h = np.array([0.0, 0.05, 0.25, 0.35, 0.2, 0.1, 0.05])
    rows = []
    for u in range(n_units):
        t = np.arange(weeks)
        adm = rng.poisson(40 * (1 + np.sin(t / 8 + u)) ** 2 + 5)
        load = adm / adm.max() * rng.uniform(0.6, 1.4)
        deaths = np.zeros(weeks)
        for i in range(K, weeks):
            lam = 0.2 * sum(true_h[k] * adm[i - k] * m(load[i - k]) for k in range(K + 1))
            deaths[i] = rng.poisson(lam)
        rows.append(pd.DataFrame({"panel": "P", "unit": u, "week": pd.date_range("2020-06-01", periods=weeks, freq="W-MON"),
                                  "admissions": adm, "rel_load": load, "deaths": deaths, "census": adm, "icu_census": adm}))
    return pd.concat(rows, ignore_index=True), true_h


def test_dataset_lags_are_gap_safe_and_aligned():
    p, _ = _synthetic(n_units=2, weeks=30)
    gap = p.loc[10, "week"]
    full = p.copy()
    p = p.drop(index=10)  # a missing week must not shift later lags
    d = make_weekly_dataset(p, kmax=3)
    row = d[(d.unit == 0) & (d.week == full.loc[20, "week"])].iloc[0]
    assert row["adm_2"] == full.loc[18, "admissions"] and row["load_1"] == full.loc[19, "rel_load"]
    # weeks whose 3-week lag window includes the gap are dropped
    assert not ((d.unit == 0) & (d.week >= gap) & (d.week <= gap + pd.Timedelta(weeks=3))).any()


def test_analytic_gradient_matches_finite_differences():
    p, _ = _synthetic(n_units=3, weeks=40)
    d = make_weekly_dataset(p, kmax=6)
    m = DistLag(K=4, n_knots=4, alpha=0.5)
    m.fit(d)  # builds encoders
    A, y = np.asarray(d[[f"adm_{k}" for k in range(5)]], float), d["deaths"].to_numpy(float)
    B, Z = m._basis(d), m._Z(d)
    D2 = np.diff(np.eye(5), n=2, axis=0)
    rng = np.random.default_rng(1)
    w = m.w_ + 0.05 * rng.normal(size=m.w_.size)
    err = check_grad(lambda x: m._nll(x, y, A, B, Z, D2)[0], lambda x: m._nll(x, y, A, B, Z, D2)[1], w)
    assert err < 1e-3 * max(1.0, np.linalg.norm(m._nll(w, y, A, B, Z, D2)[1]))


def test_nb_gradient_matches_finite_differences():
    p, _ = _synthetic(n_units=3, weeks=40)
    d = make_weekly_dataset(p, kmax=6)
    m = DistLag(K=4, n_knots=4, alpha=0.5, family="nb")
    m.fit(d)
    A, y = np.asarray(d[[f"adm_{k}" for k in range(5)]], float), d["deaths"].to_numpy(float)
    B, Z = m._basis(d), m._Z(d)
    D2 = np.diff(np.eye(5), n=2, axis=0)
    w = m.w_ + 0.05 * np.random.default_rng(1).normal(size=m.w_.size)
    w[-1] = 1.0  # simulated counts are Poisson, so the fitted log r sits on its clip; test at a finite r
    err = check_grad(lambda x: m._nll(x, y, A, B, Z, D2)[0], lambda x: m._nll(x, y, A, B, Z, D2)[1], w)
    assert err < 1e-3 * max(1.0, np.linalg.norm(m._nll(w, y, A, B, Z, D2)[1]))
    assert m.r_ > 0


def test_delay_and_multiplier_are_recovered_on_simulated_data():
    p, true_h = _synthetic()
    d = make_weekly_dataset(p, kmax=8)
    m = DistLag(K=6, n_knots=4, alpha=0.1).fit(d)
    assert abs(m.delay_.sum() - 1) < 1e-9
    assert np.abs(m.delay_ - true_h).max() < 0.08
    mult = m.multiplier(np.array([0.25, 1.0]), ref=0.25)
    assert 1.1 < mult[1] < 1.6  # truth: (1 + 0.5) / (1 + 0.125) = 1.33


def test_noload_reference_has_flat_multiplier_and_hgb_predicts_counts():
    p, _ = _synthetic(n_units=6, weeks=60)
    d = make_weekly_dataset(p, kmax=8)
    assert (DistLag(K=4, n_knots=0).fit(d).multiplier(np.array([0.1, 1.0])) == 1).all()
    pred = HGBLag(K=4, max_iter=30).fit(d).predict(d)
    assert pred.shape == (len(d),) and (pred >= 0).all()
