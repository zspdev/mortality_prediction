"""Temporal models for weekly COVID deaths given lagged hospital admissions and load (PLAN.md §6, exp. 02).

Instead of a fixed-window ratio, deaths in week t are modelled as admissions from the previous K weeks, delayed by a
learned distribution h and scaled by a load-dependent multiplier m(L):

    E[deaths_t] = exp(z_t) * ( eps + sum_{k=0..K} h_k * adm_{t-k} * m(load_{t-k}) )

h = softmax(theta) is the admission-to-death delay distribution, m(L) = exp(spline(L) @ beta), and z_t holds panel
effects plus either era effects or a smooth time trend per panel (drift from vaccination/variants). The negative
log-likelihood is Poisson with a ridge penalty on beta and a second-difference penalty on theta; the gradient is analytic.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import digamma, gammaln
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_poisson_deviance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer

from hcw_ml.models import ERA_BREAKS

KMAX = 8          # lags kept in the dataset, so every model sees the same rows
EPS = 0.1         # background deaths not traced to tracked admissions (admission-equivalents)
T0 = pd.Timestamp("2020-01-01")


STOCK_PANELS = ("IT-NUTS2", "HU-country")  # no admissions: exposure is the hospital census (patient-weeks)


def make_weekly_dataset(panel: pd.DataFrame, kmax: int = KMAX, min_adm: float = 0.0,
                        exposure: str = "admissions", panels=None) -> pd.DataFrame:
    """One row per unit-week with lagged admissions ``adm_k`` and relative load ``load_k`` (k = 0..kmax).

    ``exposure="census"`` is the stock version (B'): the census replaces admissions as the exposure, so the model
    describes deaths per patient-week (for IT and HU, which have no admissions); ``panels`` restricts the panels.
    Rows need observed deaths and complete lags. ``min_adm`` drops weeks whose lag window holds fewer
    admissions (sensitivity check against tiny denominators)."""
    p = panel[panel[exposure].notna() & (panel["panel"] != "ES-comarca")]
    if panels is not None:
        p = p[p["panel"].isin(panels)]
    p = p.sort_values(["panel", "unit", "week"]).copy()
    # Date-based lags (merge on week - k), so a missing week gives NaN instead of silently pulling in an older one.
    base = p[["panel", "unit", "week", exposure, "rel_load"]].rename(columns={exposure: "admissions"})
    for k in range(kmax + 1):
        lag = base.assign(week=base["week"] + pd.Timedelta(weeks=k)).rename(
            columns={"admissions": f"adm_{k}", "rel_load": f"load_{k}"})
        p = p.merge(lag, on=["panel", "unit", "week"], how="left")
    lag_cols = [f"adm_{k}" for k in range(kmax + 1)] + [f"load_{k}" for k in range(kmax + 1)]
    p = p.dropna(subset=["deaths"] + lag_cols)
    p = p[p[[f"adm_{k}" for k in range(kmax + 1)]].sum(axis=1) >= max(min_adm, 1e-9)].copy()
    p["era"] = sum((p["week"] >= b).astype(int) for b in ERA_BREAKS).astype(str)
    p["t"] = (p["week"] - T0).dt.days / 365.25
    p["load_growth"] = p["load_0"] - p["load_1"]
    return p.reset_index(drop=True)


def _lagmat(d, prefix, K):
    return d[[f"{prefix}_{k}" for k in range(K + 1)]].to_numpy(float)


class DistLag:
    """Distributed-lag Poisson model with a learned delay h and a load multiplier m(L).

    n_knots = 0 gives the no-load reference (m = 1). time_df > 0 replaces era effects by a per-panel smooth trend."""

    def __init__(self, K=6, n_knots=5, alpha=1.0, time_df=0, smooth_h=0.1, family="poisson"):
        self.K, self.n_knots, self.alpha, self.time_df, self.smooth_h = K, n_knots, alpha, time_df, smooth_h
        self.family = family  # "poisson" or "nb" (negative binomial, Var = mu + mu^2 / r, r fitted)

    # -- design -----------------------------------------------------------------------------------------------
    def _basis(self, d):
        L = _lagmat(d, "load", self.K)
        n, k1 = L.shape
        B = self.spl_.transform(L.reshape(-1, 1)).reshape(n, k1, -1)
        return B

    def _Z(self, d):
        P = self.panel_enc_.transform(d[["panel"]])
        if self.time_df:
            T = self.time_spl_.transform(d[["t"]])
            return np.einsum("np,nq->npq", P, T).reshape(len(d), -1)
        return np.hstack([P, self.era_enc_.transform(d[["era"]])])

    def _unpack(self, w):
        K1, pb, pc = self.K + 1, self.pb_, self.pc_
        c_end = len(w) - (1 if self.family == "nb" else 0)
        return w[:K1], w[K1:K1 + pb], w[K1 + pb:c_end]

    def _forward(self, w, A, B, Z):
        theta, beta, c = self._unpack(w)
        h = np.exp(theta - theta.max()); h /= h.sum()
        e = np.exp(B @ beta) if self.pb_ else np.ones(A.shape)
        Ae = A * e
        S = Ae @ h + EPS
        ez = np.exp(np.clip(Z @ c, -30, 30))
        return h, e, Ae, S, ez

    def _nll(self, w, y, A, B, Z, D2):
        theta, beta, c = self._unpack(w)
        h, e, Ae, S, ez = self._forward(w, A, B, Z)
        mu = ez * S
        if self.family == "nb":
            lr = w[-1]; r = np.exp(np.clip(lr, -5, 12))
            nll = -np.sum(gammaln(y + r) - gammaln(r) + r * np.log(r) + y * np.log(mu) - (r + y) * np.log(r + mu))
            dmu = r * (mu - y) / (mu * (r + mu))
            g_lr = -r * np.sum(digamma(y + r) - digamma(r) + np.log(r) + 1 - np.log(r + mu) - (r + y) / (r + mu))
        else:
            nll = np.sum(mu - y * np.log(mu))
            dmu = 1 - y / mu
        pen = self.alpha * beta @ beta + self.smooth_h * np.sum((D2 @ theta) ** 2) + 1e-3 * c @ c
        # gradient
        dS = ez * dmu
        g_c = Z.T @ (mu * dmu) + 2e-3 * c
        g_h = Ae.T @ dS
        g_theta = h * (g_h - h @ g_h) + 2 * self.smooth_h * D2.T @ (D2 @ theta)
        g_beta = (np.einsum("n,nk,nkp->p", dS, Ae * h, B) if self.pb_ else np.zeros(0)) + 2 * self.alpha * beta
        g = [g_theta, g_beta, g_c] + ([np.array([g_lr])] if self.family == "nb" else [])
        return nll + pen, np.concatenate(g)

    # -- API ----------------------------------------------------------------------------------------------------
    def fit(self, d: pd.DataFrame):
        K = self.K
        self.panel_enc_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(d[["panel"]])
        if self.time_df:
            self.time_spl_ = SplineTransformer(n_knots=self.time_df, degree=3, extrapolation="constant").fit(d[["t"]])
        else:
            self.era_enc_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(d[["era"]])
        if self.n_knots:
            self.spl_ = SplineTransformer(n_knots=self.n_knots, degree=3, extrapolation="constant",
                                          include_bias=False).fit(_lagmat(d, "load", K).reshape(-1, 1))
            B = self._basis(d)
            self.pb_ = B.shape[2]
        else:
            B, self.pb_ = None, 0
        Z = self._Z(d)
        self.pc_ = Z.shape[1]
        A, y = _lagmat(d, "adm", K), d["deaths"].to_numpy(float)
        D2 = np.diff(np.eye(K + 1), n=2, axis=0) if K >= 2 else np.zeros((0, K + 1))
        rate = y.sum() / max(A.sum(), 1.0)
        # initialise the level at the pooled rate: panel columns (or every panel x spline column) carry log(rate)
        c0 = np.zeros(self.pc_)
        c0[: self.pc_ if self.time_df else len(self.panel_enc_.categories_[0])] = np.log(rate)
        w0 = np.concatenate([np.zeros(K + 1), np.zeros(self.pb_), c0] + ([[np.log(2.0)]] if self.family == "nb" else []))
        res = minimize(self._nll, w0, args=(y, A, B, Z, D2), jac=True, method="L-BFGS-B",
                       options={"maxiter": 500, "maxfun": 2000})
        self.w_, self.converged_, self.nll_ = res.x, res.success, res.fun
        self.theta_, self.beta_, self.c_ = self._unpack(res.x)
        self.r_ = float(np.exp(np.clip(res.x[-1], -5, 12))) if self.family == "nb" else None
        self.delay_ = np.exp(self.theta_ - self.theta_.max()); self.delay_ /= self.delay_.sum()
        return self

    def predict(self, d: pd.DataFrame) -> np.ndarray:
        A = _lagmat(d, "adm", self.K)
        B = self._basis(d) if self.pb_ else None
        Z = self._Z(d)
        _, _, _, S, ez = self._forward(self.w_, A, B, Z)
        return ez * S

    def multiplier(self, loads, ref=0.25) -> np.ndarray:
        """m(L) / m(ref): the load multiplier on fatality relative to a reference relative load."""
        if not self.pb_:
            return np.ones(len(loads))
        f = lambda x: np.exp(self.spl_.transform(np.asarray(x, float).reshape(-1, 1)) @ self.beta_)
        return f(loads) / f([ref])[0]


class HGBLag:
    """Gradient boosting (Poisson loss) on the death *rate* per tracked admission, weighted by admissions in the
    window, with lagged admission shares and lagged load as features."""

    def __init__(self, K=6, learning_rate=0.1, max_depth=3, max_iter=200, min_samples_leaf=50):
        self.K, self.kw = K, dict(learning_rate=learning_rate, max_depth=max_depth, max_iter=max_iter,
                                  min_samples_leaf=min_samples_leaf)

    def _X(self, d):
        K = self.K
        A = _lagmat(d, "adm", K)
        tot = A.sum(axis=1) + EPS  # same background floor as DistLag; the K-lag window can be empty
        X = pd.DataFrame(A / tot[:, None], columns=[f"share_{k}" for k in range(K + 1)], index=d.index)
        for k in range(min(K, 4) + 1):
            X[f"load_{k}"] = d[f"load_{k}"].values
        X["load_growth"] = d["load_growth"].values
        X["log_adm"] = np.log1p(tot)
        X["panel"], X["era"] = d["panel"].values, d["era"].values
        return X, tot

    def fit(self, d):
        X, tot = self._X(d)
        self.pipe_ = Pipeline([
            ("prep", ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["panel", "era"])],
                                       remainder=SimpleImputer(strategy="median"))),
            ("m", HistGradientBoostingRegressor(loss="poisson", random_state=0, **self.kw))])
        self.pipe_.fit(X, d["deaths"].to_numpy(float) / tot, m__sample_weight=tot)
        return self

    def predict(self, d):
        X, tot = self._X(d)
        return self.pipe_.predict(X) * tot


def count_scores(model, test: pd.DataFrame, high_load=0.5) -> dict:
    """Weekly-count Poisson deviance (all weeks and high-load weeks), and bias = predicted/observed deaths - 1."""
    pred = np.clip(model.predict(test), 1e-6, None)
    y = test["deaths"].to_numpy(float)
    hi = (test["load_0"] > high_load).to_numpy()
    out = {"deviance": mean_poisson_deviance(y, pred), "bias": float(pred.sum() / y.sum() - 1), "n": len(test)}
    r = getattr(model, "r_", None)
    if r:  # held-out negative-binomial log score per unit-week (higher is better)
        out["nb_logscore"] = float(np.mean(gammaln(y + r) - gammaln(r) - gammaln(y + 1) + r * np.log(r / (r + pred))
                                           + y * np.log(pred / (r + pred))))
    out["deviance_hi"] = mean_poisson_deviance(y[hi], pred[hi]) if hi.sum() > 10 else np.nan
    out["bias_hi"] = float(pred[hi].sum() / y[hi].sum() - 1) if hi.sum() > 10 else np.nan
    return out
