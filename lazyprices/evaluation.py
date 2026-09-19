"""Performance evaluation: factor alphas with Newey-West errors, Sharpe ratio,
drawdown, per-quantile table, and the selection-bias corrections (deflated
Sharpe ratio, probability of backtest overfitting) from the ``bto`` package."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

from bto import cscv, deflated_sharpe_ratio, effective_number_of_trials, higher_moments, sharpe_ratio

FACTOR_MODELS = {
    "CAPM": ["Mkt-RF"],
    "FF3": ["Mkt-RF", "SMB", "HML"],
    "FF5+MOM": ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"],
}
MONTHS = 12


def monthly_factors(daily_factors: pd.DataFrame) -> pd.DataFrame:
    """Compound daily factor returns within calendar months."""
    g = daily_factors.index.to_period("M")
    m = (1.0 + daily_factors).groupby(g).prod() - 1.0
    m.index = m.index.to_timestamp(how="end").normalize()
    m.index.name = "Date"
    return m


def newey_west_lags(n_obs: int) -> int:
    """Newey and West (1994) automatic bandwidth ``floor(4 (T/100)^(2/9))``."""
    return int(np.floor(4 * (n_obs / 100.0) ** (2.0 / 9.0)))


def alpha_regression(excess: pd.Series, factors: pd.DataFrame, model: str = "CAPM", lags: int | None = None) -> dict:
    """OLS of ``excess`` returns on the factors of ``model`` with HAC
    (Newey-West) standard errors. Alpha is reported annualised in percent."""
    cols = FACTOR_MODELS[model]
    df = pd.concat([excess.rename("y"), factors[cols]], axis=1).dropna()
    if len(df) < len(cols) + 3:
        return {"model": model, "alpha_ann_pct": np.nan, "t_stat": np.nan, "n_obs": len(df)}
    lags = newey_west_lags(len(df)) if lags is None else lags
    x = sm.add_constant(df[cols])
    res = sm.OLS(df["y"], x).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    out = {
        "model": model,
        "alpha_ann_pct": float(res.params["const"] * MONTHS * 100),
        "t_stat": float(res.tvalues["const"]),
        "p_value": float(res.pvalues["const"]),
        "n_obs": int(res.nobs),
        "r2": float(res.rsquared),
        "nw_lags": lags,
    }
    for c in cols:
        out[f"beta_{c}"] = float(res.params[c])
    return out


def sharpe_annual(returns: pd.Series) -> float:
    r = returns.dropna()
    if len(r) < 2 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * np.sqrt(MONTHS))


def max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough loss of the compounded return series (negative)."""
    wealth = (1.0 + returns.dropna()).cumprod()
    peak = wealth.cummax()
    return float(((wealth / peak) - 1.0).min())


def mean_t_stat(returns: pd.Series, lags: int | None = None) -> float:
    r = returns.dropna()
    if len(r) < 3:
        return float("nan")
    lags = newey_west_lags(len(r)) if lags is None else lags
    res = sm.OLS(r.values, np.ones((len(r), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(res.tvalues[0])


def summary_stats(returns: pd.Series) -> dict:
    r = returns.dropna()
    return {
        "n_months": int(len(r)),
        "mean_ann_pct": float(r.mean() * MONTHS * 100),
        "vol_ann_pct": float(r.std(ddof=1) * np.sqrt(MONTHS) * 100),
        "sharpe": sharpe_annual(r),
        "t_mean_nw": mean_t_stat(r),
        "max_drawdown_pct": max_drawdown(r) * 100,
        "hit_rate": float((r > 0).mean()),
    }


def evaluate_portfolio(monthly: pd.DataFrame, factors_m: pd.DataFrame, n_quantiles: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-quantile table (excess returns over RF) and long-short alpha table."""
    fac = factors_m.reindex(monthly.index)
    rows = []
    for q in range(1, n_quantiles + 1):
        col = f"Q{q}"
        if col not in monthly:
            continue
        excess = monthly[col] - fac["RF"]
        row = {"portfolio": col, **summary_stats(excess)}
        for model in FACTOR_MODELS:
            reg = alpha_regression(excess, fac, model)
            row[f"alpha_{model}"] = reg["alpha_ann_pct"]
            row[f"t_{model}"] = reg["t_stat"]
        rows.append(row)
    for col in ("LS", "LS_net"):
        row = {"portfolio": col, **summary_stats(monthly[col])}
        for model in FACTOR_MODELS:
            reg = alpha_regression(monthly[col], fac, model)
            row[f"alpha_{model}"] = reg["alpha_ann_pct"]
            row[f"t_{model}"] = reg["t_stat"]
        rows.append(row)
    table = pd.DataFrame(rows).set_index("portfolio")

    alphas = pd.DataFrame(
        [alpha_regression(monthly[col], fac, model) | {"portfolio": col}
         for col in ("LS", "LS_net") for model in FACTOR_MODELS]
    ).set_index(["portfolio", "model"])
    return table, alphas


# --------------------------------------------------------------------------- #
# Selection-bias corrections
# --------------------------------------------------------------------------- #
@dataclass
class OverfittingReport:
    n_trials: int
    n_effective: float
    n_obs: int
    best_trial: str
    best_sharpe_annual: float
    median_sharpe_annual: float
    expected_max_sharpe_annual_raw: float
    dsr_raw: float
    expected_max_sharpe_annual_eff: float
    dsr_eff: float
    pbo: float
    prob_oos_loss: float
    cscv_splits: int
    cscv_combinations: int

    def to_json(self, path) -> None:
        with open(path, "w") as fh:
            json.dump(asdict(self), fh, indent=2)


def overfitting_analysis(trials: pd.DataFrame, n_splits: int = 16) -> OverfittingReport:
    """``trials``: months x variants matrix of long-short monthly returns.

    The deflated Sharpe ratio tests the best variant against the Sharpe that
    the luckiest of N zero-skill variants would show, given the dispersion of
    Sharpe estimates across trials; it is reported for the raw trial count and
    for the participation-ratio estimate of independent trials. PBO is the
    fraction of CSCV partitions in which the in-sample best variant ranks in
    the bottom half out of sample.
    """
    x = trials.dropna(how="any")
    sr = np.array([sharpe_ratio(x[c].to_numpy()) for c in x.columns])  # per month
    best_i = int(np.nanargmax(sr))
    best = x.columns[best_i]
    skew, kurt = higher_moments(x[best].to_numpy())
    var_sr = float(np.nanvar(sr, ddof=1)) if len(sr) > 1 else 0.0
    n_eff = effective_number_of_trials(x.to_numpy())
    dsr_raw, sr_star_raw = deflated_sharpe_ratio(sr[best_i], var_sr, len(sr), len(x), skew, kurt)
    n_eff_int = max(1, int(round(n_eff)))
    dsr_eff, sr_star_eff = deflated_sharpe_ratio(sr[best_i], var_sr, n_eff_int, len(x), skew, kurt)
    while n_splits > 2 and len(x) < 2 * n_splits:
        n_splits -= 2
    res = cscv(x.to_numpy(), n_splits=n_splits)
    ann = np.sqrt(MONTHS)
    return OverfittingReport(
        n_trials=int(len(sr)),
        n_effective=float(n_eff),
        n_obs=int(len(x)),
        best_trial=str(best),
        best_sharpe_annual=float(sr[best_i] * ann),
        median_sharpe_annual=float(np.nanmedian(sr) * ann),
        expected_max_sharpe_annual_raw=float(sr_star_raw * ann),
        dsr_raw=float(dsr_raw),
        expected_max_sharpe_annual_eff=float(sr_star_eff * ann),
        dsr_eff=float(dsr_eff),
        pbo=float(res.pbo),
        prob_oos_loss=float(res.prob_oos_loss),
        cscv_splits=int(res.n_splits),
        cscv_combinations=int(res.n_combinations),
    )
