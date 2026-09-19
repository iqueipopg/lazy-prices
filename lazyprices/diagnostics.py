"""Diagnostics beyond the calendar-time portfolio: event-time abnormal
returns, a placebo test with shuffled scores, and a bootstrap confidence
interval for the Sharpe ratio."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, evaluation, portfolio

MONTHS = 12


# --------------------------------------------------------------------------- #
# Event time
# --------------------------------------------------------------------------- #
def first_trading_day_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Price at the first trading day of every calendar month, indexed by
    month period."""
    p = prices.sort_index()
    first = p.groupby(p.index.to_period("M")).head(1)
    first.index = first.index.to_period("M")
    return first


def event_time_returns(
    ranked: pd.DataFrame,
    prices: pd.DataFrame,
    benchmark: str = config.BENCHMARK,
    horizon: int = MONTHS,
    lag_months: int = 0,
) -> pd.DataFrame:
    """Market-adjusted return of every ranked filing in each of the ``horizon``
    months after formation.

    Event month ``k`` runs from the first trading day of month
    ``filing_month + lag + k`` to the first trading day of the next month, so
    month 1 starts at the formation date used by the portfolios. Returns one
    row per (filing, event month) with the stock return, the benchmark return
    and their difference (``abnormal``).
    """
    first = first_trading_day_prices(prices)
    rows = []
    for r in ranked.itertuples():
        if r.ticker not in first.columns:
            continue
        m0 = pd.Timestamp(r.filing_date).to_period("M") + lag_months
        for k in range(1, horizon + 1):
            start, end = m0 + k, m0 + k + 1
            if end not in first.index:
                break
            ps, pe = first.at[start, r.ticker], first.at[end, r.ticker]
            bs, be = first.at[start, benchmark], first.at[end, benchmark]
            if pd.isna(ps) or pd.isna(pe) or pd.isna(bs) or pd.isna(be):
                continue
            ret, bench = pe / ps - 1.0, be / bs - 1.0
            rows.append(
                {
                    "ticker": r.ticker,
                    "fiscal_year": r.fiscal_year,
                    "quantile": int(r.quantile),
                    "month": k,
                    "ret": ret,
                    "bench": bench,
                    "abnormal": ret - bench,
                }
            )
    return pd.DataFrame(rows)


def event_time_car(events: pd.DataFrame, n_quantiles: int) -> pd.DataFrame:
    """Cumulative average abnormal return by event month for the bottom and top
    groups and their difference. Standard errors treat each fiscal-year cohort
    as one observation (Fama-MacBeth over cohorts), which is conservative
    because filings inside a cohort overlap in calendar time."""
    lo, hi = 1, n_quantiles
    per_cohort = events.groupby(["fiscal_year", "quantile", "month"])["abnormal"].mean().unstack("quantile")
    per_cohort = per_cohort.dropna(subset=[lo, hi])
    per_cohort["LS"] = per_cohort[hi] - per_cohort[lo]
    car = per_cohort.groupby(level="fiscal_year").cumsum()  # cumulate within cohort
    out = pd.DataFrame(index=pd.Index(sorted(events["month"].unique()), name="month"))
    g = car.groupby(level="month")
    out["Q_low_car"] = g[lo].mean()
    out["Q_high_car"] = g[hi].mean()
    out["LS_car"] = g["LS"].mean()
    out["LS_se"] = g["LS"].std(ddof=1) / np.sqrt(g["LS"].count())
    out["n_cohorts"] = g["LS"].count()
    out["n_filings"] = events.groupby("month")["ticker"].count()
    return out


# --------------------------------------------------------------------------- #
# Placebo: shuffle scores within cohorts
# --------------------------------------------------------------------------- #
def shuffle_within_cohorts(sim: pd.DataFrame, score_col: str, rng: np.random.Generator) -> pd.DataFrame:
    """Permute ``score_col`` within each fiscal-year cohort. The set of scores
    each cohort has, and therefore every quantile breakpoint, is unchanged;
    only which company gets which score is randomised."""
    out = sim.copy()
    for _, idx in out.groupby("fiscal_year").groups.items():
        vals = out.loc[idx, score_col].to_numpy()
        out.loc[idx, score_col] = rng.permutation(vals)
    return out


def placebo(
    sim: pd.DataFrame,
    daily_returns: pd.DataFrame,
    factors_m: pd.DataFrame,
    spec: dict,
    n_draws: int = 200,
    seed: int = 0,
    cost_bps: float = 0.0,
) -> pd.DataFrame:
    """Re-run the strategy ``n_draws`` times on shuffled scores. Returns one
    row per draw with the long-short Sharpe, mean return and FF5+MOM alpha
    t-statistic, i.e. the distribution of the headline statistics under the
    null that the text carries no information."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_draws):
        fake = shuffle_within_cohorts(sim, spec["score_col"], rng)
        _, monthly = portfolio.run_variant(fake, daily_returns, cost_bps=cost_bps, **spec)
        reg = evaluation.alpha_regression(monthly["LS"], factors_m.reindex(monthly.index), "FF5+MOM")
        rows.append(
            {
                "draw": d,
                "sharpe": evaluation.sharpe_annual(monthly["LS"]),
                "mean_ann_pct": monthly["LS"].mean() * MONTHS * 100,
                "alpha_FF5+MOM": reg["alpha_ann_pct"],
                "t_FF5+MOM": reg["t_stat"],
            }
        )
    return pd.DataFrame(rows)


def percentile_of(value: float, draws: pd.Series) -> float:
    """Share of placebo draws below ``value``."""
    return float((draws < value).mean())


# --------------------------------------------------------------------------- #
# Bootstrap confidence interval for the Sharpe ratio
# --------------------------------------------------------------------------- #
def sharpe_bootstrap_ci(
    returns: pd.Series, n_boot: int = 2000, block: int = 6, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile confidence interval for the annualised Sharpe ratio from a
    circular block bootstrap of monthly returns (block length ``block``
    months keeps short-run autocorrelation)."""
    r = returns.dropna().to_numpy()
    n = len(r)
    if n < 2 * block:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n] % n
    samples = r[idx]
    sr = samples.mean(axis=1) / samples.std(axis=1, ddof=1) * np.sqrt(MONTHS)
    lo, hi = np.percentile(sr, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)
