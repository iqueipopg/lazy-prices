"""Portfolio formation for the Lazy Prices strategy.

Each 10-K filing gets a similarity score with the previous year's filing. Within
a *cohort* the scores are sorted into ``n_quantiles`` groups (1 = least
similar, i.e. most textual change; ``n`` = most similar). A stock enters the
portfolio of its group at the close of the first trading day of the month
following its filing date (plus an optional lag in months) and is held for
twelve months. Portfolios are equal-weighted across all stocks currently held,
so overlapping annual cohorts are averaged; the long-short portfolio is long
the most-similar group and short the least-similar one.

Two cohort definitions are available:

* ``"fiscal_year"`` (default): all filings for the same fiscal year are ranked
  together. Breakpoints therefore use the scores of firms that file later in
  the same season, which is a mild look-ahead in the *ranking* (never in
  prices) for early filers. This is what "sort firms every year" means.
* ``"trailing"``: each filing is ranked against the filings published in the
  twelve months up to and including its own filing date. Fully point-in-time,
  but group sizes are not balanced when the whole cross-section shifts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

HOLDING_MONTHS = 12
MIN_NAMES = 5  # a leg with fewer stocks does not produce a long-short return


# --------------------------------------------------------------------------- #
# Ranking
# --------------------------------------------------------------------------- #
def assign_quantiles(scores: pd.Series, n_quantiles: int) -> pd.Series:
    """Rank ``scores`` into ``n_quantiles`` balanced groups labelled 1..n,
    where 1 holds the lowest scores. Ties are broken by order of appearance so
    the groups are as equal as possible. Returns NaN where the score is NaN."""
    s = scores.dropna()
    out = pd.Series(np.nan, index=scores.index, dtype=float)
    if len(s) < n_quantiles:
        return out
    ranks = s.rank(method="first")
    out.loc[s.index] = pd.qcut(ranks, n_quantiles, labels=False) + 1
    return out


def rank_filings(
    sim: pd.DataFrame,
    score_col: str,
    n_quantiles: int = 5,
    cohort: str = "fiscal_year",
    min_cohort: int | None = None,
) -> pd.DataFrame:
    """Add a ``quantile`` column to the similarity table.

    ``sim`` needs ``ticker, fiscal_year, filing_date`` and ``score_col``.
    Rows without a score, or belonging to cohorts smaller than ``min_cohort``
    (default ``2 * n_quantiles``), get NaN.
    """
    min_cohort = min_cohort or 2 * n_quantiles
    df = sim.dropna(subset=[score_col]).copy()
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    df["quantile"] = np.nan
    if cohort == "fiscal_year":
        for _, grp in df.groupby("fiscal_year"):
            if len(grp) >= min_cohort:
                df.loc[grp.index, "quantile"] = assign_quantiles(grp[score_col], n_quantiles)
    elif cohort == "trailing":
        df = df.sort_values("filing_date")
        dates = df["filing_date"].to_numpy()
        vals = df[score_col].to_numpy()
        q = np.full(len(df), np.nan)
        for i in range(len(df)):
            lo = dates[i] - np.timedelta64(365, "D")
            window = vals[(dates > lo) & (dates <= dates[i])]
            if len(window) >= min_cohort:
                # empirical quantile of the score inside its trailing window
                pct = (window < vals[i]).sum() / len(window)
                q[i] = min(int(pct * n_quantiles) + 1, n_quantiles)
        df["quantile"] = q
    else:
        raise ValueError(f"unknown cohort rule {cohort!r}")
    return df.dropna(subset=["quantile"]).astype({"quantile": int})


# --------------------------------------------------------------------------- #
# Calendar
# --------------------------------------------------------------------------- #
def first_trading_day_after(
    filing_date: pd.Timestamp, trading_days: pd.DatetimeIndex, lag_months: int = 0
) -> pd.Timestamp | None:
    """First trading day of the month that starts after ``filing_date``, shifted
    by ``lag_months`` additional months. ``None`` if beyond the sample."""
    month_start = (pd.Timestamp(filing_date) + pd.offsets.MonthBegin(1 + lag_months)).normalize()
    pos = trading_days.searchsorted(month_start)
    return trading_days[pos] if pos < len(trading_days) else None


@dataclass
class Position:
    ticker: str
    quantile: int
    entry: pd.Timestamp  # position established at this close
    exit: pd.Timestamp | None  # position closed at this close (None = still open)


def build_positions(
    ranked: pd.DataFrame,
    trading_days: pd.DatetimeIndex,
    lag_months: int = 0,
    holding_months: int = HOLDING_MONTHS,
) -> pd.DataFrame:
    """One row per (filing, position): ticker, quantile, entry and exit dates.

    Entry is the first trading day of the month after the filing (plus lag);
    exit is the first trading day ``holding_months`` months later. The position
    earns returns strictly after ``entry`` and up to and including ``exit``.
    """
    rows = []
    for r in ranked.itertuples():
        entry = first_trading_day_after(r.filing_date, trading_days, lag_months)
        if entry is None:
            continue
        exit_ = first_trading_day_after(r.filing_date, trading_days, lag_months + holding_months)
        rows.append(
            {
                "ticker": r.ticker,
                "fiscal_year": r.fiscal_year,
                "filing_date": r.filing_date,
                "quantile": int(r.quantile),
                "entry": entry,
                "exit": exit_,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Returns
# --------------------------------------------------------------------------- #
def holdings_matrix(
    positions: pd.DataFrame, trading_days: pd.DatetimeIndex, tickers: list[str]
) -> dict[int, pd.DataFrame]:
    """For each quantile, a boolean ``dates x tickers`` frame that is True on
    the days a stock is held (i.e. earns a return) in that group."""
    out = {}
    for q, grp in positions.groupby("quantile"):
        held = pd.DataFrame(False, index=trading_days, columns=tickers)
        for p in grp.itertuples():
            if p.ticker not in held.columns:
                continue
            start = trading_days.searchsorted(p.entry, side="right")  # strictly after entry
            stop = len(trading_days) if p.exit is None else trading_days.searchsorted(p.exit, side="right")
            if stop > start:
                held.iloc[start:stop, held.columns.get_loc(p.ticker)] = True
        out[int(q)] = held
    return out


def equal_weight_returns(held: pd.DataFrame, daily_returns: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Daily equal-weighted return of the held stocks, the number of stocks
    with a valid return, and the one-way turnover (sum of absolute weight
    changes, ignoring intra-day drift)."""
    r = daily_returns.reindex(index=held.index, columns=held.columns)
    valid = held & r.notna()
    n = valid.sum(axis=1)
    w = valid.astype(float).div(n.replace(0, np.nan), axis=0).fillna(0.0)
    ret = (w * r.fillna(0.0)).sum(axis=1)
    ret[n == 0] = np.nan
    turnover = w.diff().abs().sum(axis=1)
    turnover.iloc[0] = w.iloc[0].abs().sum()
    return ret, n, turnover


def portfolio_returns(
    positions: pd.DataFrame,
    daily_returns: pd.DataFrame,
    n_quantiles: int,
    cost_bps: float = 0.0,
    min_names: int = MIN_NAMES,
) -> pd.DataFrame:
    """Daily returns of every quantile portfolio, the long-short portfolio
    (top minus bottom quantile) and its version net of transaction costs.

    ``cost_bps`` is charged per unit of one-way turnover on each leg. The
    long-short return is NaN on days when either leg holds fewer than
    ``min_names`` stocks.
    Columns: ``Q1..Qn, LS, LS_net, turnover, n_long, n_short``.
    """
    days = daily_returns.index
    tickers = list(daily_returns.columns)
    held = holdings_matrix(positions, days, tickers)
    out = pd.DataFrame(index=days)
    counts, turns = {}, {}
    for q in range(1, n_quantiles + 1):
        if q in held:
            ret, n, to = equal_weight_returns(held[q], daily_returns)
        else:
            ret = pd.Series(np.nan, index=days)
            n = pd.Series(0, index=days)
            to = pd.Series(0.0, index=days)
        out[f"Q{q}"] = ret
        counts[q], turns[q] = n, to
    out["LS"] = out[f"Q{n_quantiles}"] - out["Q1"]
    out["turnover"] = turns[n_quantiles] + turns[1]
    out["LS_net"] = out["LS"] - out["turnover"] * cost_bps / 1e4
    out["n_long"] = counts[n_quantiles]
    out["n_short"] = counts[1]
    # the long-short portfolio only exists when both legs are populated
    live = (out["n_long"] >= min_names) & (out["n_short"] >= min_names)
    out.loc[~live, ["LS", "LS_net"]] = np.nan
    return out


def to_monthly(daily: pd.DataFrame, return_cols: list[str] | None = None) -> pd.DataFrame:
    """Compound daily returns into calendar-month returns; sum turnover and
    take month-end values for counts. Months with no live long-short return
    are dropped."""
    return_cols = return_cols or [c for c in daily.columns if c.startswith(("Q", "LS"))]
    grouper = daily.index.to_period("M")
    monthly = pd.DataFrame(index=pd.PeriodIndex(sorted(set(grouper)), freq="M"))
    for c in return_cols:
        monthly[c] = (1.0 + daily[c]).groupby(grouper).prod(min_count=1) - 1.0
    if "turnover" in daily:
        monthly["turnover"] = daily["turnover"].groupby(grouper).sum()
    for c in ("n_long", "n_short"):
        if c in daily:
            monthly[c] = daily[c].groupby(grouper).last()
    monthly.index = monthly.index.to_timestamp(how="end").normalize()
    monthly.index.name = "Date"
    return monthly.dropna(subset=["LS"]) if "LS" in monthly else monthly


def run_variant(
    sim: pd.DataFrame,
    daily_returns: pd.DataFrame,
    score_col: str,
    n_quantiles: int = 5,
    lag_months: int = 0,
    cohort: str = "fiscal_year",
    cost_bps: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full pipeline for one strategy variant. Returns ``(daily, monthly)``."""
    ranked = rank_filings(sim, score_col, n_quantiles=n_quantiles, cohort=cohort)
    positions = build_positions(ranked, daily_returns.index, lag_months=lag_months)
    daily = portfolio_returns(positions, daily_returns, n_quantiles, cost_bps=cost_bps)
    return daily, to_monthly(daily)
