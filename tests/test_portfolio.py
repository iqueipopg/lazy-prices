"""Quantile formation and time alignment on synthetic data."""

import numpy as np
import pandas as pd
import pytest

from lazyprices import portfolio as pf


def test_assign_quantiles_balanced_and_ordered():
    scores = pd.Series(np.arange(10, dtype=float)[::-1], index=list("abcdefghij"))
    q = pf.assign_quantiles(scores, 5)
    assert q.value_counts().tolist() == [2, 2, 2, 2, 2]
    assert q["a"] == 5 and q["b"] == 5  # highest scores -> top group
    assert q["i"] == 1 and q["j"] == 1  # lowest scores -> bottom group


def test_assign_quantiles_handles_nan_and_small_samples():
    scores = pd.Series([0.1, np.nan, 0.9, 0.5])
    q = pf.assign_quantiles(scores, 3)
    assert np.isnan(q[1]) and q[2] == 3 and q[0] == 1
    assert pf.assign_quantiles(pd.Series([0.1, 0.2]), 5).isna().all()


def test_rank_filings_by_fiscal_year_cohort():
    sim = pd.DataFrame(
        {
            "ticker": list("ABCDEFGHIJ") * 2,
            "fiscal_year": [2010] * 10 + [2011] * 10,
            "filing_date": ["2011-02-15"] * 10 + ["2012-02-15"] * 10,
            "cos_full": list(np.linspace(0.5, 0.95, 10)) + list(np.linspace(0.95, 0.5, 10)),
        }
    )
    ranked = pf.rank_filings(sim, "cos_full", n_quantiles=5)
    r = ranked.set_index(["ticker", "fiscal_year"])["quantile"]
    assert r[("J", 2010)] == 5 and r[("A", 2010)] == 1
    assert r[("A", 2011)] == 5 and r[("J", 2011)] == 1
    assert ranked.groupby(["fiscal_year", "quantile"]).size().eq(2).all()


def test_rank_filings_trailing_is_point_in_time():
    dates = pd.date_range("2011-01-31", periods=12, freq="ME")
    sim = pd.DataFrame({"ticker": [f"T{i}" for i in range(12)], "fiscal_year": 2010,
                        "filing_date": dates, "cos_full": np.linspace(0.1, 0.9, 12)})
    ranked = pf.rank_filings(sim, "cos_full", n_quantiles=2, cohort="trailing", min_cohort=4)
    # the first three filings have fewer than 4 predecessors: dropped
    assert set(ranked["ticker"]) == {f"T{i}" for i in range(3, 12)}
    # scores increase through time, so every filing is top of its own window
    assert (ranked["quantile"] == 2).all()


@pytest.fixture
def market():
    days = pd.bdate_range("2020-01-01", "2021-12-31")
    rets = pd.DataFrame(0.0, index=days, columns=["X", "Y"])
    return days, rets


def test_first_trading_day_after(market):
    days, _ = market
    assert pf.first_trading_day_after(pd.Timestamp("2020-02-14"), days) == pd.Timestamp("2020-03-02")
    assert pf.first_trading_day_after(pd.Timestamp("2020-02-29"), days) == pd.Timestamp("2020-03-02")
    assert pf.first_trading_day_after(pd.Timestamp("2020-02-14"), days, lag_months=1) == pd.Timestamp("2020-04-01")
    assert pf.first_trading_day_after(pd.Timestamp("2021-12-15"), days) is None


def test_no_look_ahead_positions_start_after_filing(market):
    """A stock that files on 2020-02-14 must earn nothing before the first
    trading day of March 2020, and the hold must end twelve months later."""
    days, rets = market
    rets.loc[:"2020-03-02", "X"] = 1.0  # huge returns up to and including entry day
    rets.loc["2020-03-03":"2021-03-01", "X"] = 0.01
    rets.loc["2021-03-02":, "X"] = 1.0  # huge returns after the exit day
    rets["Y"] = 0.0
    ranked = pd.DataFrame({"ticker": ["X", "Y"], "fiscal_year": [2019, 2019],
                           "filing_date": pd.to_datetime(["2020-02-14", "2020-02-14"]), "quantile": [2, 1]})
    pos = pf.build_positions(ranked, days)
    assert pos.set_index("ticker").loc["X", "entry"] == pd.Timestamp("2020-03-02")
    assert pos.set_index("ticker").loc["X", "exit"] == pd.Timestamp("2021-03-01")
    daily = pf.portfolio_returns(pos, rets, n_quantiles=2, min_names=1)
    live = daily["Q2"].dropna()
    assert live.index.min() == pd.Timestamp("2020-03-03")
    assert live.index.max() == pd.Timestamp("2021-03-01")
    assert np.allclose(live.values, 0.01)  # none of the 100% days leaked in
    assert np.allclose(daily["LS"].dropna().values, 0.01)


def test_equal_weight_and_turnover(market):
    days, rets = market
    rets["X"] = 0.02
    rets["Y"] = -0.02
    held = pd.DataFrame(False, index=days, columns=["X", "Y"])
    held.iloc[1:11] = True
    ret, n, to = pf.equal_weight_returns(held, rets)
    assert np.allclose(ret.iloc[1:11], 0.0) and (n.iloc[1:11] == 2).all()
    assert to.iloc[1] == pytest.approx(1.0)  # 0 -> (0.5, 0.5)
    assert to.iloc[2] == pytest.approx(0.0)
    assert to.iloc[11] == pytest.approx(1.0)  # exit


def test_to_monthly_compounds(market):
    days, _ = market
    daily = pd.DataFrame({"Q1": 0.0, "Q2": 0.01, "LS": 0.01, "LS_net": 0.01,
                          "turnover": 0.1, "n_long": 3, "n_short": 3}, index=days)
    m = pf.to_monthly(daily)
    n_jan = len(pd.bdate_range("2020-01-01", "2020-01-31"))
    assert m.iloc[0]["LS"] == pytest.approx(1.01**n_jan - 1)
    assert m.iloc[0]["turnover"] == pytest.approx(0.1 * n_jan)
    assert m.index[0] == pd.Timestamp("2020-01-31")
