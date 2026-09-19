"""Event-time alignment, placebo shuffling and the bootstrap interval."""

import numpy as np
import pandas as pd
import pytest

from lazyprices import diagnostics as dg


@pytest.fixture
def prices():
    days = pd.bdate_range("2020-01-01", "2021-12-31")
    # stock doubles at the first trading day of each month; benchmark flat
    p = pd.DataFrame({"X": 1.0, "SPY": 1.0}, index=days)
    month_starts = p.groupby(p.index.to_period("M")).head(1).index
    level = 1.0
    for d in month_starts:
        p.loc[d:, "X"] = level
        level *= 1.1
    return p


def test_event_time_months_start_at_formation(prices):
    ranked = pd.DataFrame(
        {"ticker": ["X"], "fiscal_year": [2019], "filing_date": [pd.Timestamp("2020-02-14")], "quantile": [5]}
    )
    ev = dg.event_time_returns(ranked, prices, benchmark="SPY", horizon=3)
    assert ev["month"].tolist() == [1, 2, 3]
    # first event month: first trading day of March -> first trading day of April: +10%
    assert np.allclose(ev["ret"], 0.1)
    assert np.allclose(ev["bench"], 0.0)
    assert np.allclose(ev["abnormal"], 0.1)


def test_event_time_car_and_se():
    rows = []
    for fy in (2010, 2011):
        for q, a in ((1, 0.00), (5, 0.01)):
            for k in (1, 2, 3):
                rows.append({"ticker": f"T{q}", "fiscal_year": fy, "quantile": q, "month": k, "abnormal": a})
    car = dg.event_time_car(pd.DataFrame(rows), n_quantiles=5)
    assert np.allclose(car["LS_car"], [0.01, 0.02, 0.03])
    assert np.allclose(car["LS_se"], 0.0)
    assert (car["n_cohorts"] == 2).all()


def test_shuffle_preserves_cohort_scores():
    sim = pd.DataFrame({"fiscal_year": [1, 1, 1, 2, 2], "cos_full": [0.1, 0.2, 0.3, 0.8, 0.9], "ticker": list("abcde")})
    out = dg.shuffle_within_cohorts(sim, "cos_full", np.random.default_rng(1))
    for fy, g in sim.groupby("fiscal_year"):
        assert sorted(out.loc[out.fiscal_year == fy, "cos_full"]) == sorted(g["cos_full"])
    assert (out["ticker"] == sim["ticker"]).all()


def test_sharpe_bootstrap_ci_covers_point_estimate():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.01, 0.03, 240))
    point = r.mean() / r.std(ddof=1) * np.sqrt(12)
    lo, hi = dg.sharpe_bootstrap_ci(r, n_boot=500)
    assert lo < point < hi
    assert hi - lo < 2.0
    assert dg.percentile_of(0.0, pd.Series([-1.0, 0.5, 1.0])) == pytest.approx(1 / 3)
