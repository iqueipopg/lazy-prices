"""Alpha regression against a case with known alpha, and basic statistics."""

import numpy as np
import pandas as pd
import pytest

from lazyprices import evaluation as ev


@pytest.fixture
def factors():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2000-01-31", periods=600, freq="ME")
    f = pd.DataFrame(rng.normal(0.005, 0.04, size=(600, 6)), index=idx,
                     columns=["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"])
    f["RF"] = 0.002
    return f


def test_alpha_regression_recovers_known_alpha(factors):
    rng = np.random.default_rng(1)
    alpha_m = 0.01  # 1% per month = 12% per year
    y = alpha_m + 1.2 * factors["Mkt-RF"] + rng.normal(0, 0.005, len(factors))
    for model in ("CAPM", "FF3", "FF5+MOM"):
        res = ev.alpha_regression(y, factors, model)
        assert res["alpha_ann_pct"] == pytest.approx(12.0, abs=0.6)
        assert res["t_stat"] > 20
    # loading on HML: recovered by FF3, absorbed into alpha by the CAPM
    y2 = y + 0.5 * factors["HML"]
    ff3 = ev.alpha_regression(y2, factors, "FF3")
    assert ff3["alpha_ann_pct"] == pytest.approx(12.0, abs=0.6)
    assert ff3["beta_Mkt-RF"] == pytest.approx(1.2, abs=0.02)
    assert ff3["beta_HML"] == pytest.approx(0.5, abs=0.02)
    capm = ev.alpha_regression(y2, factors, "CAPM")
    assert capm["alpha_ann_pct"] == pytest.approx(12.0 + 0.5 * factors["HML"].mean() * 1200, abs=0.6)


def test_alpha_regression_zero_alpha_is_insignificant(factors):
    rng = np.random.default_rng(2)
    y = 1.0 * factors["Mkt-RF"] + rng.normal(0, 0.02, len(factors))
    res = ev.alpha_regression(y, factors, "CAPM")
    assert abs(res["alpha_ann_pct"]) < 1.5
    assert abs(res["t_stat"]) < 2.5


def test_newey_west_lags():
    assert ev.newey_west_lags(100) == 4
    assert ev.newey_west_lags(200) == 4
    assert ev.newey_west_lags(1000) == 6


def test_sharpe_and_drawdown():
    r = pd.Series([0.01] * 24)
    assert np.isnan(ev.sharpe_annual(r))  # zero volatility
    r = pd.Series([0.1, -0.5, 0.2])
    assert ev.max_drawdown(r) == pytest.approx(-0.5)
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0.01, 0.02, 10_000))
    assert ev.sharpe_annual(r) == pytest.approx(0.5 * np.sqrt(12), rel=0.05)


def test_monthly_factors_compound():
    days = pd.bdate_range("2020-01-01", "2020-02-29")
    f = pd.DataFrame({"Mkt-RF": 0.001, "RF": 0.0}, index=days)
    m = ev.monthly_factors(f)
    n_jan = (days.month == 1).sum()
    assert m.iloc[0]["Mkt-RF"] == pytest.approx(1.001**n_jan - 1)


def test_overfitting_analysis_on_noise():
    """On pure-noise trials, the best Sharpe is not certified by the DSR."""
    rng = np.random.default_rng(4)
    trials = pd.DataFrame(rng.normal(0, 0.03, size=(240, 40)),
                          columns=[f"t{i}" for i in range(40)])
    rep = ev.overfitting_analysis(trials, n_splits=8)
    assert rep.n_trials == 40
    assert 0.0 <= rep.dsr_raw < 0.95
    assert 0.2 <= rep.pbo <= 0.8
    assert rep.cscv_combinations == 70
