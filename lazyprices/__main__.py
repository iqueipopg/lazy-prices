"""Run the whole pipeline: ``python -m lazyprices``.

Every stage is cached on disk; delete the corresponding file (or pass
``--refresh``) to recompute it.

    download  -> data/universe.csv, data/filings.csv, data/raw/
    text      -> data/processed/, results/similarity.csv
    market    -> data/prices.csv, data/factors_daily.csv
    portfolio -> results/portfolio_monthly.csv, quintile_table.csv, alphas.csv
    trials    -> results/trials.csv, trials_monthly.csv, overfitting.json
    figures   -> figures/*.png
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import time

import pandas as pd

from . import config, data, diagnostics, edgar, evaluation, figures, portfolio, text

log = logging.getLogger("lazyprices")

MAIN_SPEC = dict(score_col="cos_full", n_quantiles=5, lag_months=0, cohort="fiscal_year")
COST_BPS = 10.0
GRID = dict(
    measure=["cos", "jac"],
    section=["full", "1a", "7"],
    n_quantiles=[3, 5, 10],
    lag_months=[0, 1, 2],
)


def stage_download(refresh: bool) -> pd.DataFrame:
    config.DATA.mkdir(parents=True, exist_ok=True)
    uni_path = config.DATA / "universe.csv"
    if uni_path.exists() and not refresh:
        universe = pd.read_csv(uni_path, dtype={"cik": str})
    else:
        universe = edgar.load_universe()
        universe.to_csv(uni_path, index=False)
    idx_path = config.DATA / "filings.csv"
    if idx_path.exists() and not refresh:
        return edgar.load_filing_index(idx_path)
    return edgar.download_universe(universe, index_path=idx_path)


def stage_text(index: pd.DataFrame, refresh: bool) -> pd.DataFrame:
    sim_path = config.RESULTS / "similarity.csv"
    if sim_path.exists() and not refresh:
        return pd.read_csv(sim_path, dtype={"cik": str}, parse_dates=["report_date", "filing_date", "prev_filing_date"])
    processed = text.process_corpus(index)
    processed.to_csv(config.DATA / "filings_processed.csv", index=False)
    sim = text.compute_similarity(processed)
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    sim.to_csv(sim_path, index=False, float_format="%.6f")
    return sim


def stage_market(universe_tickers: list[str], refresh: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    prices = data.download_prices(universe_tickers, refresh=refresh)
    factors = data.download_factors(refresh=refresh)
    return prices, factors


def _returns_for_universe(prices: pd.DataFrame, sim: pd.DataFrame) -> pd.DataFrame:
    rets = data.daily_returns(prices)
    # similarity table uses SEC tickers; prices use yfinance symbols
    rets.columns = [c for c in rets.columns]
    sim["ticker"] = sim["ticker"].map(data.yf_symbol)
    return rets


def stage_portfolio(
    sim: pd.DataFrame, rets: pd.DataFrame, factors_m: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily, monthly = portfolio.run_variant(sim, rets, cost_bps=COST_BPS, **MAIN_SPEC)
    table, alphas = evaluation.evaluate_portfolio(monthly, factors_m, MAIN_SPEC["n_quantiles"])
    monthly.to_csv(config.RESULTS / "portfolio_monthly.csv", float_format="%.6f")
    table.to_csv(config.RESULTS / "quintile_table.csv", float_format="%.4f")
    alphas.to_csv(config.RESULTS / "alphas.csv", float_format="%.4f")
    return monthly, table, alphas


def stage_trials(
    sim: pd.DataFrame, rets: pd.DataFrame, factors_m: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, evaluation.OverfittingReport]:
    rows, series = [], {}
    for measure, section, nq, lag in itertools.product(*GRID.values()):
        col = f"{measure}_{section}"
        name = f"{col}|q{nq}|lag{lag}"
        _, monthly = portfolio.run_variant(sim, rets, col, n_quantiles=nq, lag_months=lag, cost_bps=COST_BPS)
        if monthly.empty:
            continue
        series[name] = monthly["LS"]
        stats = evaluation.summary_stats(monthly["LS"])
        reg = evaluation.alpha_regression(monthly["LS"], factors_m.reindex(monthly.index), "FF5+MOM")
        rows.append(
            {
                "variant": name,
                "measure": measure,
                "section": section,
                "n_quantiles": nq,
                "lag_months": lag,
                **stats,
                "alpha_FF5+MOM": reg["alpha_ann_pct"],
                "t_FF5+MOM": reg["t_stat"],
                "sharpe_net": evaluation.sharpe_annual(monthly["LS_net"]),
            }
        )
    trials = pd.DataFrame(rows)
    matrix = pd.DataFrame(series)
    report = evaluation.overfitting_analysis(matrix)
    trials["sharpe_common_sample"] = trials["variant"].map(report.trial_sharpe_annual)
    trials.to_csv(config.RESULTS / "trials.csv", index=False, float_format="%.4f")
    matrix.to_csv(config.RESULTS / "trials_monthly.csv", float_format="%.6f")
    report.to_json(config.RESULTS / "overfitting.json")
    return trials, matrix, report


def stage_robustness(sim: pd.DataFrame, rets: pd.DataFrame, factors_m: pd.DataFrame) -> pd.DataFrame:
    """Main specification with alternative ranking / sub-periods."""
    rows = []
    spec = dict(MAIN_SPEC)
    for label, kwargs, start, end in [
        ("main", spec, None, None),
        ("trailing-window ranking (point-in-time)", {**spec, "cohort": "trailing"}, None, None),
        ("main, 2009-2016", spec, None, "2016-12-31"),
        ("main, 2017-2026", spec, "2017-01-01", None),
    ]:
        _, monthly = portfolio.run_variant(sim, rets, cost_bps=COST_BPS, **kwargs)
        monthly = monthly.loc[start:end]
        stats = evaluation.summary_stats(monthly["LS"])
        reg = evaluation.alpha_regression(monthly["LS"], factors_m.reindex(monthly.index), "FF5+MOM")
        rows.append(
            {"specification": label, **stats, "alpha_FF5+MOM": reg["alpha_ann_pct"], "t_FF5+MOM": reg["t_stat"]}
        )
    rob = pd.DataFrame(rows).set_index("specification")
    rob.to_csv(config.RESULTS / "robustness.csv", float_format="%.4f")
    return rob


def stage_diagnostics(
    sim: pd.DataFrame,
    rets: pd.DataFrame,
    prices: pd.DataFrame,
    factors_m: pd.DataFrame,
    monthly: pd.DataFrame,
    report: evaluation.OverfittingReport,
    n_placebo: int,
) -> dict:
    """Event-time abnormal returns, placebo distribution and Sharpe interval."""
    ranked = portfolio.rank_filings(sim, MAIN_SPEC["score_col"], MAIN_SPEC["n_quantiles"], MAIN_SPEC["cohort"])
    events = diagnostics.event_time_returns(ranked, prices, lag_months=MAIN_SPEC["lag_months"])
    car = diagnostics.event_time_car(events, MAIN_SPEC["n_quantiles"])
    car.to_csv(config.RESULTS / "event_time.csv", float_format="%.6f")
    draws = diagnostics.placebo(sim, rets, factors_m, MAIN_SPEC, n_draws=n_placebo)
    draws.to_csv(config.RESULTS / "placebo.csv", index=False, float_format="%.4f")
    real_sharpe = evaluation.sharpe_annual(monthly["LS"])
    lo, hi = diagnostics.sharpe_bootstrap_ci(monthly["LS"])
    summary = {
        "n_draws": int(len(draws)),
        "actual_sharpe": real_sharpe,
        "actual_sharpe_ci95": [lo, hi],
        "placebo_sharpe_mean": float(draws["sharpe"].mean()),
        "placebo_sharpe_std": float(draws["sharpe"].std(ddof=1)),
        "placebo_sharpe_p05_p95": [float(draws["sharpe"].quantile(0.05)), float(draws["sharpe"].quantile(0.95))],
        "share_of_draws_below_actual": diagnostics.percentile_of(real_sharpe, draws["sharpe"]),
        "share_of_draws_below_best_variant": diagnostics.percentile_of(report.best_sharpe_annual, draws["sharpe"]),
        "share_of_draws_with_abs_t_above_2": float((draws["t_FF5+MOM"].abs() > 2).mean()),
        "event_time_ls_car_12m_pct": float(car["LS_car"].iloc[-1] * 100),
        "event_time_ls_t_12m": float(car["LS_car"].iloc[-1] / car["LS_se"].iloc[-1]),
    }
    with open(config.RESULTS / "placebo_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    return {"car": car, "draws": draws, "summary": summary}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m lazyprices")
    ap.add_argument("--refresh", action="store_true", help="ignore cached results (raw downloads are always cached)")
    ap.add_argument("--skip-figures", action="store_true")
    ap.add_argument("--placebo-draws", type=int, default=200)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    t0 = time.time()

    index = stage_download(args.refresh)
    log.info("filings: %d rows, %d companies", len(index), index["ticker"].nunique())
    sim = stage_text(index, args.refresh)
    log.info("similarity pairs: %d", len(sim))
    universe = pd.read_csv(config.DATA / "universe.csv", dtype={"cik": str})
    prices, factors = stage_market(universe["ticker"].tolist(), args.refresh)
    factors_m = evaluation.monthly_factors(factors)
    rets = _returns_for_universe(prices, sim)
    spy_m = portfolio.to_monthly(pd.DataFrame({"SPY": rets[config.BENCHMARK]}), ["SPY"])["SPY"]

    monthly, table, alphas = stage_portfolio(sim, rets, factors_m)
    log.info("main spec: %d months\n%s", len(monthly), table.round(2).to_string())
    trials, matrix, report = stage_trials(sim, rets, factors_m)
    log.info(
        "trials: %d variants; best %s; DSR raw %.3f eff %.3f; PBO %.3f",
        report.n_trials,
        report.best_trial,
        report.dsr_raw,
        report.dsr_eff,
        report.pbo,
    )
    rob = stage_robustness(sim, rets, factors_m)
    log.info("robustness:\n%s", rob.round(2).to_string())
    rc = evaluation.rank_correlations(sim, prices, [f"{m}_{s}" for m in GRID["measure"] for s in GRID["section"]])
    rc.to_csv(config.RESULTS / "rank_correlations.csv", float_format="%.4f")
    log.info("rank correlations:\n%s", rc.round(3).to_string())

    diag = stage_diagnostics(sim, rets, prices, factors_m, monthly, report, args.placebo_draws)
    log.info(
        "diagnostics: %s",
        json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in diag["summary"].items()}),
    )

    summary = {
        "n_companies": int(index["ticker"].nunique()),
        "n_filings": int(len(index)),
        "n_pairs": int(len(sim)),
        "n_pairs_with_1a": int(sim["cos_1a"].notna().sum()),
        "n_pairs_with_7": int(sim["cos_7"].notna().sum()),
        "sample_start": str(monthly.index.min().date()),
        "sample_end": str(monthly.index.max().date()),
        "n_months": int(len(monthly)),
        "main_spec": MAIN_SPEC,
        "cost_bps": COST_BPS,
        "spy_sharpe": evaluation.sharpe_annual(spy_m.reindex(monthly.index) - factors_m["RF"].reindex(monthly.index)),
        "runtime_seconds": round(time.time() - t0, 1),
    }
    with open(config.RESULTS / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    if not args.skip_figures:
        figures.make_all(monthly, spy_m, table, sim, trials, json.load(open(config.RESULTS / "overfitting.json")))
        figures.event_time(diag["car"], config.FIGURES / "event_time_car.png", MAIN_SPEC["n_quantiles"])
        figures.placebo_hist(
            diag["draws"],
            diag["summary"]["actual_sharpe"],
            report.best_sharpe_annual,
            config.FIGURES / "placebo_sharpe.png",
        )
    log.info("done in %.0f s", time.time() - t0)


if __name__ == "__main__":
    main()
