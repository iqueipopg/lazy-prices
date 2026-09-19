"""Generate the LaTeX tables and number macros of the research note from the
result files, so that every figure quoted in the text comes from
``results/`` and cannot drift from the code.

    python -m lazyprices.report      # writes paper/generated/*.tex
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import config

OUT = config.ROOT / "paper" / "generated"


def f(x: float, d: int = 2) -> str:
    return "--" if pd.isna(x) else f"{x:.{d}f}"


def tex_escape(s: str) -> str:
    return s.replace("_", r"\_").replace("|", r"$\mid$").replace("%", r"\%").replace("&", r"\&")


def macro(name: str, value: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}\n"


def table(header: list[str], rows: list[list[str]], caption: str, label: str, align: str | None = None) -> str:
    align = align or "l" + "r" * (len(header) - 1)
    lines = [
        r"\begin{table}[htbp]\centering\small",
        rf"\caption{{{caption}}}\label{{{label}}}",
        rf"\begin{{tabular}}{{{align}}}\toprule",
        " & ".join(header) + r" \\ \midrule",
    ]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule\end{tabular}\end{table}", ""]
    return "\n".join(lines)


def main(out: Path = OUT) -> None:
    out.mkdir(parents=True, exist_ok=True)
    R = config.RESULTS
    q = pd.read_csv(R / "quintile_table.csv", index_col=0)
    alphas = pd.read_csv(R / "alphas.csv").set_index(["portfolio", "model"])
    rob = pd.read_csv(R / "robustness.csv", index_col=0)
    rc = pd.read_csv(R / "rank_correlations.csv", index_col=0)
    trials = pd.read_csv(R / "trials.csv")
    ev = pd.read_csv(R / "event_time.csv", index_col=0)
    sim = pd.read_csv(R / "similarity.csv")
    ov = json.load(open(R / "overfitting.json"))
    su = json.load(open(R / "summary.json"))
    pl = json.load(open(R / "placebo_summary.json"))
    m = pd.read_csv(R / "portfolio_monthly.csv", index_col=0, parse_dates=True)

    # ------------------------------------------------------------------ numbers
    n = ""
    n += macro("nCompanies", str(su["n_companies"]))
    n += macro("nFilings", f"{su['n_filings']:,}")
    n += macro("nPairs", f"{su['n_pairs']:,}")
    n += macro("nPairsIA", f"{su['n_pairs_with_1a']:,}")
    n += macro("nPairsVII", f"{su['n_pairs_with_7']:,}")
    n += macro("nMonths", str(su["n_months"]))
    n += macro("sampleStart", pd.Timestamp(su["sample_start"]).strftime("%B %Y"))
    n += macro("sampleEnd", pd.Timestamp(su["sample_end"]).strftime("%B %Y"))
    n += macro("spySharpe", f(su["spy_sharpe"]))
    n += macro("legSize", f(m["n_long"].mean(), 0))
    n += macro("legMin", str(int(min(m["n_long"].min(), m["n_short"].min()))))
    n += macro("turnover", f(m["turnover"].mean() * 12, 1))
    for port, tag in (("LS", "LS"), ("LS_net", "LSnet")):
        r = q.loc[port]
        n += macro(f"{tag}mean", f(r["mean_ann_pct"]))
        n += macro(f"{tag}vol", f(r["vol_ann_pct"]))
        n += macro(f"{tag}sharpe", f(r["sharpe"]))
        n += macro(f"{tag}dd", f(r["max_drawdown_pct"], 1))
        n += macro(f"{tag}sharpeLo", f(r["sharpe_ci_lo"]))
        n += macro(f"{tag}sharpeHi", f(r["sharpe_ci_hi"]))
        for model, mt in (("CAPM", "CAPM"), ("FF3", "FFthree"), ("FF5+MOM", "FFfive")):
            a = alphas.loc[(port, model)]
            n += macro(f"{tag}alpha{mt}", f(a["alpha_ann_pct"]))
            n += macro(f"{tag}t{mt}", f(a["t_stat"]))
    n += macro("nTrials", str(ov["n_trials"]))
    n += macro("nTrialsPositive", str(int((trials["sharpe"] > 0).sum())))
    sig = trials[trials["t_FF5+MOM"].abs() > 2]
    n += macro("nTrialsSig", str(len(sig)))
    n += macro("nTrialsSigNeg", str(int((sig["alpha_FF5+MOM"] < 0).sum())))
    n += macro("bestTrial", tex_escape(ov["best_trial"]))
    n += macro("bestSharpe", f(ov["best_sharpe_annual"]))
    n += macro("medianSharpe", f(ov["median_sharpe_annual"]))
    n += macro("emaxRaw", f(ov["expected_max_sharpe_annual_raw"]))
    n += macro("dsrRaw", f(ov["dsr_raw"]))
    n += macro("nEff", f(ov["n_effective"], 1))
    n += macro("emaxEff", f(ov["expected_max_sharpe_annual_eff"]))
    n += macro("dsrEff", f(ov["dsr_eff"]))
    n += macro("pbo", f(ov["pbo"]))
    n += macro("probOOSloss", f(ov["prob_oos_loss"]))
    n += macro("cscvCombos", f"{ov['cscv_combinations']:,}")
    n += macro("nObsCommon", str(ov["n_obs"]))
    best = trials.set_index("variant").loc[ov["best_trial"]]
    n += macro("bestOwnAlpha", f(best["alpha_FF5+MOM"]))
    n += macro("bestOwnT", f(best["t_FF5+MOM"]))
    n += macro("bestOwnDD", f(best["max_drawdown_pct"], 0))
    worst = trials.sort_values("sharpe").iloc[0]
    n += macro("worstTrial", tex_escape(worst["variant"]))
    n += macro("worstMean", f(worst["mean_ann_pct"], 1))
    n += macro("worstAlpha", f(worst["alpha_FF5+MOM"], 1))
    n += macro("worstT", f(worst["t_FF5+MOM"]))
    n += macro("worstDD", f(worst["max_drawdown_pct"], 0))
    n += macro("nPlacebo", str(pl["n_draws"]))
    n += macro("placeboPct", f(pl["share_of_draws_below_actual"] * 100, 0))
    n += macro("placeboSd", f(pl["placebo_sharpe_std"]))
    n += macro("placeboLo", f(pl["placebo_sharpe_p05_p95"][0]))
    n += macro("placeboHi", f(pl["placebo_sharpe_p05_p95"][1]))
    n += macro("placeboAbsT", f(pl["share_of_draws_with_abs_t_above_2"] * 100, 0))
    n += macro("placeboBestPct", f(pl["share_of_draws_below_best_variant"] * 100, 1))
    n += macro("carLS", f(pl["event_time_ls_car_12m_pct"], 1))
    n += macro("carLSt", f(pl["event_time_ls_t_12m"]))
    n += macro("carQlow", f(ev["Q_low_car"].iloc[-1] * 100, 1))
    n += macro("carQhigh", f(ev["Q_high_car"].iloc[-1] * 100, 1))
    n += macro("cosMedian", f(sim["cos_full"].median(), 3))
    n += macro("cosQone", f(sim["cos_full"].quantile(0.25), 3))
    n += macro("cosQthree", f(sim["cos_full"].quantile(0.75), 3))
    n += macro("jacMedian", f(sim["jac_full"].median(), 2))
    n += macro("jacQone", f(sim["jac_full"].quantile(0.25), 2))
    n += macro("jacQthree", f(sim["jac_full"].quantile(0.75), 2))
    byyear = sim.groupby("fiscal_year")["jac_full"].median()
    n += macro("jacFirst", f(byyear.iloc[0], 2))
    n += macro("jacLast", f(byyear.iloc[-1], 2))
    n += macro("shareMissingVII", f((1 - su["n_pairs_with_7"] / su["n_pairs"]) * 100, 0))
    n += macro("shareMissingIA", f((1 - su["n_pairs_with_1a"] / su["n_pairs"]) * 100, 0))
    (out / "numbers.tex").write_text(n)

    # ------------------------------------------------------------------ tables
    rows = []
    for label, key in (
        ("Mean return (\\% p.a.)", "mean_ann_pct"),
        ("Volatility (\\% p.a.)", "vol_ann_pct"),
        ("Sharpe ratio", "sharpe"),
        ("Sharpe ratio, 95\\% bootstrap interval", None),
        ("Maximum drawdown (\\%)", "max_drawdown_pct"),
    ):
        if key is None:
            rows.append(
                [label] + [f"[{f(q.loc[p, 'sharpe_ci_lo'])}, {f(q.loc[p, 'sharpe_ci_hi'])}]" for p in ("LS", "LS_net")]
            )
        else:
            rows.append([label] + [f(q.loc[p, key]) for p in ("LS", "LS_net")])
    for model in ("CAPM", "FF3", "FF5+MOM"):
        rows.append(
            [f"{model} alpha (\\% p.a.), $t$-statistic"]
            + [
                f"{f(alphas.loc[(p, model), 'alpha_ann_pct'])} ({f(alphas.loc[(p, model), 't_stat'])})"
                for p in ("LS", "LS_net")
            ]
        )
    (out / "main_ls.tex").write_text(
        table(
            ["", "Gross", "Net of 10 bp per unit turnover"],
            rows,
            f"Long-short portfolio (Q5 minus Q1), main specification: cosine TF-IDF on the full 10-K, quintiles "
            f"within fiscal-year cohorts, formation the month after filing, twelve-month hold. "
            f"{su['n_months']} months, {pd.Timestamp(su['sample_start']).strftime('%B %Y')} to "
            f"{pd.Timestamp(su['sample_end']).strftime('%B %Y')}. Newey-West $t$-statistics with 4 lags.",
            "tab:main",
        )
    )

    rows = []
    for i in range(1, 6):
        r = q.loc[f"Q{i}"]
        name = f"Q{i}" + (" (most change)" if i == 1 else " (least change)" if i == 5 else "")
        rows.append(
            [name, f(r["mean_ann_pct"], 1), f(r["vol_ann_pct"], 1), f(r["sharpe"]), f(r["max_drawdown_pct"], 1)]
            + [f"{f(r[f'alpha_{mo}'])} ({f(r[f't_{mo}'])})" for mo in ("CAPM", "FF3", "FF5+MOM")]
        )
    (out / "quintiles.tex").write_text(
        table(
            [
                "Quintile",
                "Mean",
                "Vol",
                "Sharpe",
                "Max DD",
                "CAPM $\\alpha$ ($t$)",
                "FF3 $\\alpha$ ($t$)",
                "FF5+MOM $\\alpha$ ($t$)",
            ],
            rows,
            "Quintile portfolios, monthly returns in excess of the risk-free rate. Mean, volatility, drawdown and alphas in percent per year.",
            "tab:quintiles",
        )
    )

    rows = [
        [
            tex_escape(i),
            str(int(r["n_months"])),
            f(r["mean_ann_pct"]),
            f(r["sharpe"]),
            f"{f(r['alpha_FF5+MOM'])} ({f(r['t_FF5+MOM'])})",
        ]
        for i, r in rob.iterrows()
    ]
    (out / "robustness.tex").write_text(
        table(
            ["Specification", "Months", "Mean (\\% p.a.)", "Sharpe", "FF5+MOM $\\alpha$ ($t$)"],
            rows,
            "Robustness of the main specification.",
            "tab:robustness",
        )
    )

    names = {
        "cos_full": "Cosine, full document",
        "jac_full": "Jaccard, full document",
        "cos_1a": "Cosine, Item 1A",
        "jac_1a": "Jaccard, Item 1A",
        "cos_7": "Cosine, Item 7",
        "jac_7": "Jaccard, Item 7",
    }
    rows = [
        [
            names[i],
            f(r["mean_rank_corr"], 3),
            f(r["t_stat"]),
            f"{int(round(r['share_positive'] * r['n_years']))} of {int(r['n_years'])}",
        ]
        for i, r in rc.iterrows()
    ]
    (out / "rankcorr.tex").write_text(
        table(
            ["Score", "Mean rank correlation", "$t$-statistic", "Cohorts positive"],
            rows,
            "Spearman correlation, within each fiscal-year cohort, between the similarity score and the twelve-month return "
            "after formation; time-series mean over cohorts and its $t$-statistic.",
            "tab:rankcorr",
        )
    )

    rows = [
        [
            str(int(i)),
            f(r["Q_low_car"] * 100),
            f(r["Q_high_car"] * 100),
            f(r["LS_car"] * 100),
            f(r["LS_se"] * 100),
            str(int(r["n_filings"])),
        ]
        for i, r in ev.iterrows()
        if int(i) in (1, 3, 6, 9, 12)
    ]
    (out / "eventtime.tex").write_text(
        table(
            ["Month", "Q1 CAR", "Q5 CAR", "Q5 $-$ Q1", "s.e.", "Filings"],
            rows,
            "Cumulative market-adjusted (minus SPY) return in percent by event month after formation, averaged first within "
            "and then across fiscal-year cohorts; the standard error treats each cohort as one observation.",
            "tab:eventtime",
        )
    )

    rows = [
        ["Variants tried", str(ov["n_trials"])],
        ["Variants with a positive Sharpe ratio", str(int((trials["sharpe"] > 0).sum()))],
        [
            "Variants with $|t| > 2$ on the FF5+MOM alpha",
            f"{len(sig)} (all negative)" if (sig["alpha_FF5+MOM"] < 0).all() else str(len(sig)),
        ],
        ["Best variant on the common sample", f"\\texttt{{{tex_escape(ov['best_trial'])}}}"],
        [
            "Annualised Sharpe, best / median variant",
            f"{f(ov['best_sharpe_annual'])} / {f(ov['median_sharpe_annual'])}",
        ],
        ["E[max Sharpe] of $N$ zero-skill trials, raw $N$", f(ov["expected_max_sharpe_annual_raw"])],
        ["\\textbf{Deflated Sharpe ratio, raw $N$}", f"\\textbf{{{f(ov['dsr_raw'])}}}"],
        ["Effective number of trials (participation ratio)", f(ov["n_effective"], 1)],
        ["E[max Sharpe], effective $N$", f(ov["expected_max_sharpe_annual_eff"])],
        ["Deflated Sharpe ratio, effective $N$", f(ov["dsr_eff"])],
        [f"PBO (CSCV, $S=16$, {ov['cscv_combinations']:,} partitions)", f(ov["pbo"])],
        ["P[out-of-sample loss of the in-sample winner]", f(ov["prob_oos_loss"])],
        [
            f"Placebo: share of {pl['n_draws']} shuffles with Sharpe below the actual strategy",
            f(pl["share_of_draws_below_actual"]),
        ],
        ["Placebo: standard deviation of the Sharpe ratio across shuffles", f(pl["placebo_sharpe_std"])],
        ["Placebo: share of shuffles with $|t| > 2$", f(pl["share_of_draws_with_abs_t_above_2"])],
    ]
    (out / "overfitting.tex").write_text(
        table(["", "Value"], rows, "Selection-bias corrections and placebo test.", "tab:overfitting", "lr")
    )

    t = trials.sort_values("sharpe", ascending=False)
    lines = [
        r"\begin{longtable}{lrrrrr}",
        r"\caption{All variants, sorted by annualised Sharpe ratio of the long-short portfolio on each variant's own sample. "
        r"Alphas in percent per year (FF5+MOM, Newey-West $t$).}\label{tab:trials}\\",
        r"\toprule Variant & Months & Mean (\% p.a.) & Sharpe & $\alpha$ & $t$ \\ \midrule \endfirsthead",
        r"\toprule Variant & Months & Mean (\% p.a.) & Sharpe & $\alpha$ & $t$ \\ \midrule \endhead",
        r"\bottomrule \endfoot",
    ]
    for _, r in t.iterrows():
        lines.append(
            f"\\texttt{{{tex_escape(r['variant'])}}} & {int(r['n_months'])} & {f(r['mean_ann_pct'])} & {f(r['sharpe'])} & "
            f"{f(r['alpha_FF5+MOM'])} & {f(r['t_FF5+MOM'])} \\\\"
        )
    lines.append(r"\end{longtable}")
    (out / "trials.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote {len(list(out.glob('*.tex')))} files to {out}")


if __name__ == "__main__":
    main()
