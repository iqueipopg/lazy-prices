"""Figures for the README. Static PNGs drawn with matplotlib.

Colour use follows one rule per job: the strategy and the benchmark are two
categorical series (blue, orange, fixed order); quantiles are ordinal, so
they use one hue stepped light to dark; trials are a magnitude distribution.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # categorical, fixed order
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e6e5e1"


def _style(ax, title: str, ylabel: str = ""):
    ax.set_title(title, loc="left", fontsize=11, color=TEXT)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)


def _blues(n: int) -> list:
    return [plt.cm.Blues(x) for x in np.linspace(0.35, 0.95, n)]


def cumulative_returns(monthly: pd.DataFrame, spy_monthly: pd.Series, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ls = (1 + monthly["LS"]).cumprod()
    ls_net = (1 + monthly["LS_net"]).cumprod()
    spy = (1 + spy_monthly.reindex(monthly.index)).cumprod()
    ax.plot(ls.index, ls, color=SERIES[0], lw=2, label="Long-short (gross)")
    ax.plot(ls_net.index, ls_net, color=SERIES[0], lw=1.2, ls="--", label="Long-short (10 bp per unit turnover)")
    ax.plot(spy.index, spy, color=SERIES[1], lw=2, label="SPY")
    ax.axhline(1.0, color=GRID, lw=1)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks([0.5, 0.7, 1, 1.5, 2, 3, 4, 6, 8, 10, 12])
    _style(ax, "Growth of 1 unit: long most-similar quintile, short least-similar quintile", "log scale")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def quantile_returns(table: pd.DataFrame, path: Path) -> None:
    q = table[table.index.str.startswith("Q")]
    n = len(q)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, col, title in zip(
        axes,
        ("mean_ann_pct", "alpha_FF5+MOM"),
        ("Annualised excess return by quintile (%)", "FF5+MOM alpha by quintile (% per year)"),
        strict=False,
    ):
        bars = ax.bar(q.index, q[col], color=_blues(n), width=0.6)
        for b, v, t in zip(bars, q[col], q.get("t_FF5+MOM", pd.Series(np.nan, index=q.index)), strict=False):
            label = f"{v:.1f}" if col == "mean_ann_pct" else f"{v:.1f}\n(t={t:.1f})"
            ax.annotate(
                label,
                (b.get_x() + b.get_width() / 2, v),
                ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=8,
                color=TEXT,
                xytext=(0, 3 if v >= 0 else -3),
                textcoords="offset points",
            )
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.margins(y=0.2)
        _style(ax, title)
        ax.set_xlabel(f"Q1 = most textual change, Q{n} = least", color=MUTED, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def similarity_by_year(sim: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=False)
    years = sorted(sim["fiscal_year"].unique())
    for ax, col, title in zip(
        axes,
        ("cos_full", "jac_full"),
        ("Cosine similarity (TF-IDF), full 10-K", "Jaccard similarity, full 10-K"),
        strict=False,
    ):
        data = [sim.loc[sim["fiscal_year"] == y, col].dropna().to_numpy() for y in years]
        bp = ax.boxplot(
            data,
            tick_labels=[str(y)[2:] for y in years],
            widths=0.55,
            patch_artist=True,
            showfliers=True,
            flierprops=dict(marker=".", markersize=3, color=MUTED, alpha=0.6),
            medianprops=dict(color=TEXT, lw=1.2),
        )
        for patch in bp["boxes"]:
            patch.set(facecolor="#cfe0f6", edgecolor=SERIES[0], linewidth=1)
        for k in ("whiskers", "caps"):
            for line in bp[k]:
                line.set(color=SERIES[0], linewidth=1)
        _style(ax, title, "similarity with previous year's 10-K")
        ax.set_xlabel("fiscal year (20xx)", color=MUTED, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def dsr_trials(trials: pd.DataFrame, report: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.2))
    col = "sharpe_common_sample" if "sharpe_common_sample" in trials else "sharpe"
    t = trials.sort_values(col).reset_index(drop=True)
    colors = [SERIES[1] if v == report["best_trial"] else "#9dbfe9" for v in t["variant"]]
    ax.bar(np.arange(len(t)), t[col], color=colors, width=0.8)
    ax.axhline(
        report["expected_max_sharpe_annual_raw"],
        color=SERIES[1],
        lw=1.2,
        ls="--",
        label=f"E[max Sharpe] of {report['n_trials']} noise trials (raw N): "
        f"{report['expected_max_sharpe_annual_raw']:.2f}",
    )
    ax.axhline(
        report["expected_max_sharpe_annual_eff"],
        color=SERIES[2],
        lw=1.2,
        ls=":",
        label=f"E[max Sharpe], effective N = {report['n_effective']:.1f}: "
        f"{report['expected_max_sharpe_annual_eff']:.2f}",
    )
    ax.axhline(0, color=MUTED, lw=0.8)
    _style(
        ax,
        f"Annualised Sharpe of the {report['n_trials']} variants on the common sample\n"
        f"best: {report['best_trial']}; deflated Sharpe ratio = {report['dsr_raw']:.2f}; PBO = {report['pbo']:.2f}",
        "annualised Sharpe",
    )
    ax.set_xlabel("variants, sorted (measure x section x groups x formation lag)", color=MUTED, fontsize=9)
    ax.set_xticks([])
    ax.margins(y=0.15)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_all(
    monthly: pd.DataFrame,
    spy_monthly: pd.Series,
    table: pd.DataFrame,
    sim: pd.DataFrame,
    trials: pd.DataFrame,
    report: dict,
    out_dir: Path = config.FIGURES,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cumulative_returns(monthly, spy_monthly, out_dir / "cumulative_long_short_vs_spy.png")
    quantile_returns(table, out_dir / "quintile_returns.png")
    similarity_by_year(sim, out_dir / "similarity_by_year.png")
    dsr_trials(trials, report, out_dir / "dsr_trials.png")


def event_time(car: pd.DataFrame, path: Path, n_quantiles: int = 5) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    m = car.index
    ax.plot(m, car["Q_high_car"] * 100, color=SERIES[0], lw=2, marker="o", ms=4, label=f"Q{n_quantiles}: least change")
    ax.plot(m, car["Q_low_car"] * 100, color=SERIES[1], lw=2, marker="o", ms=4, label="Q1: most change")
    ax.plot(m, car["LS_car"] * 100, color=TEXT, lw=1.5, ls="--", label=f"Q{n_quantiles} minus Q1")
    band = 1.96 * car["LS_se"] * 100
    ax.fill_between(m, (car["LS_car"] * 100 - band), (car["LS_car"] * 100 + band), color=TEXT, alpha=0.08, lw=0)
    ax.axhline(0, color=MUTED, lw=0.8)
    _style(ax, "Cumulative market-adjusted return after the 10-K filing (%, 95% band over cohorts)", "% cumulative")
    ax.set_xlabel("months after formation", color=MUTED, fontsize=9)
    ax.set_xticks(list(m))
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def placebo_hist(draws: pd.DataFrame, real_sharpe: float, best_sharpe: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(draws["sharpe"], bins=30, color="#9dbfe9", edgecolor="white")
    ax.axvline(real_sharpe, color=SERIES[0], lw=2, label=f"actual strategy: {real_sharpe:.2f}")
    ax.axvline(best_sharpe, color=SERIES[1], lw=2, ls="--", label=f"best of 54 variants: {best_sharpe:.2f}")
    _style(
        ax,
        f"Long-short Sharpe under {len(draws)} within-cohort shuffles of the similarity scores",
        "number of draws",
    )
    ax.set_xlabel("annualised Sharpe ratio", color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
