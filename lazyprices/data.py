"""Market data: adjusted daily prices from yfinance and daily Fama-French
factors from the Kenneth French Data Library. Everything is cached under
``data/``."""

from __future__ import annotations

import io
import logging
import re
import zipfile
from pathlib import Path

import pandas as pd
import requests

from . import config

log = logging.getLogger(__name__)

FRENCH_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FF5_ZIP = "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
MOM_ZIP = "F-F_Momentum_Factor_daily_CSV.zip"

PRICES_CSV = config.DATA / "prices.csv"
FACTORS_CSV = config.DATA / "factors_daily.csv"


def yf_symbol(ticker: str) -> str:
    """SEC tickers use '-' for share classes (BRK-B); so does yfinance.
    Dots are converted for safety."""
    return ticker.replace(".", "-")


def download_prices(tickers: list[str], start: str = "2007-01-01", path: Path = PRICES_CSV, refresh: bool = False) -> pd.DataFrame:
    """Adjusted daily closes for ``tickers`` plus the benchmark, wide format
    (index ``Date``, one column per ticker)."""
    if path.exists() and not refresh:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    import yfinance as yf

    symbols = sorted({yf_symbol(t) for t in tickers} | {config.BENCHMARK})
    log.info("downloading %d symbols from yfinance", len(symbols))
    raw = yf.download(symbols, start=start, auto_adjust=True, progress=False, threads=True, group_by="column")
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    close = close.dropna(how="all")
    close.index.name = "Date"
    missing = [s for s in symbols if s not in close.columns or close[s].notna().sum() == 0]
    if missing:
        log.warning("no price data for %s", missing)
    path.parent.mkdir(parents=True, exist_ok=True)
    close.to_csv(path, float_format="%.6f")
    return close


def _parse_french_csv(text: str, columns: list[str]) -> pd.DataFrame:
    """Parse a French daily CSV: skip the header prose, stop at the first blank
    line after the data (annual tables follow in some files)."""
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if re.match(r"^\s*\d{8},", ln))
    end = start
    while end < len(lines) and re.match(r"^\s*\d{8},", lines[end]):
        end += 1
    body = "\n".join(lines[start:end])
    df = pd.read_csv(io.StringIO(body), header=None, names=["Date"] + columns)
    df["Date"] = pd.to_datetime(df["Date"].astype(str), format="%Y%m%d")
    df = df.set_index("Date").astype(float) / 100.0  # percent -> decimal
    df = df.mask(df <= -0.99)  # -99.99 / -999 are missing markers
    return df


def _fetch_zip_csv(name: str, cache_dir: Path) -> str:
    cache_dir.mkdir(parents=True, exist_ok=True)
    zpath = cache_dir / name
    if not zpath.exists():
        r = requests.get(FRENCH_BASE + name, timeout=120)
        r.raise_for_status()
        zpath.write_bytes(r.content)
    with zipfile.ZipFile(zpath) as zf:
        inner = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
        return zf.read(inner).decode("utf-8", errors="replace")


def download_factors(path: Path = FACTORS_CSV, refresh: bool = False) -> pd.DataFrame:
    """Daily FF5 factors plus momentum, decimals, indexed by date."""
    if path.exists() and not refresh:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    cache = config.DATA / "french"
    ff5 = _parse_french_csv(_fetch_zip_csv(FF5_ZIP, cache), ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"])
    mom = _parse_french_csv(_fetch_zip_csv(MOM_ZIP, cache), ["Mom"])
    fac = ff5.join(mom, how="inner")
    fac.to_csv(path, float_format="%.6f")
    return fac


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.sort_index().pct_change(fill_method=None)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    uni = pd.read_csv(config.DATA / "universe.csv", dtype={"cik": str})
    p = download_prices(uni["ticker"].tolist())
    log.info("prices: %s .. %s, %d columns", p.index.min().date(), p.index.max().date(), p.shape[1])
    f = download_factors()
    log.info("factors: %s .. %s", f.index.min().date(), f.index.max().date())
