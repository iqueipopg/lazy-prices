"""SEC EDGAR access: universe, filing index and 10-K download with caching.

Everything downloaded is cached under ``data/raw`` so that a second run makes
no network requests. Requests carry the identifying User-Agent that the SEC
requires and are throttled to at most ``SEC_MAX_RPS`` per second across
threads, with exponential back-off on transient errors.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

from . import config

log = logging.getLogger(__name__)

SUBMISSIONS_URL = "https://data.sec.gov/submissions/{name}"
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{doc}"
ANNUAL_FORMS = {"10-K", "10-K405"}

# Companies that moved to a new CIK after a holding-company reorganisation.
# Filings of the predecessor registrant are attributed to the current ticker.
PREDECESSOR_CIKS: dict[str, list[str]] = {
    "XOM": ["0000034088"],    # Exxon Mobil Corp -> ExxonMobil Holdings Corp (2026)
    "GOOGL": ["0001288776"],  # Google Inc. -> Alphabet Inc. (2015)
    "DIS": ["0001001039"],    # TWDC Enterprises 18 Corp. -> The Walt Disney Company (2019)
    "BLK": ["0001364742"],    # BlackRock Finance, Inc. -> BlackRock, Inc. (2024)
    "MDT": ["0000064670"],    # Medtronic, Inc. -> Medtronic plc (2015)
}


# --------------------------------------------------------------------------- #
# Universe
# --------------------------------------------------------------------------- #
def load_universe(cache_dir: Path | None = None) -> pd.DataFrame:
    """Read CIK, ticker and company name from the LUCA cache file names and
    submissions JSONs. Only reads; never writes into the LUCA tree.

    Returns a DataFrame with columns ``cik`` (10-digit string), ``ticker``,
    ``name``, sorted by ticker.
    """
    cache_dir = Path(cache_dir or config.LUCA_CACHE)
    rows = []
    ciks = {p.name.split("_")[0] for p in cache_dir.glob("*_submissions.json")}
    ciks |= {p.name.split("_")[0] for p in cache_dir.glob("*_facts.json")}
    for cik in sorted(ciks):
        sub = cache_dir / f"{cik}_submissions.json"
        ticker, name = None, None
        if sub.exists():
            with open(sub) as fh:
                d = json.load(fh)
            tickers = d.get("tickers") or []
            ticker = tickers[0] if tickers else None
            name = d.get("name")
        rows.append({"cik": cik.zfill(10), "ticker": ticker, "name": name})
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["ticker"]).sort_values("ticker").reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# HTTP client with rate limit and retries
# --------------------------------------------------------------------------- #
class RateLimiter:
    """Thread-safe minimum spacing between requests."""

    def __init__(self, max_per_second: float):
        self.min_interval = 1.0 / max_per_second
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            if now < self._next:
                time.sleep(self._next - now)
                now = time.monotonic()
            self._next = max(now, self._next) + self.min_interval


class EdgarClient:
    def __init__(
        self,
        user_agent: str = config.SEC_USER_AGENT,
        max_rps: float = config.SEC_MAX_RPS,
        raw_dir: Path = config.RAW,
        max_retries: int = 6,
    ):
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}
        )
        self.limiter = RateLimiter(max_rps)
        self.raw_dir = Path(raw_dir)
        self.max_retries = max_retries

    def get(self, url: str, timeout: float = 60.0) -> bytes:
        delay = 1.0
        for attempt in range(1, self.max_retries + 1):
            self.limiter.wait()
            try:
                r = self.session.get(url, timeout=timeout)
            except requests.RequestException as exc:  # connection problems
                log.warning("request error %s (%s), attempt %d", url, exc, attempt)
            else:
                if r.status_code == 200:
                    return r.content
                if r.status_code == 404:
                    raise FileNotFoundError(url)
                log.warning("HTTP %s for %s, attempt %d", r.status_code, url, attempt)
                if r.status_code == 429:
                    delay = max(delay, 10.0)
            time.sleep(delay)
            delay = min(delay * 2, 60.0)
        raise RuntimeError(f"giving up on {url}")

    def _cached(self, path: Path, url: str) -> bytes:
        if path.exists() and path.stat().st_size > 0:
            return path.read_bytes()
        data = self.get(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        tmp.write_bytes(data)
        tmp.replace(path)
        return data

    # -- submissions index --------------------------------------------------
    def submissions(self, cik: str) -> dict:
        """Full submissions index (recent block plus paginated files)."""
        cik = cik.zfill(10)
        name = f"CIK{cik}.json"
        main = json.loads(
            self._cached(self.raw_dir / "submissions" / name, SUBMISSIONS_URL.format(name=name))
        )
        parts = [pd.DataFrame(main["filings"]["recent"])]
        for extra in main["filings"].get("files", []):
            n = extra["name"]
            d = json.loads(
                self._cached(self.raw_dir / "submissions" / n, SUBMISSIONS_URL.format(name=n))
            )
            parts.append(pd.DataFrame(d))
        main["_filings"] = pd.concat(parts, ignore_index=True)
        return main

    def document_path(self, cik: str, accession: str, doc: str) -> Path:
        return self.raw_dir / cik.zfill(10) / accession.replace("-", "") / Path(doc).name

    def document(self, cik: str, accession: str, doc: str) -> bytes:
        url = ARCHIVES_URL.format(
            cik=int(cik), acc_nodash=accession.replace("-", ""), doc=doc
        )
        return self._cached(self.document_path(cik, accession, doc), url)


# --------------------------------------------------------------------------- #
# Filing index
# --------------------------------------------------------------------------- #
def annual_filings(
    filings: pd.DataFrame,
    first_fy: int = config.FIRST_FISCAL_YEAR,
    last_fy: int = config.LAST_FISCAL_YEAR,
) -> pd.DataFrame:
    """Select original 10-K filings (no amendments) whose period of report
    falls in ``[first_fy, last_fy]``.

    ``fiscal_year`` is the calendar year in which the reporting period ends,
    except that periods ending in the first two weeks of January (52/53-week
    years) belong to the previous year. Exact duplicates of a period keep the
    earliest filing; if two distinct periods still share a label (a mislabelled
    EDGAR ``reportDate``), the later period is kept.
    """
    f = filings.copy()
    f = f[f["form"].isin(ANNUAL_FORMS)]
    f["filing_date"] = pd.to_datetime(f["filingDate"])
    f["report_date"] = pd.to_datetime(f["reportDate"], errors="coerce")
    f = f.dropna(subset=["report_date"])
    f["fiscal_year"] = (f["report_date"] - pd.Timedelta(days=15)).dt.year
    f = f[(f["fiscal_year"] >= first_fy) & (f["fiscal_year"] <= last_fy)]
    f = f.sort_values(["report_date", "filing_date"]).drop_duplicates("report_date", keep="first")
    f = f.drop_duplicates("fiscal_year", keep="last").sort_values("fiscal_year")
    cols = ["fiscal_year", "report_date", "filing_date", "accessionNumber", "primaryDocument", "form"]
    out = f[cols].rename(
        columns={"accessionNumber": "accession", "primaryDocument": "primary_document"}
    )
    return out.reset_index(drop=True)


@dataclass
class DownloadResult:
    cik: str
    filer_cik: str
    ticker: str
    accession: str
    fiscal_year: int
    report_date: pd.Timestamp
    filing_date: pd.Timestamp
    path: Path | None
    error: str | None = None


def _fetch_one(client: EdgarClient, cik: str, filer_cik: str, ticker: str, row: pd.Series) -> DownloadResult:
    primary = row["primary_document"] if isinstance(row["primary_document"], str) else ""
    candidates = [primary] if primary else []
    candidates.append(f"{row['accession']}.txt")  # complete submission file
    base = dict(
        cik=cik,
        filer_cik=filer_cik,
        ticker=ticker,
        accession=row["accession"],
        fiscal_year=int(row["fiscal_year"]),
        report_date=row["report_date"],
        filing_date=row["filing_date"],
    )
    last_error = "no candidate document"
    for doc in candidates:
        try:
            client.document(filer_cik, row["accession"], doc)
            return DownloadResult(path=client.document_path(filer_cik, row["accession"], doc), **base)
        except Exception as exc:  # noqa: BLE001 - recorded in the index
            last_error = f"{type(exc).__name__}: {exc}"
    return DownloadResult(path=None, error=last_error, **base)


def build_filing_index(universe: pd.DataFrame, client: EdgarClient) -> list[tuple[str, str, str, pd.Series]]:
    """``(company_cik, filer_cik, ticker, filing_row)`` for every 10-K to fetch,
    including filings of predecessor registrants listed in ``PREDECESSOR_CIKS``.
    When the current and a predecessor registrant both filed for the same
    fiscal year, the current registrant's filing is kept."""
    tasks = []
    for _, u in universe.iterrows():
        frames = []
        for filer in [u["cik"]] + PREDECESSOR_CIKS.get(u["ticker"], []):
            try:
                sub = client.submissions(filer)
            except Exception as exc:  # noqa: BLE001
                log.error("submissions failed for %s: %s", filer, exc)
                continue
            ann = annual_filings(sub["_filings"])
            ann["filer_cik"] = filer.zfill(10)
            frames.append(ann)
        if not frames:
            continue
        ann = pd.concat(frames).drop_duplicates("fiscal_year", keep="first").sort_values("fiscal_year")
        for _, row in ann.iterrows():
            tasks.append((u["cik"], row["filer_cik"], u["ticker"], row))
    return tasks


def download_universe(
    universe: pd.DataFrame,
    client: EdgarClient | None = None,
    workers: int = 4,
    index_path: Path = config.DATA / "filings.csv",
) -> pd.DataFrame:
    """Download the 10-K primary documents for every company in ``universe``
    and write an index (one row per filing) to ``index_path``.
    """
    client = client or EdgarClient()
    tasks = build_filing_index(universe, client)
    log.info("%d 10-K filings to fetch for %d companies", len(tasks), len(universe))

    results: list[DownloadResult] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_fetch_one, client, cik, filer, tic, row) for cik, filer, tic, row in tasks]
        for i, fut in enumerate(as_completed(futs), 1):
            results.append(fut.result())
            if i % 100 == 0:
                log.info("downloaded %d / %d", i, len(futs))
    df = pd.DataFrame(
        [
            {
                "cik": r.cik,
                "filer_cik": r.filer_cik,
                "ticker": r.ticker,
                "fiscal_year": r.fiscal_year,
                "report_date": r.report_date.date(),
                "filing_date": r.filing_date.date(),
                "accession": r.accession,
                "path": str(r.path.relative_to(config.ROOT)) if r.path else None,
                "error": r.error,
            }
            for r in results
        ]
    ).sort_values(["ticker", "fiscal_year"]).reset_index(drop=True)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(index_path, index=False)
    return df


def load_filing_index(index_path: Path = config.DATA / "filings.csv") -> pd.DataFrame:
    return pd.read_csv(
        index_path, dtype={"cik": str, "filer_cik": str}, parse_dates=["report_date", "filing_date"]
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    uni = load_universe()
    config.DATA.mkdir(parents=True, exist_ok=True)
    uni.to_csv(config.DATA / "universe.csv", index=False)
    log.info("universe: %d companies", len(uni))
    out = download_universe(uni)
    log.info("done: %d filings, %d errors", len(out), out["error"].notna().sum())
