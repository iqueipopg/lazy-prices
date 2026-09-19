"""10-K text processing: HTML cleaning, item extraction and document
similarity between consecutive annual reports of the same company.

Two similarity measures from Cohen, Malloy and Nguyen (2020) are implemented:

* **cosine similarity** between TF-IDF vectors, where the inverse document
  frequency is computed point-in-time from 10-Ks filed *before* the month of
  the later filing (so the score for a filing only depends on documents that
  were public when it was published);
* **Jaccard similarity** between the sets of distinct terms of the two
  documents.

Both are computed for the full document and for Item 1A (Risk Factors) and
Item 7 (Management's Discussion and Analysis).
"""

from __future__ import annotations

import html
import logging
import re
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from sklearn.feature_extraction.text import CountVectorizer

from . import config

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

TOKEN_PATTERN = r"(?u)\b[a-z][a-z]+\b"
SECTIONS = ("full", "1A", "7")
MIN_SECTION_CHARS = 2000  # shorter extractions are treated as missing
_BLOCK_TAGS = ["p", "div", "br", "tr", "li", "table", "h1", "h2", "h3", "h4", "h5", "h6", "hr", "pre", "blockquote"]

_ITEM_PATTERNS = {
    "1A": (
        r"item\s*1a\W{0,10}risk\s+factors",
        r"item\s*1b\W{0,10}unresolved|item\s*2\W{0,10}(?:properties|description\s+of\s+propert)",
    ),
    "7": (
        r"item\s*7\W{0,10}management",
        r"item\s*7a\W{0,10}quantitative|item\s*8\W{0,10}(?:consolidated\s+)?financial\s+statements",
    ),
}


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def _split_sgml_documents(raw: str) -> str:
    """For complete-submission ``.txt`` files, keep only the first 10-K
    ``<DOCUMENT>`` block (drops exhibits and graphics)."""
    docs = re.findall(r"<DOCUMENT>(.*?)</DOCUMENT>", raw, flags=re.S | re.I)
    if not docs:
        return raw
    for d in docs:
        m = re.search(r"<TYPE>\s*([^\s<]+)", d, flags=re.I)
        if m and m.group(1).upper().startswith("10-K"):
            return d
    return docs[0]


def html_to_text(raw: bytes | str) -> str:
    """Convert a 10-K primary document (HTML, inline XBRL or plain text) to
    normalised plain text with one line per block-level element."""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    raw = _split_sgml_documents(raw)
    if re.search(r"<\s*(html|body|div|p|table)\b", raw, flags=re.I):
        soup = BeautifulSoup(raw, "lxml")
        for tag in soup(["script", "style", "head", "title"]):
            tag.decompose()
        # inline-XBRL hidden facts and any element hidden through CSS
        for tag in soup.find_all(re.compile(r"^ix:(header|hidden)$", re.I)):
            tag.decompose()
        for tag in soup.find_all(style=re.compile(r"display\s*:\s*none", re.I)):
            tag.decompose()
        # newlines only at block boundaries, so inline spans never split words
        for tag in soup.find_all(_BLOCK_TAGS):
            tag.insert_before("\n")
            tag.insert_after("\n")
        for tag in soup.find_all("td"):
            tag.insert_after(" ")
        text = soup.get_text("")
    else:
        text = re.sub(r"<[^>]+>", " ", raw)  # plain text with SGML markers
    text = html.unescape(text)
    text = text.replace("\xa0", " ").replace("​", "")
    lines = (re.sub(r"[ \t\r\f\v]+", " ", ln).strip() for ln in text.split("\n"))
    text = "\n".join(ln for ln in lines if ln)
    return text


def extract_item(text: str, item: str) -> str:
    """Return the text of ``item`` ('1A' or '7') or an empty string.

    All occurrences of the start heading are paired with the first end heading
    that follows them and the longest span is kept. Line-anchored matches are
    tried first (headings sit on their own line after cleaning; table-of-contents
    entries and cross-references produce short spans and lose).
    """
    start_pat, end_pat = _ITEM_PATTERNS[item]
    low = text.lower()
    for anchored in (True, False):
        prefix = r"^[\s\W]*" if anchored else r""
        starts = [m.start() for m in re.finditer(prefix + start_pat, low, re.M)]
        ends = [m.start() for m in re.finditer(prefix + end_pat, low, re.M)]
        best = ""
        if starts and ends:
            ends_arr = np.array(ends)
            for s in starts:
                nxt = ends_arr[ends_arr > s]
                if len(nxt) == 0:
                    continue
                span = text[s : int(nxt[0])]
                if len(span) > len(best):
                    best = span
        if len(best) >= MIN_SECTION_CHARS:
            return best.strip()
    return ""


def tokens(text: str) -> list[str]:
    """Lower-case alphabetic tokens of at least two letters."""
    return re.findall(TOKEN_PATTERN, text.lower())


# --------------------------------------------------------------------------- #
# Similarity measures (small, testable versions on token lists)
# --------------------------------------------------------------------------- #
def jaccard_similarity(a, b) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return float("nan")
    return len(sa & sb) / len(sa | sb)


def cosine_similarity(u, v) -> float:
    u = np.asarray(u, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return float("nan")
    return float(u @ v / (nu * nv))


def tfidf_cosine(a: list[str], b: list[str], idf: dict[str, float] | None = None) -> float:
    """Cosine similarity between TF-IDF vectors of two token lists. With
    ``idf=None`` every term has weight 1 (plain term-frequency cosine)."""
    vocab = sorted(set(a) | set(b))
    idx = {t: i for i, t in enumerate(vocab)}
    w = np.array([idf.get(t, 1.0) if idf else 1.0 for t in vocab])
    u = np.zeros(len(vocab))
    v = np.zeros(len(vocab))
    for t in a:
        u[idx[t]] += 1
    for t in b:
        v[idx[t]] += 1
    return cosine_similarity(u * w, v * w)


# --------------------------------------------------------------------------- #
# Corpus processing
# --------------------------------------------------------------------------- #
def processed_paths(cik: str, accession: str, out_dir: Path = config.PROCESSED) -> dict[str, Path]:
    """Accession numbers are unique across EDGAR, so ``cik`` (the company the
    filing is attributed to) is only used to organise the tree."""
    base = Path(out_dir) / cik / accession.replace("-", "")
    return {sec: base / f"{sec}.txt" for sec in SECTIONS}


def _process_filing(args: tuple[str, str, str, str]) -> dict:
    cik, accession, raw_path, out_dir = args
    paths = processed_paths(cik, accession, Path(out_dir))
    if all(p.exists() for p in paths.values()):
        lens = {s: p.stat().st_size for s, p in paths.items()}
    else:
        try:
            text = html_to_text(Path(raw_path).read_bytes())
        except Exception as exc:  # noqa: BLE001
            log.error("failed to parse %s: %s", raw_path, exc)
            text = ""
        parts = {"full": text, "1A": extract_item(text, "1A"), "7": extract_item(text, "7")}
        paths["full"].parent.mkdir(parents=True, exist_ok=True)
        for sec, p in paths.items():
            p.write_text(parts[sec], encoding="utf-8")
        lens = {s: len(parts[s]) for s in SECTIONS}
    return {"cik": cik, "accession": accession, **{f"len_{s}": lens[s] for s in SECTIONS}}


def process_corpus(index: pd.DataFrame, out_dir: Path = config.PROCESSED, workers: int = 4) -> pd.DataFrame:
    """Clean every downloaded filing and store full text and items 1A and 7 as
    plain-text files under ``out_dir``. Returns the index with character
    counts per section appended."""
    idx = index.dropna(subset=["path"]).copy()
    args = [(r.cik, r.accession, str(config.ROOT / r.path), str(out_dir)) for r in idx.itertuples()]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(_process_filing, args, chunksize=8))
    lens = pd.DataFrame(rows)
    return idx.merge(lens, on=["cik", "accession"], how="left")


def _month_index(dates: pd.Series) -> np.ndarray:
    d = pd.to_datetime(dates)
    return (d.dt.year * 12 + d.dt.month).to_numpy()


def _pit_idf(counts: sp.csr_matrix, months: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Point-in-time smoothed IDF. Returns ``(unique_months, idf_matrix)`` where
    row ``k`` holds the IDF computed from all documents filed in months
    strictly before ``unique_months[k]``."""
    presence = (counts > 0).astype(np.int64)
    uniq = np.unique(months)
    n_feat = counts.shape[1]
    idf = np.empty((len(uniq), n_feat))
    df = np.zeros(n_feat)
    n_docs = 0
    for k, m in enumerate(uniq):
        # sklearn's smooth idf: log((1 + n) / (1 + df)) + 1
        idf[k] = np.log((1.0 + n_docs) / (1.0 + df)) + 1.0
        rows = np.where(months == m)[0]
        df += np.asarray(presence[rows].sum(axis=0)).ravel()
        n_docs += len(rows)
    return uniq, idf


def _row_cosine(x: sp.csr_matrix, y: sp.csr_matrix) -> np.ndarray:
    num = np.asarray(x.multiply(y).sum(axis=1)).ravel()
    den = np.sqrt(np.asarray(x.multiply(x).sum(axis=1)).ravel()) * np.sqrt(
        np.asarray(y.multiply(y).sum(axis=1)).ravel()
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def _row_jaccard(x: sp.csr_matrix, y: sp.csr_matrix) -> np.ndarray:
    px = (x > 0).astype(np.int64)
    py = (y > 0).astype(np.int64)
    inter = np.asarray(px.multiply(py).sum(axis=1)).ravel()
    union = np.asarray(px.sum(axis=1)).ravel() + np.asarray(py.sum(axis=1)).ravel() - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        return inter / union


def pair_consecutive(index: pd.DataFrame, min_gap_days: int = 270, max_gap_days: int = 500) -> pd.DataFrame:
    """Pair each filing with the previous annual filing of the same company.
    Pairs whose report dates are not roughly one year apart are dropped."""
    idx = index.sort_values(["cik", "report_date"]).copy()
    prev = idx.groupby("cik")[["accession", "fiscal_year", "report_date", "filing_date"]].shift(1)
    idx["prev_accession"] = prev["accession"]
    idx["prev_fiscal_year"] = prev["fiscal_year"]
    idx["prev_report_date"] = prev["report_date"]
    idx["prev_filing_date"] = prev["filing_date"]
    idx = idx.dropna(subset=["prev_accession"])
    gap = (idx["report_date"] - idx["prev_report_date"]).dt.days
    idx = idx[(gap >= min_gap_days) & (gap <= max_gap_days)].copy()
    idx["prev_fiscal_year"] = idx["prev_fiscal_year"].astype(int)
    return idx.reset_index(drop=True)


def compute_similarity(
    index: pd.DataFrame,
    processed_dir: Path = config.PROCESSED,
    max_features: int | None = 200_000,
) -> pd.DataFrame:
    """Similarity between consecutive 10-Ks for every section.

    ``index`` must contain ``cik, ticker, fiscal_year, report_date,
    filing_date, accession, path`` for every processed filing.
    """
    idx = index.dropna(subset=["path"]).reset_index(drop=True).copy()
    idx["month"] = _month_index(idx["filing_date"])
    pairs = pair_consecutive(idx)
    pos = {a: i for i, a in enumerate(idx["accession"])}
    cur = pairs["accession"].map(pos).to_numpy()
    prv = pairs["prev_accession"].map(pos).to_numpy()

    out = pairs[
        [
            "cik",
            "ticker",
            "fiscal_year",
            "prev_fiscal_year",
            "report_date",
            "filing_date",
            "prev_filing_date",
            "accession",
            "prev_accession",
        ]
    ].copy()
    for sec in SECTIONS:
        files = [str(processed_paths(r.cik, r.accession, processed_dir)[sec]) for r in idx.itertuples()]
        lengths = np.array([Path(f).stat().st_size if Path(f).exists() else 0 for f in files])
        vec = CountVectorizer(
            input="filename",
            encoding="utf-8",
            decode_error="replace",
            token_pattern=TOKEN_PATTERN,
            lowercase=True,
            max_features=max_features,
            dtype=np.int32,
        )
        counts = vec.fit_transform(files).tocsr()
        log.info("section %s: %d docs, %d terms", sec, *counts.shape)
        months, idf = _pit_idf(counts, idx["month"].to_numpy())
        month_row = {m: k for k, m in enumerate(months)}
        w = idf[[month_row[m] for m in idx.loc[cur, "month"]]]
        x = counts[cur].multiply(w).tocsr()
        y = counts[prv].multiply(w).tocsr()
        cos = _row_cosine(x, y)
        jac = _row_jaccard(counts[cur], counts[prv])
        valid = (lengths[cur] >= MIN_SECTION_CHARS) & (lengths[prv] >= MIN_SECTION_CHARS)
        cos[~valid] = np.nan
        jac[~valid] = np.nan
        key = sec.lower()
        out[f"cos_{key}"] = cos
        out[f"jac_{key}"] = jac
        out[f"len_{key}"] = lengths[cur]
    return out.sort_values(["ticker", "fiscal_year"]).reset_index(drop=True)
