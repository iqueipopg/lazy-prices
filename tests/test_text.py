"""Similarity measures and 10-K text handling on synthetic inputs."""

import math

import numpy as np
import pandas as pd
import pytest

from lazyprices import text


def test_jaccard_known_values():
    assert text.jaccard_similarity(["a", "b", "c"], ["b", "c", "d"]) == pytest.approx(0.5)
    assert text.jaccard_similarity(["a", "a", "b"], ["a", "b", "b"]) == pytest.approx(1.0)
    assert text.jaccard_similarity(["a"], ["b"]) == pytest.approx(0.0)
    assert math.isnan(text.jaccard_similarity([], []))


def test_tf_cosine_known_values():
    # counts: u = (2, 1), v = (1, 2): cos = 4 / 5
    assert text.tfidf_cosine(["a", "a", "b"], ["a", "b", "b"]) == pytest.approx(0.8)
    assert text.tfidf_cosine(["x", "y"], ["x", "y"]) == pytest.approx(1.0)
    assert text.tfidf_cosine(["x"], ["y"]) == pytest.approx(0.0)


def test_tfidf_cosine_downweights_common_terms():
    # 'the' is shared but has near-zero idf; only the informative terms matter
    idf = {"the": 0.0, "risk": 2.0, "growth": 2.0}
    plain = text.tfidf_cosine(["the", "risk"], ["the", "growth"])
    weighted = text.tfidf_cosine(["the", "risk"], ["the", "growth"], idf)
    assert plain == pytest.approx(0.5)
    assert weighted == pytest.approx(0.0)


def test_cosine_similarity_vectors():
    assert text.cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert text.cosine_similarity([3, 4], [3, 4]) == pytest.approx(1.0)
    assert math.isnan(text.cosine_similarity([0, 0], [1, 1]))


SYNTHETIC_10K = """
UNITED STATES SECURITIES AND EXCHANGE COMMISSION
FORM 10-K
TABLE OF CONTENTS
Item 1. Business 3
Item 1A. Risk Factors 10
Item 1B. Unresolved Staff Comments 20
Item 7. Management's Discussion and Analysis of Financial Condition 30
Item 7A. Quantitative and Qualitative Disclosures About Market Risk 45
PART I
Item 1. Business
We make widgets. See Item 1A. Risk Factors for a discussion of risks.
{business}
Item 1A. Risk Factors
{risk}
Item 1B. Unresolved Staff Comments
None.
Item 2. Properties
We own a factory.
PART II
Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations
{mdna}
Item 7A. Quantitative and Qualitative Disclosures About Market Risk
Interest rate risk is limited.
Item 8. Financial Statements and Supplementary Data
"""


def _doc(risk_word="regulation", mdna_word="revenue"):
    filler = lambda w: " ".join(f"{w} sentence number {i}." for i in range(400))  # noqa: E731
    return SYNTHETIC_10K.format(business=filler("widgets"), risk=filler(risk_word), mdna=filler(mdna_word))


def test_extract_items_skip_table_of_contents_and_cross_references():
    doc = _doc()
    risk = text.extract_item(doc, "1A")
    mdna = text.extract_item(doc, "7")
    assert risk.lower().startswith("item 1a")
    assert "regulation sentence number 399" in risk
    assert "widgets sentence" not in risk  # the cross-reference in Item 1 did not win
    assert "Unresolved Staff Comments\nNone." not in risk
    assert mdna.lower().startswith("item 7")
    assert "revenue sentence number 399" in mdna
    assert "Interest rate risk" not in mdna


def test_extract_item_missing_returns_empty():
    assert text.extract_item("Item 1. Business\nshort text", "1A") == ""


def test_html_to_text_strips_markup_and_hidden_elements():
    raw = b"""<html><head><title>X</title><style>p{}</style></head><body>
    <ix:header><ix:hidden>HIDDEN FACT</ix:hidden></ix:header>
    <div style="display:none">INVISIBLE</div>
    <p>Item&nbsp;1A.&#160;Risk&nbsp;Factors</p><table><tr><td>Cell&amp;Co</td></tr></table>
    </body></html>"""
    out = text.html_to_text(raw)
    assert "HIDDEN FACT" not in out and "INVISIBLE" not in out and "<" not in out
    assert "Item 1A. Risk Factors" in out
    assert "Cell&Co" in out


def test_compute_similarity_on_synthetic_corpus(tmp_path):
    """Two firms, three years: identical consecutive docs give 1.0, disjoint
    vocabularies give 0.0, and the point-in-time IDF never looks ahead."""
    rows = []
    docs = {
        ("A", 2010): "alpha beta gamma " * 300,
        ("A", 2011): "alpha beta gamma " * 300,  # identical -> 1.0
        ("A", 2012): "delta epsilon zeta " * 300,  # disjoint -> 0.0
        ("B", 2010): "alpha beta omega " * 300,
        ("B", 2011): "alpha beta omega theta " * 300,
        ("B", 2012): "alpha beta omega theta " * 300,
    }
    for (tic, fy), body in docs.items():
        acc = f"000-{tic}-{fy}"
        paths = text.processed_paths(f"cik{tic}", acc, tmp_path)
        paths["full"].parent.mkdir(parents=True)
        for p in paths.values():
            p.write_text(body)
        rows.append(
            {
                "cik": f"cik{tic}",
                "ticker": tic,
                "fiscal_year": fy,
                "accession": acc,
                "report_date": pd.Timestamp(f"{fy}-12-31"),
                "filing_date": pd.Timestamp(f"{fy + 1}-02-20"),
                "path": "x",
            }
        )
    sim = text.compute_similarity(pd.DataFrame(rows), processed_dir=tmp_path, max_features=None)
    a = sim.set_index(["ticker", "fiscal_year"])
    assert a.loc[("A", 2011), "cos_full"] == pytest.approx(1.0)
    assert a.loc[("A", 2011), "jac_full"] == pytest.approx(1.0)
    assert a.loc[("A", 2012), "cos_full"] == pytest.approx(0.0)
    assert a.loc[("A", 2012), "jac_full"] == pytest.approx(0.0)
    # B 2011 vs 2010: sets {alpha,beta,omega,theta} vs {alpha,beta,omega} -> 3/4
    assert a.loc[("B", 2011), "jac_full"] == pytest.approx(0.75)
    assert a.loc[("B", 2012), "cos_1a"] == pytest.approx(1.0)
    assert len(sim) == 4  # first year of each firm has no predecessor


def test_pit_idf_uses_only_earlier_months():
    import scipy.sparse as sp

    counts = sp.csr_matrix(np.array([[1, 0], [1, 1], [0, 1]]))
    months = np.array([1, 2, 2])
    uniq, idf = text._pit_idf(counts, months)
    assert list(uniq) == [1, 2]
    # month 1: nothing before -> smooth idf log(1/1)+1 = 1 for all terms
    assert np.allclose(idf[0], 1.0)
    # month 2: one earlier doc containing term 0 only
    assert idf[1, 0] == pytest.approx(np.log(2 / 2) + 1)
    assert idf[1, 1] == pytest.approx(np.log(2 / 1) + 1)
