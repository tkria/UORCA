"""Unit tests for the theme-map embedding step.

Covers spec sections 5.1 (input text), 5.7 (QuerySim), 7 (the Bedrock and
``OPENAI_API_KEY`` guards) and 9.7 (the cost estimate comes from a token count).

**No test here makes a real API call.** Every path that would construct an OpenAI client
replaces :func:`uorca.identification.theme_map.embed._make_client` with a stub, and the
two guard tests replace it with a stub that raises if it is ever called — that is how
"raises *before* any client is constructed" is asserted rather than assumed.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from uorca.identification.theme_map import embed as embed_module
from uorca.identification.theme_map.embed import (
    ThemeMapProviderError,
    ThemeMapQueryMissing,
    build_texts,
    check_provider,
    embed_texts,
    estimate_cost,
    query_similarity,
    read_query,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


class _ExplodingClientFactory:
    """Stands in for ``_make_client`` and fails loudly if anything constructs a client."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self):  # pragma: no cover - the point is that it is never reached
        self.calls += 1
        raise AssertionError(
            "an OpenAI client was constructed; the guard should have raised first"
        )


class _FakeEmbeddings:
    def __init__(self, vectors_by_text: dict[str, list[float]], dim: int) -> None:
        self._vectors = vectors_by_text
        self._dim = dim
        self.batch_sizes: list[int] = []

    def create(self, *, model: str, input: list[str]):  # noqa: A002 - mirrors the SDK
        self.batch_sizes.append(len(input))
        data = []
        for index, text in enumerate(input):
            vector = self._vectors.get(text)
            if vector is None:
                vector = [float(len(text) % 7)] + [0.0] * (self._dim - 1)
            data.append(_FakeDatum(index=index, embedding=vector))
        return _FakeResponse(data=data)


class _FakeDatum:
    def __init__(self, index: int, embedding: list[float]) -> None:
        self.index = index
        self.embedding = embedding


class _FakeResponse:
    def __init__(self, data: list[_FakeDatum]) -> None:
        self.data = data


class _FakeClient:
    def __init__(self, vectors_by_text: dict[str, list[float]], dim: int) -> None:
        self.embeddings = _FakeEmbeddings(vectors_by_text, dim)


@pytest.fixture
def openai_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the OpenAI provider and a present key, independent of the local config file.

    ``UORCA_AI_PROVIDER`` wins over ``~/.uorca/config.yaml`` in ``uorca.ai_provider``, so
    setting it here keeps these tests deterministic on a developer machine configured for
    Bedrock.
    """
    monkeypatch.setenv("UORCA_AI_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")


def _frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    columns = pd.Index(["GEO_Accession", "Title", "Summary", "Valid"])
    return pd.DataFrame(rows, columns=columns)


# --------------------------------------------------------------------------------------
# check_provider — spec section 7
# --------------------------------------------------------------------------------------


def test_check_provider_raises_on_bedrock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UORCA_AI_PROVIDER", "bedrock")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")

    with pytest.raises(ThemeMapProviderError) as excinfo:
        check_provider()

    message = str(excinfo.value)
    assert "bedrock" in message.lower()
    # The message must name the config key the user has to change (spec section 7).
    assert "ai_provider.provider" in message
    assert "UORCA_AI_PROVIDER" in message


def test_check_provider_raises_before_a_client_is_constructed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spec 10.1 item 5: the Bedrock guard fires before any spend is possible."""
    monkeypatch.setenv("UORCA_AI_PROVIDER", "bedrock")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    factory = _ExplodingClientFactory()
    monkeypatch.setattr(embed_module, "_make_client", factory)

    with pytest.raises(ThemeMapProviderError):
        embed_texts(["some text"])

    assert factory.calls == 0


def test_check_provider_raises_when_api_key_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("UORCA_AI_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ThemeMapProviderError) as excinfo:
        check_provider()

    assert "OPENAI_API_KEY" in str(excinfo.value)


def test_missing_api_key_raises_before_a_client_is_constructed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("UORCA_AI_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    factory = _ExplodingClientFactory()
    monkeypatch.setattr(embed_module, "_make_client", factory)

    with pytest.raises(ThemeMapProviderError):
        embed_texts(["some text"])

    assert factory.calls == 0


def test_check_provider_accepts_openai_with_a_key(openai_provider: None) -> None:
    assert check_provider() is None


# --------------------------------------------------------------------------------------
# build_texts — spec section 5.1
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "scaffold", ["Purpose:", "Methods:", "Results:", "Conclusions:"]
)
def test_build_texts_strips_each_scaffolding_prefix(scaffold: str) -> None:
    """Spec 10.1 item 6, one parametrisation per prefix named in spec 5.1."""
    df = _frame(
        [
            {
                "GEO_Accession": "GSE1",
                "Title": "Cardiac fibroblast atlas",
                "Summary": f"{scaffold} we profiled ventricular tissue.",
                "Valid": "Yes",
            }
        ]
    )

    accessions, texts = build_texts(df)

    assert accessions == ["GSE1"]
    assert scaffold not in texts[0]
    assert scaffold.rstrip(":").lower() not in texts[0].lower()
    assert "ventricular tissue" in texts[0]
    assert texts[0].startswith("Cardiac fibroblast atlas. ")


def test_build_texts_keeps_only_valid_rows() -> None:
    df = _frame(
        [
            {"GEO_Accession": "GSE1", "Title": "A", "Summary": "a", "Valid": "Yes"},
            {"GEO_Accession": "GSE2", "Title": "B", "Summary": "b", "Valid": "No"},
        ]
    )

    accessions, texts = build_texts(df)

    assert accessions == ["GSE1"]
    assert len(texts) == 1


def test_build_texts_drops_duplicate_sub_series_and_keeps_the_first() -> None:
    """Spec 5.1 step 2: sub-series share a bracket-stripped title."""
    df = _frame(
        [
            {
                "GEO_Accession": "GSE100",
                "Title": "Adipocyte differentiation atlas [RNA-seq]",
                "Summary": "First sub-series.",
                "Valid": "Yes",
            },
            {
                "GEO_Accession": "GSE101",
                "Title": "Adipocyte differentiation atlas [ATAC-seq]",
                "Summary": "Second sub-series.",
                "Valid": "Yes",
            },
            {
                "GEO_Accession": "GSE102",
                "Title": "A different study entirely",
                "Summary": "Unrelated.",
                "Valid": "Yes",
            },
        ]
    )

    accessions, texts = build_texts(df)

    assert accessions == ["GSE100", "GSE102"]
    assert "First sub-series." in texts[0]


def test_build_texts_logs_before_and_after_counts(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """CONSTITUTION-analysis section 2: every filter logs before and after counts."""
    df = _frame(
        [
            {"GEO_Accession": "GSE1", "Title": "Shared title [a]", "Summary": "x", "Valid": "Yes"},
            {"GEO_Accession": "GSE2", "Title": "Shared title [b]", "Summary": "y", "Valid": "Yes"},
            {"GEO_Accession": "GSE3", "Title": "Other", "Summary": "z", "Valid": "No"},
        ]
    )

    with caplog.at_level(logging.INFO, logger=embed_module.__name__):
        build_texts(df)

    messages = [record.getMessage() for record in caplog.records]
    joined = "\n".join(messages)
    # Valid filter: 3 rows in, 2 out. Duplicate filter: 2 rows in, 1 out.
    assert "3" in joined and "2" in joined and "1" in joined
    assert any("Valid" in message for message in messages)
    assert any("duplicate" in message.lower() for message in messages)


def test_build_texts_raises_on_a_missing_column() -> None:
    df = pd.DataFrame([{"GEO_Accession": "GSE1", "Title": "A", "Valid": "Yes"}])

    with pytest.raises(KeyError):
        build_texts(df)


# --------------------------------------------------------------------------------------
# estimate_cost — spec section 9.7
# --------------------------------------------------------------------------------------


def test_estimate_cost_lands_near_the_measured_reference() -> None:
    """Measured references: $0.0038 at n=793 and $0.0105 at n=2,065 (spec 9.7)."""
    # A realistic Title + Summary is a couple of hundred tokens.
    text = (
        "Transcriptional profiling of ventricular tissue in hypertrophic cardiomyopathy. "
        "We performed bulk RNA sequencing on septal myectomy samples and matched controls "
        "to characterise the disease signature across fibrosis, sarcomere and metabolic "
        "gene programmes in human myocardium. "
    ) * 8
    texts = [text] * 793

    tokens, usd = estimate_cost(texts)

    assert tokens > 0
    assert usd == pytest.approx(tokens / 1_000_000 * 0.02)
    # Order of magnitude: single-digit thousandths of a dollar.
    assert 0.001 < usd < 0.02


def test_estimate_cost_of_nothing_is_zero() -> None:
    assert estimate_cost([]) == (0, 0.0)


# --------------------------------------------------------------------------------------
# embed_texts
# --------------------------------------------------------------------------------------


def test_embed_texts_returns_a_float32_matrix(
    monkeypatch: pytest.MonkeyPatch, openai_provider: None
) -> None:
    dim = 1536
    client = _FakeClient({}, dim)
    monkeypatch.setattr(embed_module, "_make_client", lambda: client)

    matrix = embed_texts(["alpha", "beta", "gamma"])

    assert matrix.shape == (3, dim)
    assert matrix.dtype == np.float32


def test_embed_texts_raises_when_the_api_returns_the_wrong_width(
    monkeypatch: pytest.MonkeyPatch, openai_provider: None
) -> None:
    client = _FakeClient({"alpha": [1.0, 2.0, 3.0]}, 3)
    monkeypatch.setattr(embed_module, "_make_client", lambda: client)

    with pytest.raises(ValueError, match="1536"):
        embed_texts(["alpha"])


# --------------------------------------------------------------------------------------
# query_similarity — spec section 5.7
# --------------------------------------------------------------------------------------


def test_query_similarity_ranks_an_identical_vector_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embeddings = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],  # identical to the query direction
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    query_vector = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    monkeypatch.setattr(
        embed_module, "embed_texts", lambda texts, **kwargs: query_vector
    )

    sims, ranks = query_similarity("a research query", embeddings)

    assert len(sims) == 3
    assert ranks[1] == 1
    assert sims[1] == pytest.approx(1.0)
    assert sorted(ranks) == [1, 2, 3]


def test_query_similarity_raises_on_a_zero_vector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embeddings = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    monkeypatch.setattr(
        embed_module,
        "embed_texts",
        lambda texts, **kwargs: np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
    )

    with pytest.raises(ValueError, match="zero"):
        query_similarity("a research query", embeddings)


# --------------------------------------------------------------------------------------
# read_query — spec section 5.7 ("It is never hardcoded")
# --------------------------------------------------------------------------------------


def test_read_query_reads_the_input_query(tmp_path: Path) -> None:
    (tmp_path / "identification_metadata.json").write_text(
        json.dumps({"input_query": "PPARgamma agonists in adipocytes"}), encoding="utf-8"
    )

    assert read_query(tmp_path) == "PPARgamma agonists in adipocytes"


def test_read_query_raises_when_the_metadata_file_is_missing(tmp_path: Path) -> None:
    with pytest.raises(ThemeMapQueryMissing) as excinfo:
        read_query(tmp_path)

    assert "identification_metadata.json" in str(excinfo.value)


def test_read_query_raises_when_the_key_is_missing(tmp_path: Path) -> None:
    (tmp_path / "identification_metadata.json").write_text(
        json.dumps({"threshold_used": 7.0}), encoding="utf-8"
    )

    with pytest.raises(ThemeMapQueryMissing) as excinfo:
        read_query(tmp_path)

    assert "input_query" in str(excinfo.value)


def test_read_query_raises_when_the_query_is_blank(tmp_path: Path) -> None:
    (tmp_path / "identification_metadata.json").write_text(
        json.dumps({"input_query": "   "}), encoding="utf-8"
    )

    with pytest.raises(ThemeMapQueryMissing):
        read_query(tmp_path)
