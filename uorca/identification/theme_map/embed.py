"""Text preparation, provider guards, cost estimation and embedding (spec 5.1, 5.7, 7, 9.7).

This is the only module in the package that spends money. Everything that could spend is
gated behind :func:`check_provider`, which raises before a client object is ever
constructed.

Nothing here degrades quietly. A missing API key, a Bedrock-configured provider, a
missing research query, a wrong-width embedding matrix and a zero-norm vector all raise.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ... import ai_provider
from .types import Float32Array, ThemeMapConfig

logger = logging.getLogger(__name__)

__all__ = [
    "ThemeMapEmbeddingError",
    "ThemeMapProviderError",
    "ThemeMapQueryMissing",
    "PRICE_PER_1M_TOKENS",
    "MAX_TOKENS_PER_INPUT",
    "BATCH_SIZE",
    "METADATA_FILENAME",
    "QUERY_KEY",
    "build_texts",
    "check_provider",
    "estimate_cost",
    "embed_texts",
    "query_similarity",
    "read_query",
]


#: USD per million tokens for ``text-embedding-3-small``.
PRICE_PER_1M_TOKENS = 0.02

#: Hard input limit of ``text-embedding-3-small``. Longer documents are truncated.
MAX_TOKENS_PER_INPUT = 8191

#: Documents per embeddings request. Matches the validated experiment.
BATCH_SIZE = 256

#: The tokeniser ``text-embedding-3-small`` uses.
TOKEN_ENCODING = "cl100k_base"

METADATA_FILENAME = "identification_metadata.json"
QUERY_KEY = "input_query"

#: A trailing bracketed qualifier, e.g. ``"Foo bar [RNA-seq]"`` -> ``"Foo bar"``. GEO
#: sub-series of one super-series differ only by this qualifier (spec 5.1 step 2).
_BRACKET = re.compile(r"\s*[\[\(][^\]\)]*[\]\)]\s*")

#: The summary scaffolding named in spec 5.1 step 1. Singular/plural variants of the same
#: four words are covered; no other word is stripped.
_SCAFFOLD = re.compile(r"\b(Purpose|Methods?|Results?|Conclusions?)\s*:", re.IGNORECASE)


class ThemeMapEmbeddingError(RuntimeError):
    """Base class for every failure raised by the embedding step."""


class ThemeMapProviderError(ThemeMapEmbeddingError):
    """The configured provider cannot produce embeddings (spec section 7)."""


class ThemeMapQueryMissing(ThemeMapEmbeddingError):
    """The run's research query could not be read from ``identification_metadata.json``.

    Spec 5.7 forbids a hardcoded fallback, so this is a hard failure rather than a
    default query.
    """


# --------------------------------------------------------------------------------------
# Input text — spec section 5.1
# --------------------------------------------------------------------------------------


def strip_title(title: str) -> str:
    """Normalise a title for duplicate detection: drop bracketed qualifiers, casefold.

    ``"Adipocyte atlas [RNA-seq]"`` and ``"Adipocyte atlas (scRNA-seq)"`` both reduce to
    ``"adipocyte atlas"``, which is how GEO sub-series of one super-series are detected.
    """
    return _BRACKET.sub(" ", str(title)).strip().casefold()


def _string_column(frame: pd.DataFrame, column: str) -> list[str]:
    """One column as a list of strings, with missing values as empty strings."""
    return pd.Series(frame[column]).fillna("").astype(str).tolist()


def build_texts(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Build the embedding corpus from a ``Dataset_identification_result.csv`` frame.

    Population and text recipe are fixed by spec section 5.1: rows where
    ``Valid == "Yes"``, text is ``Title + ". " + Summary``, and ``Species`` is
    **excluded** — species is a strong lexical signal that pulls clusters toward organism
    rather than biology.

    Two preparation steps, in order:

    1. Strip ``Purpose:``, ``Methods:``, ``Results:`` and ``Conclusions:`` scaffolding
       from the summary.
    2. Drop duplicate sub-series on the bracket-stripped title, keeping the first
       occurrence. Sub-series duplicates are 9.3-10.5% of rows.

    Every filter logs its before and after counts at INFO, as
    ``CONSTITUTION-analysis.md`` section 2 requires.

    Args:
        df: The parsed results CSV. Must carry ``GEO_Accession``, ``Title``, ``Summary``
            and ``Valid`` columns.

    Returns:
        ``(accessions, texts)``, row-aligned and in input order.

    Raises:
        KeyError: A required column is absent. The caller is handed a broken CSV rather
            than a silently empty corpus.
    """
    required = ("GEO_Accession", "Title", "Summary", "Valid")
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(
            f"Dataset_identification_result.csv is missing required column(s) {missing}; "
            f"present columns are {list(df.columns)}"
        )

    n_all = len(df)
    valid: pd.DataFrame = df.loc[pd.Series(df["Valid"]).astype(str) == "Yes"]
    logger.info("[filter] Valid == 'Yes': %d -> %d rows", n_all, len(valid))

    titles = _string_column(valid, "Title")
    summaries = _string_column(valid, "Summary")
    accessions_in = _string_column(valid, "GEO_Accession")

    # Filter A: a dataset with neither a title nor a summary has no text to embed.
    assembled: list[tuple[str, str, str]] = []  # (accession, title, text)
    for accession, title, summary in zip(accessions_in, titles, summaries):
        title = title.strip()
        summary = " ".join(_SCAFFOLD.sub(" ", summary).split())
        text = f"{title}. {summary}" if (title and summary) else (title or summary)
        if text.strip():
            assembled.append((accession, title, text))
    logger.info(
        "[filter] non-empty Title+Summary text: %d -> %d rows", len(valid), len(assembled)
    )

    # Filter B: sub-series of one super-series share a bracket-stripped title. Keep the
    # first occurrence (spec 5.1 step 2).
    accessions: list[str] = []
    texts: list[str] = []
    seen: set[str] = set()
    for accession, title, text in assembled:
        key = strip_title(title)
        if key and key in seen:
            continue
        seen.add(key)
        accessions.append(accession)
        texts.append(text)
    logger.info(
        "[filter] duplicate sub-series on the bracket-stripped title: %d -> %d rows",
        len(assembled),
        len(accessions),
    )
    logger.info("Embedding corpus: %d datasets from %d retrieved rows", len(accessions), n_all)

    return accessions, texts


# --------------------------------------------------------------------------------------
# Provider guards — spec section 7
# --------------------------------------------------------------------------------------


def configured_provider() -> str:
    """The configured AI provider, read exactly as ``uorca.ai_provider`` reads it.

    Priority: ``UORCA_AI_PROVIDER``, then ``~/.uorca/config.yaml`` ``ai_provider.provider``,
    then ``"openai"``. The helpers are reused rather than reimplemented so the two can
    never drift apart.
    """
    config = ai_provider._load_ai_config()
    provider = ai_provider._get_config_value(
        config,
        "ai_provider",
        "provider",
        env_var=ai_provider.UORCA_AI_PROVIDER,
        default="openai",
    )
    return (provider or "openai").strip().lower()


def check_provider() -> None:
    """Raise unless OpenAI embeddings can actually be produced (spec section 7).

    Two conditions, both checked **before any client is constructed and before any
    spend**:

    * the configured provider is ``bedrock`` — Bedrock exposes no
      ``text-embedding-3-small``, and the theme map's validated numbers were measured on
      that model, so there is no honest fallback;
    * ``OPENAI_API_KEY`` is absent.

    **This Bedrock guard applies to the embedding step only.** The naming call in
    :mod:`~uorca.identification.theme_map.naming` goes through
    :func:`uorca.ai_provider.get_model` and works on any provider, Bedrock included.

    Raises:
        ThemeMapProviderError: On either condition. The Bedrock message names the config
            key to change.
    """
    provider = configured_provider()
    if provider == "bedrock":
        raise ThemeMapProviderError(
            "The theme map embeds text with OpenAI 'text-embedding-3-small', and the "
            "configured AI provider is 'bedrock', which offers no equivalent model. "
            "Set ai_provider.provider to 'openai' in ~/.uorca/config.yaml, or set the "
            "UORCA_AI_PROVIDER environment variable to 'openai', and try again. "
            "(Only the embedding step is affected; theme naming works on any provider.)"
        )

    if not os.environ.get("OPENAI_API_KEY"):
        raise ThemeMapProviderError(
            "OPENAI_API_KEY is not set, so the theme map cannot embed anything. Put it "
            "in your .env file or export it, and try again."
        )


def _make_client():
    """Construct the OpenAI client.

    Isolated in its own function so that the tests can prove :func:`check_provider`
    raises *before* a client is ever constructed.
    """
    from openai import OpenAI

    return OpenAI()


# --------------------------------------------------------------------------------------
# Cost — spec section 9.7
# --------------------------------------------------------------------------------------


def _encoding():
    import tiktoken

    return tiktoken.get_encoding(TOKEN_ENCODING)


def estimate_cost(texts: list[str]) -> tuple[int, float]:
    """Count the tokens that would be billed, and price them.

    Spec 9.7 requires the number shown on the ``Build theme map`` button to come from a
    real token count rather than a guess. Measured references: $0.0038 at n=793 and
    $0.0105 at n=2,065.

    Documents longer than :data:`MAX_TOKENS_PER_INPUT` are truncated before sending, so
    they are counted truncated here too.

    Returns:
        ``(token_count, usd)``.
    """
    if not texts:
        return 0, 0.0

    encoding = _encoding()
    tokens = sum(min(len(encoding.encode(text)), MAX_TOKENS_PER_INPUT) for text in texts)
    return tokens, tokens / 1_000_000 * PRICE_PER_1M_TOKENS


def _truncate(text: str, encoding) -> str:
    token_ids = encoding.encode(text)
    if len(token_ids) <= MAX_TOKENS_PER_INPUT:
        return text
    return encoding.decode(token_ids[:MAX_TOKENS_PER_INPUT])


# --------------------------------------------------------------------------------------
# Embedding
# --------------------------------------------------------------------------------------


def embed_texts(
    texts: list[str], *, config: ThemeMapConfig | None = None
) -> Float32Array:
    """Embed every text with ``text-embedding-3-small``, in batches.

    The API returns unit-norm vectors, but nothing downstream relies on that: the
    clustering step L2-normalises the matrix itself (spec 5.2) and
    :func:`query_similarity` normalises both sides.

    Args:
        texts: The corpus, in the row order the caller wants preserved.
        config: The fixed pipeline configuration; supplies the model name and the
            expected width.

    Returns:
        A float32 ``n x embedding_dim`` matrix, row-aligned with *texts*.

    Raises:
        ThemeMapProviderError: The provider or key guard fired (before any spend).
        ValueError: The API returned the wrong number of vectors or the wrong width.
    """
    settings = config or ThemeMapConfig()
    check_provider()

    if not texts:
        raise ValueError("embed_texts was given no texts; there is nothing to embed.")

    encoding = _encoding()
    payload = [_truncate(text, encoding) for text in texts]

    client = _make_client()
    total = len(payload)
    vectors: list[list[float]] = []
    logger.info(
        "Embedding %d documents with %s in batches of %d",
        total,
        settings.embedding_model,
        BATCH_SIZE,
    )
    for start in range(0, total, BATCH_SIZE):
        batch = payload[start : start + BATCH_SIZE]
        response = client.embeddings.create(model=settings.embedding_model, input=batch)
        # The API returns objects in request order; sort on index defensively so a row
        # can never silently end up beside the wrong accession.
        received = sorted(response.data, key=lambda datum: datum.index)
        if len(received) != len(batch):
            raise ValueError(
                f"Embedding batch starting at {start} asked for {len(batch)} vectors "
                f"and received {len(received)}."
            )
        vectors.extend(datum.embedding for datum in received)
        logger.info("Embedded %d/%d documents", min(start + BATCH_SIZE, total), total)

    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.shape != (total, settings.embedding_dim):
        raise ValueError(
            f"Expected an embedding matrix of shape ({total}, {settings.embedding_dim}) "
            f"from {settings.embedding_model}, got {matrix.shape}."
        )
    return matrix


# --------------------------------------------------------------------------------------
# QuerySim — spec section 5.7
# --------------------------------------------------------------------------------------


def query_similarity(
    query: str,
    embeddings: Float32Array,
    *,
    config: ThemeMapConfig | None = None,
) -> tuple[list[float], list[int]]:
    """Cosine similarity of every dataset to the run's research query.

    **QuerySim is a sortable column and nothing more. It never drives colour, position
    or clustering.** It is also decisively worse than the LLM relevance score: on the 11
    ground-truth datasets of the PPARgamma benchmark the LLM score gave a median rank of
    10 of 1,199 and put all 11 in the top 100, while cosine similarity gave a median rank
    of 70 and a worst rank of 467 (spec 5.7).

    The query is embedded with the same model in one extra API call, then dot-producted
    against every dataset vector. Both sides are L2-normalised first, so the result is a
    true cosine and does not depend on the API happening to return unit vectors.

    Args:
        query: The run's research query, read from ``identification_metadata.json``.
        embeddings: The ``n x d`` dataset matrix.
        config: The fixed pipeline configuration.

    Returns:
        ``(query_sim, query_sim_rank)``, both row-aligned with *embeddings*. Rank 1 is the
        highest similarity. Ties break on row order.

    Raises:
        ValueError: The query is blank, the matrix is not 2-D, its width does not match
            the query vector, or any row has zero norm.
    """
    if not query.strip():
        raise ValueError("query_similarity was given a blank query.")

    matrix = np.asarray(embeddings, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError(
            f"query_similarity expects a non-empty 2-D embedding matrix, got shape "
            f"{matrix.shape}."
        )

    query_matrix = np.asarray(embed_texts([query], config=config), dtype=np.float64)
    if query_matrix.shape[0] != 1:
        raise ValueError(
            f"Embedding the query returned {query_matrix.shape[0]} vectors, expected 1."
        )
    if query_matrix.shape[1] != matrix.shape[1]:
        raise ValueError(
            f"Query vector has width {query_matrix.shape[1]} but the dataset matrix has "
            f"width {matrix.shape[1]}; they must come from the same model."
        )

    query_vector = _l2_normalise(query_matrix, "query")[0]
    normalised = _l2_normalise(matrix, "dataset")

    sims = normalised @ query_vector

    # Rank 1 is the highest similarity. argsort is stable, so ties break on row order.
    order = np.argsort(-sims, kind="stable")
    ranks = np.empty(sims.shape[0], dtype=int)
    ranks[order] = np.arange(1, sims.shape[0] + 1)

    return [float(value) for value in sims], [int(value) for value in ranks]


def _l2_normalise(matrix: np.ndarray, what: str) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms == 0):
        zero_rows = np.flatnonzero(norms == 0).tolist()
        raise ValueError(
            f"{what} embedding row(s) {zero_rows[:10]} have zero norm; a zero vector "
            "cannot be compared and would silently become NaN."
        )
    return matrix / norms[:, None]


# --------------------------------------------------------------------------------------
# The research query — spec section 5.7 ("It is never hardcoded")
# --------------------------------------------------------------------------------------


def read_query(run_dir: Path | str) -> str:
    """Read the run's research query from ``identification_metadata.json``.

    Spec 5.7: the query is never hardcoded. Every failure to read it raises, so a theme
    map is never built against the wrong question.

    Raises:
        ThemeMapQueryMissing: The file is absent or unreadable, the ``input_query`` key is
            absent, or the query is blank.
    """
    path = Path(run_dir) / METADATA_FILENAME
    if not path.is_file():
        raise ThemeMapQueryMissing(
            f"{path} does not exist, so the research query for this run cannot be read. "
            "The theme map never falls back to a hardcoded query."
        )

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ThemeMapQueryMissing(f"{path} could not be read as JSON: {exc}") from exc

    if not isinstance(payload, dict) or QUERY_KEY not in payload:
        raise ThemeMapQueryMissing(
            f"{path} has no '{QUERY_KEY}' key, so the research query for this run is "
            "unknown."
        )

    query = str(payload[QUERY_KEY])
    if not query.strip():
        raise ThemeMapQueryMissing(f"{path} holds a blank '{QUERY_KEY}'.")
    return query
