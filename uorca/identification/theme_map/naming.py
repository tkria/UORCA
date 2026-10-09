"""Theme keywords, representative titles and the single naming call (spec 5.4, 5.5, 5.6).

Three steps, in the order the pipeline uses them:

1. :func:`ctfidf` — the evidence. Real class-based TF-IDF over the clustered texts.
2. :func:`select_representatives` — the illustration. Real titles from each theme.
3. :func:`name_themes` — one structured LLM call that turns both into a label and a
   one-line description per theme.

Unlike :mod:`~uorca.identification.theme_map.embed`, this module works on **any**
provider: it goes through :func:`uorca.ai_provider.get_model`, so Bedrock is fine here.
The Bedrock guard belongs to the embedding step alone.

Every failure raises. Spec section 7 requires that a failed naming call leave the
artifacts unwritten so that a retry starts clean.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

__all__ = [
    "ThemeNamingError",
    "ThemeNamingInput",
    "ThemeNaming",
    "ThemeNameRecord",
    "ThemeNames",
    "PROMPT_FILENAME",
    "load_naming_prompt",
    "ctfidf",
    "select_representatives",
    "name_themes",
]


#: The naming prompt lives beside the identification prompts, loaded the same way they
#: are: ``uorca/identification/prompts/<name>.txt``.
PROMPT_FILENAME = "name_themes.txt"

#: A trailing bracketed qualifier. Sub-series of one super-series differ only by this.
_BRACKET = re.compile(r"\s*[\[\(][^\]\)]*[\]\)]\s*")


class ThemeNamingError(RuntimeError):
    """The naming call failed, or returned something unusable.

    Spec section 7: raising here means no artifacts are written, so a retry is clean.
    """


@dataclass(frozen=True)
class ThemeNamingInput:
    """What the model is shown about one theme (spec 5.6).

    A dataclass rather than a dict because this crosses a module boundary: the clustering
    step builds these and this module consumes them.
    """

    theme_id: int
    n: int
    keywords: list[str] = field(default_factory=list)
    representatives: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ThemeNaming:
    """What the model returned for one theme, validated and matched back to its id."""

    theme_id: int
    label: str
    description: str


class ThemeNameRecord(BaseModel):
    """One theme's name, as the model returns it."""

    id: int = Field(description="The theme id, echoed back unchanged.")
    label: str = Field(description="A 2-5 word Title Case label naming the biology.")
    description: str = Field(description="One sentence, at most 25 words.")


class ThemeNames(BaseModel):
    """The whole naming response: one record per theme, every theme named."""

    themes: list[ThemeNameRecord]


def load_naming_prompt() -> str:
    """Load the naming system prompt from ``uorca/identification/prompts/``.

    Raises:
        FileNotFoundError: The bundled prompt is missing. A missing bundled asset must
            raise rather than degrade into an unprompted call (CLAUDE.md, issue 9).
    """
    path = Path(__file__).resolve().parents[1] / "prompts" / PROMPT_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"Theme-naming prompt not found at {path}")
    return path.read_text(encoding="utf-8").strip()


# --------------------------------------------------------------------------------------
# c-TF-IDF — spec section 5.4
# --------------------------------------------------------------------------------------


def ctfidf(
    texts_by_theme: Mapping[int, Sequence[str]],
    all_texts: Sequence[str],
    *,
    top_n: int = 12,
    min_df: int = 3,
) -> dict[int, list[str]]:
    """Rank each theme's distinguishing terms by real class-based TF-IDF.

    The score is ``tf * log(1 + A/f)`` on L1-normalised class counts, with square-root
    damping of frequent words. ``A`` is the mean word count per class and ``f`` is the
    corpus frequency of the term across the classes.

    **Plain sklearn TF-IDF is not acceptable here and must not be substituted.** On this
    text it pins 87.1% of terms at maximum inverse document frequency, so the ranking
    collapses to raw frequency and every theme comes back labelled with GEO boilerplate
    (spec 5.4).

    The vocabulary is fitted on *all_texts* — the whole valid corpus, unthemed rows
    included — so ``min_df`` is judged against the run, not against one cluster.

    Args:
        texts_by_theme: Theme id -> the texts of that theme's datasets. Noise (theme
            ``-1``) is not a theme and must not be passed in.
        all_texts: Every text in the corpus, used to fit the vocabulary.
        top_n: How many terms to return per theme.
        min_df: Minimum number of corpus documents a term must appear in.

    Returns:
        Theme id -> its top terms, best first.

    Raises:
        ValueError: The mapping is empty, a theme has no texts, or the corpus is empty.
    """
    from sklearn.feature_extraction.text import CountVectorizer

    if not texts_by_theme:
        raise ValueError(
            "ctfidf needs at least one theme; an empty mapping means clustering produced "
            "nothing and the k >= 3 guard should have fired first."
        )
    empty = [theme_id for theme_id, texts in texts_by_theme.items() if len(texts) == 0]
    if empty:
        raise ValueError(f"Theme(s) {empty} have no texts; a theme is never empty.")
    if not all_texts:
        raise ValueError("ctfidf was given an empty corpus.")

    vectorizer = CountVectorizer(
        stop_words="english", min_df=min_df, ngram_range=(1, 2), max_features=100_000
    )
    vectorizer.fit(all_texts)
    vocabulary = np.asarray(vectorizer.get_feature_names_out())

    theme_ids = sorted(texts_by_theme)
    counts = np.vstack(
        [
            np.asarray(vectorizer.transform(texts_by_theme[theme_id]).sum(axis=0)).ravel()
            for theme_id in theme_ids
        ]
    ).astype(float)

    # tf: L1-normalised per class, square-root damped so that a merely frequent word
    # cannot dominate a distinctive one.
    tf = np.sqrt(counts / np.clip(counts.sum(axis=1, keepdims=True), 1, None))
    # idf: A is the mean word count per class, f the corpus frequency of the term.
    mean_words_per_class = counts.sum() / len(theme_ids)
    corpus_frequency = np.clip(counts.sum(axis=0), 1, None)
    idf = np.log(1 + mean_words_per_class / corpus_frequency)

    scores = tf * idf
    return {
        theme_id: vocabulary[np.argsort(-scores[index])[:top_n]].tolist()
        for index, theme_id in enumerate(theme_ids)
    }


# --------------------------------------------------------------------------------------
# Representative selection — spec section 5.5
# --------------------------------------------------------------------------------------


def _strip_title(title: str) -> str:
    return _BRACKET.sub(" ", str(title)).strip().casefold()


def select_representatives(
    theme_id: int,
    accessions: Sequence[str],
    titles: Sequence[str],
    labels: np.ndarray,
    probabilities: np.ndarray,
    n: int = 12,
) -> list[str]:
    """Pick 10-15 titles that illustrate one theme, ranked by cluster membership.

    Ranking is by the HDBSCAN ``probabilities_`` value, descending, with the accession as
    a deterministic tie-break. Titles are then deduplicated on the bracket-stripped
    title, so a super-series and its sub-series count once.

    Centroid-nearest selection was rejected in spec 5.5 and must not be reintroduced: it
    is biased — the top-8 internal cosine exceeds the whole-cluster cosine by 25-60%, so
    the sample looks far more coherent than the cluster is — and duplicate-contaminated:
    one observed theme's top 5 held the same study three times.

    Args:
        theme_id: Which theme to sample. Noise (``-1``) is not a theme.
        accessions: Row-aligned accessions.
        titles: Row-aligned titles.
        labels: Row-aligned HDBSCAN labels.
        probabilities: Row-aligned HDBSCAN ``probabilities_``.
        n: How many to return, capped by how many unique titles the theme has. Spec 5.5
            fixes the band at 10-15.

    Returns:
        Up to *n* original (unstripped) titles, strongest membership first.

    Raises:
        ValueError: *n* is outside the 10-15 band, the inputs are ragged, or the theme
            holds no datasets.
    """
    if not 10 <= n <= 15:
        raise ValueError(
            f"select_representatives takes 10 to 15 titles per theme (spec 5.5), got n={n}."
        )

    label_array = np.asarray(labels)
    probability_array = np.asarray(probabilities, dtype=float)
    lengths = {
        "accessions": len(accessions),
        "titles": len(titles),
        "labels": int(label_array.shape[0]),
        "probabilities": int(probability_array.shape[0]),
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(
            f"accessions, titles, labels and probabilities must all be the same length, "
            f"got {lengths}."
        )

    members = np.flatnonzero(label_array == theme_id)
    if members.size == 0:
        raise ValueError(
            f"Theme {theme_id} has no datasets; a theme with no members cannot be named."
        )

    ordered = sorted(
        members.tolist(),
        key=lambda index: (-probability_array[index], str(accessions[index])),
    )

    seen: set[str] = set()
    representatives: list[str] = []
    for index in ordered:
        title = str(titles[index]).strip()
        if not title:
            continue
        key = _strip_title(title)
        if key in seen:
            continue
        seen.add(key)
        representatives.append(title)
        if len(representatives) == n:
            break
    return representatives


# --------------------------------------------------------------------------------------
# Naming — spec section 5.6 and decision 10
# --------------------------------------------------------------------------------------


def _run_naming_agent(prompt: str, user_input: str, output_type: type) -> Any:
    """Make the structured call, in the same shape as the rest of the repo.

    Isolated in its own function so that the tests can replace it without touching
    pydantic-ai or the provider configuration.
    """
    from pydantic_ai import Agent

    from ...ai_provider import get_model

    agent = Agent(get_model(), output_type=output_type, system_prompt=prompt)
    return agent.run_sync(user_input).output


def name_themes(theme_inputs: Sequence[ThemeNamingInput]) -> list[ThemeNaming]:
    """Name every theme in one structured LLM call (spec 5.6, decision 10).

    **One call for the whole run, and every theme is named** — not the largest N. Ranking
    by size and naming only the top N would bury the useful themes: in the PPARgamma run
    the theme with the highest mean relevance was the second largest, not the largest.
    One call costs the same whether it names 20 themes or 40.

    Args:
        theme_inputs: One record per theme, carrying its c-TF-IDF terms, its size and its
            representative titles.

    Returns:
        One :class:`ThemeNaming` per input, in ascending theme id order.

    Raises:
        ValueError: *theme_inputs* is empty.
        ThemeNamingError: The call failed, a theme came back unnamed, an unknown id came
            back, or a label or description is blank. Spec section 7 keeps the artifacts
            unwritten in every one of these cases.
    """
    if not theme_inputs:
        raise ValueError(
            "name_themes was given no themes; the k >= 3 guard should have fired first."
        )

    prompt = load_naming_prompt()
    payload = [
        {
            "id": item.theme_id,
            "n": item.n,
            "keywords": list(item.keywords),
            "representatives": list(item.representatives),
        }
        for item in sorted(theme_inputs, key=lambda item: item.theme_id)
    ]
    user_input = f"Themes to name:\n{json.dumps(payload, indent=2)}"

    logger.info("Naming %d themes in one structured call", len(payload))
    try:
        response = _run_naming_agent(prompt, user_input, ThemeNames)
    except Exception as exc:  # re-raised, never swallowed
        raise ThemeNamingError(
            f"The theme-naming call failed: {type(exc).__name__}: {exc}. No theme-map "
            "artifacts were written, so a retry starts clean."
        ) from exc

    records = getattr(response, "themes", None)
    if not records:
        raise ThemeNamingError(
            "The theme-naming call returned no theme records at all."
        )

    by_id: dict[int, ThemeNameRecord] = {}
    for record in records:
        if record.id in by_id:
            raise ThemeNamingError(
                f"The theme-naming call returned theme id {record.id} twice."
            )
        by_id[record.id] = record

    expected = {item.theme_id for item in theme_inputs}
    unknown = sorted(set(by_id) - expected)
    if unknown:
        raise ThemeNamingError(
            f"The theme-naming call invented theme id(s) {unknown}; the supplied ids were "
            f"{sorted(expected)}."
        )
    missing = sorted(expected - set(by_id))
    if missing:
        raise ThemeNamingError(
            f"The theme-naming call left theme id(s) {missing} unnamed. Every theme must "
            "be named (spec decision 10)."
        )

    named: list[ThemeNaming] = []
    for theme_id in sorted(expected):
        record = by_id[theme_id]
        if not record.label.strip():
            raise ThemeNamingError(f"Theme {theme_id} came back with a blank label.")
        if not record.description.strip():
            raise ThemeNamingError(f"Theme {theme_id} came back with a blank description.")
        named.append(
            ThemeNaming(
                theme_id=theme_id,
                label=record.label.strip(),
                description=record.description.strip(),
            )
        )
    return named
