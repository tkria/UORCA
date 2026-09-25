"""Unit tests for theme keywords, representatives and the single naming call.

Covers spec sections 5.4 (real c-TF-IDF), 5.5 (representative selection), 5.6 and
decision 10 (one structured call, every theme named) and section 7 (a failed naming call
raises so no artifacts are written).

No test here makes a real API call: :func:`name_themes` reaches the model through
``naming._run_naming_agent``, which every test replaces with a stub.
"""

from __future__ import annotations

import numpy as np
import pytest

from uorca.identification.theme_map import naming as naming_module
from uorca.identification.theme_map.naming import (
    ThemeNaming,
    ThemeNamingError,
    ThemeNamingInput,
    ctfidf,
    name_themes,
    select_representatives,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------------------
# c-TF-IDF — spec section 5.4
# --------------------------------------------------------------------------------------


BOILERPLATE = "Expression profiling by high throughput sequencing of samples."


def _themed_corpus() -> tuple[dict[int, list[str]], list[str]]:
    """Three themes. One planted term per theme, one boilerplate phrase in all of them."""
    planted = {0: "cardiomyopathy", 1: "melanoma", 2: "nephropathy"}
    texts_by_theme: dict[int, list[str]] = {}
    for theme_id, term in planted.items():
        texts_by_theme[theme_id] = [
            f"{BOILERPLATE} Study {index} of human {term} tissue." for index in range(8)
        ]
    all_texts = [text for texts in texts_by_theme.values() for text in texts]
    return texts_by_theme, all_texts


def test_ctfidf_ranks_a_planted_term_above_boilerplate() -> None:
    """Spec 10.1 item 3. Plain sklearn TF-IDF cannot do this on GEO text (spec 5.4)."""
    texts_by_theme, all_texts = _themed_corpus()

    terms = ctfidf(texts_by_theme, all_texts)

    assert set(terms) == {0, 1, 2}
    for theme_id, planted in ((0, "cardiomyopathy"), (1, "melanoma"), (2, "nephropathy")):
        ranking = terms[theme_id]
        assert planted in ranking, f"theme {theme_id} ranking: {ranking}"
        boilerplate_positions = [
            index
            for index, term in enumerate(ranking)
            if term in {"expression", "profiling", "sequencing", "expression profiling"}
        ]
        planted_position = ranking.index(planted)
        for position in boilerplate_positions:
            assert planted_position < position


def test_ctfidf_returns_the_requested_number_of_terms() -> None:
    texts_by_theme, all_texts = _themed_corpus()

    terms = ctfidf(texts_by_theme, all_texts, top_n=5)

    assert all(len(values) == 5 for values in terms.values())


def test_ctfidf_raises_on_an_empty_theme_mapping() -> None:
    with pytest.raises(ValueError, match="at least one theme"):
        ctfidf({}, ["some text"])


def test_ctfidf_raises_on_an_empty_theme() -> None:
    _, all_texts = _themed_corpus()

    with pytest.raises(ValueError, match="no texts"):
        ctfidf({0: [], 1: all_texts[:8]}, all_texts)


# --------------------------------------------------------------------------------------
# Representative selection — spec section 5.5
# --------------------------------------------------------------------------------------


def test_select_representatives_drops_a_repeated_bracket_stripped_title() -> None:
    """Spec 10.1 item 4. One observed theme's top 5 held the same study three times."""
    accessions = ["GSE1", "GSE2", "GSE3", "GSE4"]
    titles = [
        "Adipocyte atlas [RNA-seq]",
        "Adipocyte atlas [ATAC-seq]",
        "Adipocyte atlas (scRNA-seq)",
        "A quite different study",
    ]
    labels = np.array([0, 0, 0, 0])
    probabilities = np.array([0.9, 0.8, 0.7, 0.6])

    reps = select_representatives(0, accessions, titles, labels, probabilities, n=10)

    assert reps == ["Adipocyte atlas [RNA-seq]", "A quite different study"]


def test_select_representatives_returns_between_ten_and_fifteen_when_large_enough() -> None:
    size = 30
    accessions = [f"GSE{index}" for index in range(size)]
    titles = [f"Distinct study number {index}" for index in range(size)]
    labels = np.zeros(size, dtype=int)
    probabilities = np.linspace(0.1, 1.0, size)

    reps = select_representatives(0, accessions, titles, labels, probabilities)

    assert 10 <= len(reps) <= 15
    assert len(reps) == 12  # the default n
    # Highest probability first.
    assert reps[0] == f"Distinct study number {size - 1}"


def test_select_representatives_ignores_other_themes_and_noise() -> None:
    accessions = ["GSE1", "GSE2", "GSE3"]
    titles = ["In theme", "Other theme", "Noise"]
    labels = np.array([0, 1, -1])
    probabilities = np.array([0.5, 0.9, 0.0])

    assert select_representatives(0, accessions, titles, labels, probabilities, n=10) == [
        "In theme"
    ]


def test_select_representatives_rejects_an_n_outside_the_spec_band() -> None:
    accessions = ["GSE1"]
    titles = ["Only"]
    labels = np.array([0])
    probabilities = np.array([1.0])

    for bad_n in (8, 16):
        with pytest.raises(ValueError, match="10"):
            select_representatives(0, accessions, titles, labels, probabilities, n=bad_n)


def test_select_representatives_raises_on_ragged_input() -> None:
    with pytest.raises(ValueError, match="same length"):
        select_representatives(
            0, ["GSE1", "GSE2"], ["Only one title"], np.array([0, 0]), np.array([1.0, 1.0])
        )


def test_select_representatives_raises_when_the_theme_is_empty() -> None:
    with pytest.raises(ValueError, match="no datasets"):
        select_representatives(
            7, ["GSE1"], ["Only"], np.array([0]), np.array([1.0]), n=10
        )


# --------------------------------------------------------------------------------------
# name_themes — spec section 5.6, decision 10, section 7
# --------------------------------------------------------------------------------------


def _inputs(count: int = 3) -> list[ThemeNamingInput]:
    return [
        ThemeNamingInput(
            theme_id=index,
            n=20 + index,
            keywords=[f"term{index}a", f"term{index}b"],
            representatives=[f"Study {index}-{jndex}" for jndex in range(3)],
        )
        for index in range(count)
    ]


class _Recorder:
    def __init__(self, response) -> None:
        self.response = response
        self.calls = 0

    def __call__(self, prompt: str, user_input: str, output_type):
        self.calls += 1
        self.prompt = prompt
        self.user_input = user_input
        return self.response


def test_name_themes_makes_one_call_and_names_every_theme(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spec decision 10: every theme is named, not only the largest N."""
    theme_inputs = _inputs(3)
    response = naming_module.ThemeNames(
        themes=[
            naming_module.ThemeNameRecord(
                id=item.theme_id,
                label=f"Label {item.theme_id}",
                description=f"Description {item.theme_id}",
            )
            for item in theme_inputs
        ]
    )
    recorder = _Recorder(response)
    monkeypatch.setattr(naming_module, "_run_naming_agent", recorder)

    named = name_themes(theme_inputs)

    assert recorder.calls == 1
    assert [item.theme_id for item in named] == [0, 1, 2]
    assert named[1] == ThemeNaming(theme_id=1, label="Label 1", description="Description 1")
    # The keywords and representatives are what the model is shown (spec 5.6).
    assert "term1a" in recorder.user_input
    assert "Study 1-0" in recorder.user_input


def test_name_themes_raises_when_the_model_call_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spec section 7: artifacts stay unwritten so a retry is clean."""

    def explode(prompt: str, user_input: str, output_type):
        raise RuntimeError("upstream 500")

    monkeypatch.setattr(naming_module, "_run_naming_agent", explode)

    with pytest.raises(ThemeNamingError) as excinfo:
        name_themes(_inputs(3))

    assert "upstream 500" in str(excinfo.value)


def test_name_themes_raises_when_a_theme_is_unnamed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    theme_inputs = _inputs(3)
    response = naming_module.ThemeNames(
        themes=[
            naming_module.ThemeNameRecord(id=0, label="Label 0", description="d0"),
            naming_module.ThemeNameRecord(id=1, label="Label 1", description="d1"),
        ]
    )
    monkeypatch.setattr(naming_module, "_run_naming_agent", _Recorder(response))

    with pytest.raises(ThemeNamingError, match="2"):
        name_themes(theme_inputs)


def test_name_themes_raises_on_a_blank_label(monkeypatch: pytest.MonkeyPatch) -> None:
    theme_inputs = _inputs(3)
    response = naming_module.ThemeNames(
        themes=[
            naming_module.ThemeNameRecord(
                id=item.theme_id, label="   " if item.theme_id == 1 else "ok",
                description="d",
            )
            for item in theme_inputs
        ]
    )
    monkeypatch.setattr(naming_module, "_run_naming_agent", _Recorder(response))

    with pytest.raises(ThemeNamingError, match="label"):
        name_themes(theme_inputs)


def test_name_themes_raises_on_an_empty_input() -> None:
    with pytest.raises(ValueError, match="no themes"):
        name_themes([])


def test_naming_prompt_file_exists_and_is_loadable() -> None:
    prompt = naming_module.load_naming_prompt()

    assert prompt.strip()
    assert "label" in prompt.lower()
