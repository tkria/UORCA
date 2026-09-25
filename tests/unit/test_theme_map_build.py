"""Unit tests for ``build_theme_map`` and the ``__main__`` entry point (spec section 8).

These tests wire the real orchestration together over stubbed *spend* seams. Nothing
here calls OpenAI: :func:`uorca.identification.theme_map.embed.embed_texts`,
:func:`~uorca.identification.theme_map.embed.check_provider` and
:func:`uorca.identification.theme_map.naming._run_naming_agent` are the only three
places that could cost money, and every test replaces all three.

The real c-TF-IDF, the real representative selection, the real naming validation, the
real ``query_similarity`` arithmetic and the real artifact writer all run.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from uorca.identification.theme_map import (
    artifacts_exist,
    build_theme_map,
    read_artifacts,
    theme_map_dir,
)
from uorca.identification.theme_map import cluster as cluster_module
from uorca.identification.theme_map import embed as embed_module
from uorca.identification.theme_map import naming as naming_module
from uorca.identification.theme_map.cluster import ClusterResult, CorpusTooSmallError
from uorca.identification.theme_map.embed import (
    ThemeMapProviderError,
    ThemeMapQueryMissing,
)
from uorca.identification.theme_map.naming import ThemeNameRecord, ThemeNames

pytestmark = pytest.mark.unit


TOPICS = {
    0: "cardiac hypertrophy heart failure cardiomyocyte remodelling",
    1: "neural progenitor cortical neuron differentiation stem cell",
    2: "adipocyte browning thermogenesis white adipose tissue",
}

#: Which theme each of the 60 valid rows is planted in. The last five are noise.
PLAN: list[int] = [0] * 20 + [1] * 20 + [2] * 15 + [-1] * 5


def _write_run(
    run_dir: Path,
    *,
    n_valid: int = 60,
    n_invalid: int = 5,
    with_metadata: bool = True,
) -> Path:
    """Write a results CSV (and metadata) that looks like an identification run."""
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(n_valid):
        theme = PLAN[index % len(PLAN)]
        topic = TOPICS[theme if theme >= 0 else 0]
        # Rows 40-54 are deliberately unscored, so one whole theme has no relevance.
        unscored = 40 <= index < 55
        rows.append(
            {
                "GEO_Accession": f"GSE{100000 + index}",
                "Title": f"Study {index} of {topic}",
                "Summary": (
                    f"Purpose: we profiled {topic}. Methods: RNA sequencing of samples. "
                    f"Results: {topic} genes changed. Conclusions: {topic} matters."
                ),
                "Species": "Homo sapiens",
                "Valid": "Yes",
                "RelevanceScore": "" if unscored else round(1.0 + (index % 9), 1),
            }
        )
    for index in range(n_invalid):
        rows.append(
            {
                "GEO_Accession": f"GSE{900000 + index}",
                "Title": f"Rejected study {index} of microarray profiling",
                "Summary": "Microarray, not RNA-seq.",
                "Species": "Mus musculus",
                "Valid": "No",
                "RelevanceScore": "",
            }
        )
    pd.DataFrame(rows).to_csv(run_dir / "Dataset_identification_result.csv", index=False)

    if with_metadata:
        (run_dir / "identification_metadata.json").write_text(
            json.dumps({"input_query": "What drives cardiac hypertrophy?"}),
            encoding="utf-8",
        )
    return run_dir


def _fake_embeddings(texts: list[str]) -> np.ndarray:
    """Deterministic unit-ish vectors, one per text. Never touches the network."""
    matrix = np.zeros((len(texts), 4), dtype=np.float32)
    for row, text in enumerate(texts):
        seed = abs(hash(text)) % 9973
        rng = np.random.default_rng(seed)
        matrix[row] = rng.normal(size=4).astype(np.float32)
        if not matrix[row].any():  # pragma: no cover - vanishingly unlikely
            matrix[row, 0] = 1.0
    return matrix


def _fake_cluster(embeddings, *, config=None) -> ClusterResult:
    n = int(np.asarray(embeddings).shape[0])
    labels = np.asarray([PLAN[i % len(PLAN)] for i in range(n)], dtype=np.int_)
    probabilities = np.where(labels == -1, 0.0, 0.9).astype(np.float64)
    coords = np.column_stack(
        [np.arange(n, dtype=np.float32), np.arange(n, dtype=np.float32) * 0.5]
    ).astype(np.float32)
    return ClusterResult(
        coords=coords,
        labels=labels,
        probabilities=probabilities,
        k=3,
        noise_fraction=float((labels == -1).mean()),
        silhouette=0.4,
        explained_variance_ratio_cumulative_at_50=0.57,
    )


def _fake_naming(prompt: str, user_input: str, output_type: type):
    payload = json.loads(user_input.split("Themes to name:", 1)[1])
    return ThemeNames(
        themes=[
            ThemeNameRecord(
                id=item["id"],
                label=f"Theme {item['id']} Label",
                description=f"Description of theme {item['id']}.",
            )
            for item in payload
        ]
    )


@pytest.fixture
def stubbed(monkeypatch: pytest.MonkeyPatch) -> dict[str, list]:
    """Replace every seam that could spend money, and record the call order."""
    calls: dict[str, list] = {"order": [], "embed": [], "provider": []}

    def check_provider() -> None:
        calls["order"].append("check_provider")
        calls["provider"].append(True)

    def embed_texts(texts, *, config=None):
        calls["order"].append("embed_texts")
        calls["embed"].append(list(texts))
        return _fake_embeddings(list(texts))

    def cluster_embeddings(embeddings, *, config=None):
        calls["order"].append("cluster_embeddings")
        return _fake_cluster(embeddings, config=config)

    def run_naming_agent(prompt, user_input, output_type):
        calls["order"].append("name_themes")
        return _fake_naming(prompt, user_input, output_type)

    monkeypatch.setattr(embed_module, "check_provider", check_provider)
    monkeypatch.setattr(embed_module, "embed_texts", embed_texts)
    monkeypatch.setattr(cluster_module, "cluster_embeddings", cluster_embeddings)
    monkeypatch.setattr(naming_module, "_run_naming_agent", run_naming_agent)
    return calls


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_build_theme_map_writes_a_complete_artifact_set(
    tmp_path: Path, stubbed: dict[str, list]
) -> None:
    run_dir = _write_run(tmp_path / "run")

    result = build_theme_map(run_dir)

    assert artifacts_exist(run_dir)
    assert len(result.points) == 60
    assert result.embeddings.shape[0] == 60
    assert result.provenance.n_datasets == 60
    assert result.provenance.n_themes == 3
    assert result.provenance.estimated_cost_usd > 0

    # Round-trips through disk unchanged.
    reloaded = read_artifacts(run_dir)
    assert [p.geo_accession for p in reloaded.points] == [
        p.geo_accession for p in result.points
    ]
    assert [t.id for t in reloaded.themes] == [0, 1, 2]


@pytest.mark.unit
def test_build_theme_map_checks_the_provider_before_it_embeds(
    tmp_path: Path, stubbed: dict[str, list]
) -> None:
    build_theme_map(_write_run(tmp_path / "run"))
    order = stubbed["order"]
    assert order.index("check_provider") < order.index("embed_texts")


@pytest.mark.unit
def test_build_theme_map_keeps_points_row_aligned_with_embeddings(
    tmp_path: Path, stubbed: dict[str, list]
) -> None:
    run_dir = _write_run(tmp_path / "run")
    result = build_theme_map(run_dir)

    accessions = [p.geo_accession for p in result.points]
    assert accessions == [f"GSE{100000 + i}" for i in range(60)]
    # The invalid rows never reach the map.
    assert not [a for a in accessions if a.startswith("GSE9")]

    corpus = stubbed["embed"][0]
    expected = _fake_embeddings(corpus)
    np.testing.assert_allclose(result.embeddings, expected)


@pytest.mark.unit
def test_unthemed_points_keep_theme_minus_one_and_zero_probability(
    tmp_path: Path, stubbed: dict[str, list]
) -> None:
    run_dir = _write_run(tmp_path / "run")
    build_theme_map(run_dir)
    # Read back from disk: -1 must survive the round trip as unthemed (spec 10.1.8).
    result = read_artifacts(run_dir)
    unthemed = [p for p in result.points if p.theme_id == -1]
    assert unthemed, "the planted corpus has five unthemed datasets"
    assert all(p.theme_prob == 0.0 for p in unthemed)
    assert {t.id for t in result.themes} == {0, 1, 2}


@pytest.mark.unit
def test_theme_relevance_is_none_when_no_member_was_scored(
    tmp_path: Path, stubbed: dict[str, list]
) -> None:
    result = build_theme_map(_write_run(tmp_path / "run"))
    by_id = {theme.id: theme for theme in result.themes}

    # Rows 40-54 are theme 2 and all unscored.
    assert by_id[2].mean_relevance is None
    assert by_id[2].max_relevance is None
    # Theme 0 is fully scored.
    assert by_id[0].mean_relevance is not None
    assert by_id[0].max_relevance is not None
    assert by_id[0].max_relevance >= by_id[0].mean_relevance


@pytest.mark.unit
def test_every_theme_carries_keywords_and_representatives(
    tmp_path: Path, stubbed: dict[str, list]
) -> None:
    result = build_theme_map(_write_run(tmp_path / "run"))
    for theme in result.themes:
        assert theme.keywords, f"theme {theme.id} has no c-TF-IDF keywords"
        assert theme.representatives, f"theme {theme.id} has no representative titles"
        assert theme.label == f"Theme {theme.id} Label"
        assert theme.n == sum(1 for p in result.points if p.theme_id == theme.id)


@pytest.mark.unit
def test_query_similarity_is_computed_and_ranked(
    tmp_path: Path, stubbed: dict[str, list]
) -> None:
    result = build_theme_map(_write_run(tmp_path / "run"))
    ranks = sorted(p.query_sim_rank for p in result.points)
    assert ranks == list(range(1, 61))
    # The query is embedded through the same seam, as one extra call.
    assert len(stubbed["embed"]) == 2
    assert stubbed["embed"][1] == ["What drives cardiac hypertrophy?"]


# ---------------------------------------------------------------------------
# Failure handling — spec section 7. Every one of these raises and writes nothing.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_missing_results_csv_raises(tmp_path: Path, stubbed: dict[str, list]) -> None:
    run_dir = tmp_path / "empty"
    run_dir.mkdir()
    with pytest.raises(FileNotFoundError) as excinfo:
        build_theme_map(run_dir)
    assert "Dataset_identification_result.csv" in str(excinfo.value)
    assert not theme_map_dir(run_dir).exists()


@pytest.mark.unit
def test_corpus_below_the_minimum_raises_before_any_spend(
    tmp_path: Path, stubbed: dict[str, list]
) -> None:
    run_dir = _write_run(tmp_path / "small", n_valid=40)
    with pytest.raises(CorpusTooSmallError):
        build_theme_map(run_dir)
    assert stubbed["embed"] == [], "nothing may be embedded once the guard has fired"
    assert not theme_map_dir(run_dir).exists()


@pytest.mark.unit
def test_provider_guard_raises_before_any_spend(
    tmp_path: Path, stubbed: dict[str, list], monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse() -> None:
        raise ThemeMapProviderError("provider is bedrock")

    monkeypatch.setattr(embed_module, "check_provider", refuse)
    run_dir = _write_run(tmp_path / "run")
    with pytest.raises(ThemeMapProviderError):
        build_theme_map(run_dir)
    assert stubbed["embed"] == []
    assert not theme_map_dir(run_dir).exists()


@pytest.mark.unit
def test_missing_query_raises_before_any_spend(
    tmp_path: Path, stubbed: dict[str, list]
) -> None:
    """The query is never hardcoded (spec 5.7), and a run without one costs nothing."""
    run_dir = _write_run(tmp_path / "run", with_metadata=False)
    with pytest.raises(ThemeMapQueryMissing):
        build_theme_map(run_dir)
    assert stubbed["embed"] == []
    assert not theme_map_dir(run_dir).exists()


@pytest.mark.unit
def test_a_failed_naming_call_writes_no_artifacts(
    tmp_path: Path, stubbed: dict[str, list], monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(prompt, user_input, output_type):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(naming_module, "_run_naming_agent", explode)
    run_dir = _write_run(tmp_path / "run")
    with pytest.raises(naming_module.ThemeNamingError):
        build_theme_map(run_dir)
    assert not theme_map_dir(run_dir).exists()


@pytest.mark.unit
def test_a_degenerate_clustering_run_writes_no_artifacts(
    tmp_path: Path, stubbed: dict[str, list], monkeypatch: pytest.MonkeyPatch
) -> None:
    def degenerate(embeddings, *, config=None):
        raise cluster_module.DegenerateClusteringError(
            "k=1", k=1, noise_fraction=0.9
        )

    monkeypatch.setattr(cluster_module, "cluster_embeddings", degenerate)
    run_dir = _write_run(tmp_path / "run")
    with pytest.raises(cluster_module.DegenerateClusteringError):
        build_theme_map(run_dir)
    assert not theme_map_dir(run_dir).exists()


@pytest.mark.unit
def test_a_previously_published_map_survives_a_failed_rebuild(
    tmp_path: Path, stubbed: dict[str, list], monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = _write_run(tmp_path / "run")
    first = build_theme_map(run_dir)

    def explode(prompt, user_input, output_type):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(naming_module, "_run_naming_agent", explode)
    with pytest.raises(naming_module.ThemeNamingError):
        build_theme_map(run_dir)

    assert artifacts_exist(run_dir)
    assert [p.geo_accession for p in read_artifacts(run_dir).points] == [
        p.geo_accession for p in first.points
    ]


# ---------------------------------------------------------------------------
# The dry run and the __main__ entry point — spec section 8
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_estimate_theme_map_build_counts_datasets_and_tokens(
    tmp_path: Path, stubbed: dict[str, list]
) -> None:
    from uorca.identification.theme_map import estimate_theme_map_build

    estimate = estimate_theme_map_build(_write_run(tmp_path / "run"))
    assert estimate.n_datasets == 60
    assert estimate.n_tokens > 0
    assert estimate.cost_usd > 0
    assert stubbed["embed"] == [], "an estimate never embeds anything"


@pytest.mark.unit
def test_main_dry_run_prints_the_estimate_and_spends_nothing(
    tmp_path: Path, stubbed: dict[str, list], capsys: pytest.CaptureFixture[str]
) -> None:
    from uorca.identification.theme_map.__main__ import main

    run_dir = _write_run(tmp_path / "run")
    exit_code = main(["--run_dir", str(run_dir), "--dry-run"])

    captured = capsys.readouterr().out
    assert exit_code == 0
    assert "60" in captured
    assert "$" in captured
    assert stubbed["embed"] == []
    assert not theme_map_dir(run_dir).exists()


@pytest.mark.unit
def test_main_builds_the_map_when_not_a_dry_run(
    tmp_path: Path, stubbed: dict[str, list]
) -> None:
    from uorca.identification.theme_map.__main__ import main

    run_dir = _write_run(tmp_path / "run")
    assert main(["--run_dir", str(run_dir)]) == 0
    assert artifacts_exist(run_dir)


@pytest.mark.unit
def test_main_does_not_swallow_a_build_failure(
    tmp_path: Path, stubbed: dict[str, list], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The terminal entry point raises; only the identification call site catches."""
    from uorca.identification.theme_map.__main__ import main

    def refuse() -> None:
        raise ThemeMapProviderError("provider is bedrock")

    monkeypatch.setattr(embed_module, "check_provider", refuse)
    run_dir = _write_run(tmp_path / "run")
    with pytest.raises(ThemeMapProviderError):
        main(["--run_dir", str(run_dir)])
    assert not theme_map_dir(run_dir).exists()
