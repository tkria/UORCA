"""Unit tests for theme-map artifact persistence.

Covers spec section 6 (the four artifact files, written as an atomic set) and the
round-trip fidelity requirements in spec section 10.1 items 7 and 8.

No API calls are made here: the artifact layer is pure filesystem I/O.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from uorca.identification.theme_map.artifacts import (
    ThemeMapArtifactError,
    ThemeMapArtifactsMissing,
    ThemeMapBuildFailed,
    artifacts_exist,
    read_artifacts,
    read_failure,
    write_artifacts,
    write_failure,
)
from uorca.identification.theme_map.types import (
    DatasetPoint,
    Provenance,
    Theme,
    ThemeMapConfig,
    ThemeMapResult,
)

pytestmark = pytest.mark.unit


def _make_result(config: ThemeMapConfig | None = None) -> ThemeMapResult:
    """A small but complete result, deliberately including the awkward values."""
    cfg = config or ThemeMapConfig()
    points = [
        DatasetPoint(
            geo_accession="GSE000001",
            x=1.5,
            y=-2.25,
            theme_id=0,
            theme_prob=0.91,
            relevance_score=7.25,
            query_sim=0.4123,
            query_sim_rank=1,
        ),
        DatasetPoint(
            geo_accession="GSE000002",
            x=-0.125,
            y=3.0,
            theme_id=-1,  # unthemed; must survive as -1
            theme_prob=0.0,
            relevance_score=None,  # unscored; must survive as None
            query_sim=0.2,
            query_sim_rank=2,
        ),
        DatasetPoint(
            geo_accession="GSE000003",
            x=0.0,
            y=0.0,
            theme_id=1,
            theme_prob=0.5,
            relevance_score=0.0,  # a real zero, distinct from None
            query_sim=-0.05,
            query_sim_rank=3,
        ),
    ]
    themes = [
        Theme(
            id=0,
            label="Adipocyte differentiation",
            description="Studies of adipogenesis in cultured cells.",
            n=1,
            keywords=["adipocyte", "differentiation", "pparg"],
            representatives=["Adipocyte differentiation time course"],
            mean_relevance=7.25,
            max_relevance=7.25,
        ),
        Theme(
            id=1,
            label="Hepatic steatosis",
            description="Liver fat accumulation models.",
            n=1,
            keywords=["liver", "steatosis"],
            representatives=["High fat diet liver RNA-seq"],
            mean_relevance=None,  # must survive as None
            max_relevance=None,
        ),
    ]
    provenance = Provenance(
        embedding_model=cfg.embedding_model,
        seed=cfg.tsne_random_state,
        omp_num_threads="8",
        sklearn_version="1.7.0",
        numpy_version="2.3.1",
        hdbscan_version="0.8.40",
        n_datasets=3,
        n_themes=2,
        noise_fraction=1.0 / 3.0,
        estimated_cost_usd=0.0038,
        timestamp="2026-09-16T12:00:00+00:00",
        config=cfg,
    )
    embeddings = np.arange(3 * 4, dtype=np.float32).reshape(3, 4) / 7.0
    return ThemeMapResult(
        points=points, themes=themes, embeddings=embeddings, provenance=provenance
    )


def test_round_trip_returns_equal_values(tmp_path: Path) -> None:
    result = _make_result()
    write_artifacts(tmp_path, result)

    loaded = read_artifacts(tmp_path)

    assert loaded.points == result.points
    assert loaded.themes == result.themes
    assert loaded.provenance == result.provenance
    assert loaded.provenance.config == result.provenance.config
    np.testing.assert_array_equal(loaded.embeddings, result.embeddings)
    assert loaded.embeddings.dtype == np.float32


def test_written_files_are_exactly_the_four_named_in_the_spec(tmp_path: Path) -> None:
    write_artifacts(tmp_path, _make_result())

    names = sorted(p.name for p in (tmp_path / "theme_map").iterdir())
    assert names == ["embeddings.npy", "provenance.json", "theme_map.csv", "themes.json"]
    # No temporary staging directory is left beside it.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["theme_map"]


def test_unthemed_theme_id_survives_as_minus_one(tmp_path: Path) -> None:
    write_artifacts(tmp_path, _make_result())

    loaded = read_artifacts(tmp_path)
    unthemed = [p for p in loaded.points if p.geo_accession == "GSE000002"][0]

    assert unthemed.theme_id == -1
    assert unthemed.theme_id is not None
    assert unthemed.theme_id != 1
    assert isinstance(unthemed.theme_id, int)
    assert unthemed.theme_prob == 0.0


def test_missing_relevance_score_survives_as_none(tmp_path: Path) -> None:
    write_artifacts(tmp_path, _make_result())

    loaded = read_artifacts(tmp_path)
    by_accession = {p.geo_accession: p for p in loaded.points}

    unscored = by_accession["GSE000002"]
    assert unscored.relevance_score is None
    # Explicitly not the two values a naive float cast would produce.
    assert not (isinstance(unscored.relevance_score, float))

    # A genuine zero must not be confused with a missing score.
    zero = by_accession["GSE000003"]
    assert zero.relevance_score == 0.0
    assert zero.relevance_score is not None

    # Theme-level optional relevance behaves the same way.
    themes = {t.id: t for t in loaded.themes}
    assert themes[1].mean_relevance is None
    assert themes[1].max_relevance is None
    assert themes[0].mean_relevance == 7.25


def test_partial_write_leaves_no_theme_map_directory(tmp_path: Path, monkeypatch) -> None:
    from uorca.identification.theme_map import artifacts as artifacts_module

    def boom(*_args, **_kwargs):
        raise RuntimeError("provenance write exploded")

    # provenance.json is written last, so earlier files are already on disk.
    monkeypatch.setattr(artifacts_module, "_write_provenance", boom)

    with pytest.raises(RuntimeError, match="provenance write exploded"):
        write_artifacts(tmp_path, _make_result())

    assert not (tmp_path / "theme_map").exists()
    assert list(tmp_path.iterdir()) == []
    assert artifacts_exist(tmp_path) is False


def test_write_artifacts_replaces_an_existing_directory(tmp_path: Path) -> None:
    write_artifacts(tmp_path, _make_result())
    write_failure(tmp_path, "an older, failed attempt")
    assert read_failure(tmp_path) == "an older, failed attempt"

    write_artifacts(tmp_path, _make_result())

    assert read_failure(tmp_path) is None
    assert artifacts_exist(tmp_path) is True
    assert sorted(p.name for p in tmp_path.iterdir()) == ["theme_map"]


def test_read_artifacts_raises_when_only_failed_txt_is_present(tmp_path: Path) -> None:
    message = "k = 2 with noise fraction 0.66; fewer than 3 themes"
    write_failure(tmp_path, message)

    with pytest.raises(ThemeMapBuildFailed) as excinfo:
        read_artifacts(tmp_path)

    assert message in str(excinfo.value)
    assert isinstance(excinfo.value, ThemeMapArtifactError)


def test_read_artifacts_raises_when_nothing_is_present(tmp_path: Path) -> None:
    with pytest.raises(ThemeMapArtifactsMissing) as excinfo:
        read_artifacts(tmp_path)

    assert "theme_map" in str(excinfo.value)
    assert isinstance(excinfo.value, ThemeMapArtifactError)


def test_read_failure_returns_none_when_absent(tmp_path: Path) -> None:
    assert read_failure(tmp_path) is None
    (tmp_path / "theme_map").mkdir()
    assert read_failure(tmp_path) is None


def test_artifacts_exist_is_false_for_absent_dir_and_for_failure_only(tmp_path: Path) -> None:
    assert artifacts_exist(tmp_path) is False

    write_failure(tmp_path, "build failed before any artifact was written")
    assert artifacts_exist(tmp_path) is False

    write_artifacts(tmp_path, _make_result())
    assert artifacts_exist(tmp_path) is True

    # An incomplete set is not a set.
    (tmp_path / "theme_map" / "themes.json").unlink()
    assert artifacts_exist(tmp_path) is False


def test_config_defaults_match_the_spec() -> None:
    cfg = ThemeMapConfig()

    assert cfg.embedding_model == "text-embedding-3-small"
    assert cfg.embedding_dim == 1536
    assert cfg.pca_components == 50
    assert cfg.pca_svd_solver == "covariance_eigh"
    assert cfg.tsne_perplexity == 30
    assert cfg.tsne_init == "pca"
    assert cfg.tsne_learning_rate == "auto"
    assert cfg.tsne_random_state == 42
    assert cfg.hdbscan_min_cluster_size == 15
    assert cfg.hdbscan_min_samples == 3
    assert cfg.hdbscan_cluster_selection_method == "leaf"
    assert cfg.hdbscan_approx_min_span_tree is False
    assert cfg.min_datasets == 50
    assert cfg.min_themes == 3
    assert cfg.unit_norm_tolerance == 1e-3


def test_build_theme_map_is_wired_and_fails_loudly_on_a_bogus_run() -> None:
    """Phase 3 replaced the ``NotImplementedError`` stub this test used to pin.

    The statement it makes is the same one: calling ``build_theme_map`` on a directory
    that is not an identification run raises rather than returning something empty. It
    now raises the real failure — there is no results CSV to build a population from —
    instead of the placeholder. Orchestration itself is covered by
    ``tests/unit/test_theme_map_build.py``.
    """
    from uorca.identification.theme_map import build_theme_map

    with pytest.raises(FileNotFoundError, match="Dataset_identification_result.csv"):
        build_theme_map(Path("/nonexistent/run"))
