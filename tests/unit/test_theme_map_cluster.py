"""Unit tests for the theme-map numeric pipeline (spec sections 5.2, 5.3, 5.8, 7).

No API call is made anywhere in this file: :func:`cluster_embeddings` takes an embedding
matrix and never talks to a provider. The vectors here are synthetic.

Spec section 10.1 items 1 and 2 (the ``k >= 3`` guard and the minimum-size guard) are
covered by :func:`test_unstructured_vectors_raise_the_k_guard` and
:func:`test_fewer_than_fifty_datasets_raises_the_minimum_size_guard`.
"""

from __future__ import annotations

import numpy as np
import pytest

from uorca.identification.theme_map.cluster import (
    ClusterResult,
    CorpusTooSmallError,
    DegenerateClusteringError,
    ThemeMapClusterError,
    build_provenance,
    cluster_embeddings,
    make_clusterer,
)
from uorca.identification.theme_map.types import Provenance, ThemeMapConfig

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Synthetic corpora
# ---------------------------------------------------------------------------


def _planted(
    n_groups: int = 8,
    per_group: int = 40,
    dim: int = 256,
    spread: float = 0.06,
    seed: int = 0,
) -> np.ndarray:
    """Vectors with planted structure: tight groups around well-separated centres.

    The centres are drawn in a high-dimensional space, where random directions are very
    nearly orthogonal, so the groups are genuinely separable rather than separable only
    after projection.
    """
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(n_groups, dim))
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    blocks = [
        centres[g] + spread * rng.normal(size=(per_group, dim)) for g in range(n_groups)
    ]
    matrix = np.vstack(blocks)
    # Project back onto the unit sphere. Real embeddings from text-embedding-3-small are
    # unit vectors, and spec 5.2 step 1 asserts that, so a fixture that is not unit-norm
    # would test a corpus the pipeline never sees.
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix.astype(np.float32)


def _unstructured(n: int = 60, dim: int = 256, seed: int = 0) -> np.ndarray:
    """Vectors with no separable structure: isotropic noise on the unit sphere."""
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(n, dim))
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix.astype(np.float32)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
#
# The pipeline runs a full t-SNE, so the planted corpus is clustered once per module
# rather than once per test. CLAUDE.md asks for a unit suite under five seconds.


@pytest.fixture(scope="module")
def planted_embeddings() -> np.ndarray:
    return _planted()


@pytest.fixture(scope="module")
def planted_result(planted_embeddings: np.ndarray) -> ClusterResult:
    return cluster_embeddings(planted_embeddings)


@pytest.fixture(scope="module")
def unstructured_error() -> DegenerateClusteringError:
    with pytest.raises(DegenerateClusteringError) as excinfo:
        cluster_embeddings(_unstructured())
    return excinfo.value


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_planted_structure_produces_at_least_three_themes(
    planted_result: ClusterResult,
) -> None:
    assert isinstance(planted_result, ClusterResult)
    assert planted_result.k >= 3


def test_coords_have_shape_n_by_two(
    planted_embeddings: np.ndarray, planted_result: ClusterResult
) -> None:
    n = planted_embeddings.shape[0]

    assert planted_result.coords.shape == (n, 2)
    assert planted_result.labels.shape == (n,)
    assert planted_result.probabilities.shape == (n,)


def test_noise_labels_are_counted_as_noise_and_never_as_a_theme(
    planted_result: ClusterResult,
) -> None:
    result = planted_result
    labels = result.labels
    theme_ids = set(int(v) for v in np.unique(labels)) - {-1}

    assert -1 not in theme_ids
    assert result.k == len(theme_ids)
    assert result.noise_fraction == pytest.approx(float((labels == -1).mean()))
    # A corpus with planted structure must not be mostly noise.
    assert result.noise_fraction < 0.5


def test_explained_variance_is_reported_as_a_cumulative_fraction(
    planted_result: ClusterResult,
) -> None:
    assert 0.0 < planted_result.explained_variance_ratio_cumulative_at_50 <= 1.0


# ---------------------------------------------------------------------------
# Guards (spec section 7). Both raise; neither degrades.
# ---------------------------------------------------------------------------


def test_fewer_than_fifty_datasets_raises_the_minimum_size_guard() -> None:
    small = _planted(n_groups=3, per_group=16, dim=256)  # 48 datasets
    assert small.shape[0] < ThemeMapConfig().min_datasets

    with pytest.raises(CorpusTooSmallError) as excinfo:
        cluster_embeddings(small)

    message = str(excinfo.value)
    assert "48" in message
    assert "50" in message
    assert "45" in message  # three themes x min_cluster_size 15
    assert "70" in message  # ...and noise at 30.9-37.4% pushes the real floor there
    assert isinstance(excinfo.value, ThemeMapClusterError)


def test_unstructured_vectors_raise_the_k_guard(
    unstructured_error: DegenerateClusteringError,
) -> None:
    message = str(unstructured_error)
    assert "k=" in message
    # The message must report the noise fraction as well as k.
    assert "noise" in message.lower()
    assert "%" in message
    assert isinstance(unstructured_error, ThemeMapClusterError)


def test_k_guard_reports_the_actual_k_and_noise_fraction(
    unstructured_error: DegenerateClusteringError,
) -> None:
    assert unstructured_error.k < ThemeMapConfig().min_themes
    assert 0.0 <= unstructured_error.noise_fraction <= 1.0
    assert f"k={unstructured_error.k}" in str(unstructured_error)
    assert f"{unstructured_error.noise_fraction:.1%}" in str(unstructured_error)


def test_a_non_two_dimensional_matrix_raises() -> None:
    with pytest.raises(ThemeMapClusterError):
        cluster_embeddings(np.zeros((10, 10, 10), dtype=np.float32))


def test_non_unit_norm_rows_raise_rather_than_being_renormalised() -> None:
    """Spec 5.2 step 1 asserts unit norm; it does not renormalise.

    Renormalising would silently change the clustering, because every parameter in
    spec 5.2 was measured on the unit vectors that text-embedding-3-small returns.
    """
    rng = np.random.default_rng(0)
    scaled = rng.normal(size=(200, 60))
    scaled /= np.linalg.norm(scaled, axis=1, keepdims=True)
    scaled *= 3.0

    with pytest.raises(ThemeMapClusterError, match="not unit-norm"):
        cluster_embeddings(scaled)


def test_float32_rounding_of_unit_vectors_is_inside_the_tolerance() -> None:
    """Stored embeddings are float32, which rounds unit norm to within ~5.7e-4.

    That must pass. The tolerance exists to admit exactly this, and to reject a
    provider that returns genuinely unnormalised vectors.
    """
    rng = np.random.default_rng(1)
    vectors = rng.normal(size=(200, 60))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    rounded = vectors.astype(np.float32)

    deviation = float(np.abs(np.linalg.norm(rounded.astype(np.float64), axis=1) - 1.0).max())
    assert deviation < ThemeMapConfig().unit_norm_tolerance

    # It must not raise the unit-norm guard. Any other outcome is acceptable here:
    # this test is about the guard, not about what isotropic noise clusters into.
    try:
        cluster_embeddings(rounded)
    except ThemeMapClusterError as exc:
        assert "not unit-norm" not in str(exc)


def test_too_few_features_for_pca_raises_before_sklearn_does() -> None:
    rng = np.random.default_rng(0)
    too_narrow = rng.normal(size=(200, 20)).astype(np.float32)

    with pytest.raises(ThemeMapClusterError):
        cluster_embeddings(too_narrow)


# ---------------------------------------------------------------------------
# The fixed parameters (spec 5.2, 5.3)
# ---------------------------------------------------------------------------


def test_fitted_hdbscan_carries_leaf_selection_and_exact_min_span_tree(
    planted_result: ClusterResult,
) -> None:
    """Spec 5.3: ``leaf`` is the decision that dominates all others.

    Asserted on a real fitted estimator, not on a mock, and on the very factory that
    :func:`cluster_embeddings` uses, so the two cannot drift apart.
    """
    clusterer = make_clusterer(ThemeMapConfig())
    clusterer.fit(planted_result.coords)

    assert clusterer.cluster_selection_method == "leaf"
    assert clusterer.approx_min_span_tree is False
    assert clusterer.min_cluster_size == 15
    assert clusterer.min_samples == 3
    assert hasattr(clusterer, "labels_")


def test_clusterer_is_the_hdbscan_package_not_the_sklearn_one() -> None:
    """Spec section 8: the two disagree on ``min_samples``, so the choice is load-bearing."""
    clusterer = make_clusterer(ThemeMapConfig())

    assert type(clusterer).__module__.startswith("hdbscan")


# ---------------------------------------------------------------------------
# Provenance (spec 5.8)
# ---------------------------------------------------------------------------


def test_build_provenance_captures_a_version_for_each_of_the_three_libraries(
    planted_result: ClusterResult,
) -> None:
    provenance = build_provenance(planted_result, estimated_cost_usd=0.0038)

    assert isinstance(provenance, Provenance)
    for version in (
        provenance.sklearn_version,
        provenance.numpy_version,
        provenance.hdbscan_version,
    ):
        assert isinstance(version, str)
        assert version
        assert version[0].isdigit()


def test_build_provenance_records_omp_num_threads(
    planted_result: ClusterResult, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Spec 5.8: at n=2,065 the thread count changes the t-SNE output, so it is recorded."""
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    assert build_provenance(planted_result, estimated_cost_usd=0.0).omp_num_threads == "8"

    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    assert build_provenance(planted_result, estimated_cost_usd=0.0).omp_num_threads is None


def test_build_provenance_carries_the_cluster_numbers_and_the_config(
    planted_result: ClusterResult,
) -> None:
    result = planted_result
    provenance = build_provenance(result, estimated_cost_usd=0.0105)

    assert provenance.n_datasets == result.labels.shape[0]
    assert provenance.n_themes == result.k
    assert provenance.noise_fraction == pytest.approx(result.noise_fraction)
    assert provenance.estimated_cost_usd == pytest.approx(0.0105)
    assert provenance.seed == ThemeMapConfig().tsne_random_state
    assert provenance.embedding_model == ThemeMapConfig().embedding_model
    assert provenance.config == ThemeMapConfig()
    # An ISO-8601 timestamp, not a placeholder.
    assert provenance.timestamp
    assert "T" in provenance.timestamp
