"""The theme-map numeric pipeline: PCA -> t-SNE -> HDBSCAN (spec sections 5.2, 5.3, 7).

**Coordinates and labels are persisted as artifacts. Regeneration from the seed is NOT
promised.** Spec section 5.8 measured this directly: at n=793 and n=807 the t-SNE output
is bit-identical across ``OMP_NUM_THREADS`` 8 versus 1, but at n=2,065 it is not --
ARI 0.275, 15-nearest-neighbour overlap 0.526, and k moved from 40 to 38. An identical
thread count always reproduces exactly, which is why :func:`build_provenance` records
``OMP_NUM_THREADS``. Nothing in this module may claim that a seed alone reproduces a map;
read ``theme_map/`` back through :mod:`.artifacts` instead of recomputing it.

Every numeric parameter here lives on :class:`~.types.ThemeMapConfig` and is fixed by
spec section 5.2, with the reason for each one quoted beside its use below. They are not
defaults anyone may tune, and none of them may be made adaptive: a configuration fitted
on two corpora already collapsed to a 43.9% catch-all cluster on a held-out third.

The clustering step is the one place in this feature where a wrong answer is silent. A
degenerate run imports cleanly, raises nothing, and returns ``labels_`` that are all
``-1``. The two guards in :func:`cluster_embeddings` are the only defence against that,
so both raise and neither degrades.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version as _package_version

import hdbscan
import numpy as np
import sklearn
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from .types import Float32Array, Provenance, ThemeMapConfig

__all__ = [
    "ThemeMapClusterError",
    "CorpusTooSmallError",
    "DegenerateClusteringError",
    "ClusterResult",
    "make_clusterer",
    "cluster_embeddings",
    "build_provenance",
]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ThemeMapClusterError(RuntimeError):
    """Base class for every failure of the numeric pipeline (spec section 7)."""


class CorpusTooSmallError(ThemeMapClusterError):
    """Raised below ``config.min_datasets`` valid datasets (spec section 7)."""


class DegenerateClusteringError(ThemeMapClusterError):
    """Raised when HDBSCAN found fewer than ``config.min_themes`` themes.

    ``k`` and ``noise_fraction`` are carried on the exception as well as in the message
    so that a caller can report them without re-parsing text.
    """

    def __init__(self, message: str, *, k: int, noise_fraction: float) -> None:
        super().__init__(message)
        self.k = k
        self.noise_fraction = noise_fraction


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class ClusterResult:
    """What one clustering run produced.

    ``eq`` is disabled for the same reason as on
    :class:`~.types.ThemeMapResult`: a generated ``__eq__`` over a numpy field returns an
    array rather than a bool. Compare the fields instead.
    """

    coords: Float32Array
    """t-SNE coordinates, n x 2. This is the display space *and* the clustering space."""

    labels: NDArray[np.int_]
    """HDBSCAN labels, length n. ``-1`` is unthemed and is never a theme id."""

    probabilities: NDArray[np.float64]
    """HDBSCAN ``probabilities_``, length n. ``0.0`` for unthemed points."""

    k: int
    """Number of themes, excluding the ``-1`` noise label."""

    noise_fraction: float
    """Fraction of datasets labelled ``-1``. Measured at 30.9-37.4% on real corpora."""

    silhouette: float
    """Silhouette over the themed points only, in the 2-D space they were clustered in."""

    explained_variance_ratio_cumulative_at_50: float
    """Cumulative PCA variance retained. 55.2-60.0% on real embeddings (spec 5.3)."""


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def make_clusterer(config: ThemeMapConfig) -> hdbscan.HDBSCAN:
    """Construct the validated HDBSCAN estimator.

    This is deliberately the ``hdbscan`` package and **not**
    ``sklearn.cluster.HDBSCAN``. Spec section 8: sklearn's ``min_samples`` counts the
    point itself, so ``min_samples=3`` there is ``min_samples=2`` here, and every
    measured number in spec section 5.2 came from this package.

    It is a separate factory so that the estimator :func:`cluster_embeddings` fits is the
    same object a test can inspect; the two cannot drift apart.
    """
    return hdbscan.HDBSCAN(
        # Spec 5.3: fixed, not scaled by n. A scaled value makes two runs incomparable --
        # a 15-dataset theme would be named at n=793 and grey noise at n=2,065.
        min_cluster_size=config.hdbscan_min_cluster_size,
        # Spec section 8: the `hdbscan` package's meaning of min_samples, which excludes
        # the point itself. Do not carry this value to sklearn.cluster.HDBSCAN.
        min_samples=config.hdbscan_min_samples,
        metric="euclidean",
        # Spec 5.3: THE decision that dominates all others. Across an 84-point grid on
        # three corpora, "leaf" produced a catch-all cluster in 0% of configurations and
        # "eom" in 39-61%. "eom" is also unstable: one step in min_samples swung the
        # largest cluster from 43.9% to 12.5%.
        cluster_selection_method=config.hdbscan_cluster_selection_method,
        # Spec 5.3: the hdbscan default is True, which is approximate.
        approx_min_span_tree=config.hdbscan_approx_min_span_tree,
    )


def cluster_embeddings(
    embeddings: Float32Array | NDArray[np.floating],
    *,
    config: ThemeMapConfig | None = None,
) -> ClusterResult:
    """Assert unit norm -> PCA(50) -> t-SNE(2) -> HDBSCAN on the two t-SNE coordinates.

    Args:
        embeddings: n x ``config.embedding_dim`` matrix, one row per valid dataset.
        config: the fixed configuration of spec 5.2. Defaults to
            :class:`~.types.ThemeMapConfig`.

    Returns:
        A :class:`ClusterResult`, row-aligned with ``embeddings``.

    Raises:
        CorpusTooSmallError: fewer than ``config.min_datasets`` datasets.
        DegenerateClusteringError: fewer than ``config.min_themes`` themes.
        ThemeMapClusterError: the input is not a usable 2-D float matrix, or its
            rows are not unit-norm within ``config.unit_norm_tolerance``.
    """
    config = config or ThemeMapConfig()

    matrix = np.asarray(embeddings)
    if matrix.ndim != 2:
        raise ThemeMapClusterError(
            "Theme map clustering needs a 2-D embedding matrix "
            f"(n_datasets x n_features); got an array with shape {matrix.shape}."
        )

    n_datasets, n_features = matrix.shape

    # --- Guard 1: corpus size (spec section 7) -----------------------------------
    if n_datasets < config.min_datasets:
        raise CorpusTooSmallError(
            f"Theme map corpus is too small: {n_datasets} valid datasets, minimum "
            f"{config.min_datasets}. Three themes at min_cluster_size="
            f"{config.hdbscan_min_cluster_size} need "
            f"{config.min_themes * config.hdbscan_min_cluster_size} clustered datasets, "
            "and noise runs at 30.9-37.4% on the validated corpora, so a run needs "
            "roughly 70 valid datasets before three themes are arithmetically possible. "
            "Widen the identification query or lower the validity filter; do not lower "
            "this guard."
        )

    if n_features < config.pca_components:
        raise ThemeMapClusterError(
            f"Theme map clustering needs at least {config.pca_components} features per "
            f"dataset for PCA(n_components={config.pca_components}); got {n_features}. "
            f"Embeddings from {config.embedding_model} have {config.embedding_dim}."
        )

    # --- 1. Assert unit norm (spec 5.2, corrected 2026-09-17) --------------------
    # float64 throughout: the validated run cast the stored float32 matrix up before PCA.
    #
    # This step ASSERTS unit norm; it does not renormalise. Measured 2026-09-16 on the
    # four real corpora: the script that produced the spec 5.2 table
    # (docs/theme_map_validation/real_embeddings/exp_d_final.py) never renormalised --
    # it fed PCA the raw float32-to-float64 matrix, although its own docstring claimed
    # otherwise. text-embedding-3-small returns unit vectors, and float32 storage rounds
    # them to within 5.7e-4, so renormalising is a perturbation at the 1e-4 level that
    # t-SNE amplifies: k fell by 1-3 and noise rose 3.5-4.9 points against the spec 5.2
    # table. Asserting instead reproduces all four validated rows exactly and still
    # catches a provider that ever returns non-unit vectors.
    working = matrix.astype(np.float64, copy=True)
    norms = np.linalg.norm(working, axis=1)
    deviation = float(np.abs(norms - 1.0).max())
    if deviation > config.unit_norm_tolerance:
        worst = int(np.abs(norms - 1.0).argmax())
        raise ThemeMapClusterError(
            f"Embedding rows are not unit-norm: the worst row ({worst}) has norm "
            f"{norms[worst]:.6f}, which deviates by {deviation:.2e} from 1.0 and exceeds "
            f"the tolerance {config.unit_norm_tolerance:.0e}. Every parameter in spec 5.2 "
            f"was measured on unit vectors from {config.embedding_model}. Renormalising "
            "here would silently change the clustering, so fix the embedding step "
            "instead."
        )

    # --- 2. PCA (spec 5.2) --------------------------------------------------------
    pca = PCA(
        # Spec 5.3: retains 55.2-60.0% of variance on real embeddings. The TF-IDF
        # surrogate reported 12.5-19.9% and was badly wrong.
        n_components=config.pca_components,
        # Spec 5.3: PCA(n_components=50) on 1536-dimension input auto-selects the
        # randomized solver, which is seeded but not exact. This one is exact.
        svd_solver=config.pca_svd_solver,
        random_state=config.tsne_random_state,
    )
    components = pca.fit_transform(working)
    explained = float(np.sum(pca.explained_variance_ratio_))

    # --- 3. t-SNE (spec 5.2) ------------------------------------------------------
    tsne = TSNE(
        n_components=2,
        # Spec 5.3: fixed, not scaled. The earlier min(30, max(5, n/100)) rule could only
        # ever fall below the sklearn default, and cost 0.53-0.60 of 15-NN overlap.
        perplexity=config.tsne_perplexity,
        init=config.tsne_init,
        # sklearn's type stub declares learning_rate as `str`, but the runtime signature
        # is `float | "auto"` and its own parameter validation accepts both. Spec 5.2
        # fixes "auto"; ThemeMapConfig types the field `str | float` to match the
        # runtime. The stub is the narrow thing here, so the suppression sits on it.
        learning_rate=config.tsne_learning_rate,  # pyright: ignore[reportArgumentType]
        # Spec 5.8: the seed does not by itself promise reproduction across thread counts.
        random_state=config.tsne_random_state,
        # Pinned to the value the validation run used, which is also the sklearn 1.7
        # default. sklearn has renamed and re-defaulted t-SNE arguments before (init and
        # learning_rate in 1.2, n_iter -> max_iter in 1.5), so it is stated, not assumed.
        max_iter=1000,
    )
    coords = np.asarray(tsne.fit_transform(components), dtype=np.float64)

    # --- 4. HDBSCAN, on the two t-SNE coordinates (spec 5.2) ----------------------
    # Spec 5.3: cluster in the 2-D display space. Clustering in 50 dimensions gives ARI
    # 0.075-0.109 against the 2-D partition, so colour and position would encode nearly
    # disjoint information, which a user reads as a broken feature.
    clusterer = make_clusterer(config)
    clusterer.fit(coords)
    labels = np.asarray(clusterer.labels_, dtype=np.int_)
    probabilities = np.asarray(clusterer.probabilities_, dtype=np.float64)

    theme_ids = sorted(set(int(v) for v in np.unique(labels)) - {-1})
    k = len(theme_ids)
    noise_fraction = float((labels == -1).mean())

    # --- Guard 2: k >= min_themes (spec section 7) -------------------------------
    # This is the only defence against a silent wrong answer: a degenerate run imports
    # cleanly, raises nothing, and returns labels_ that are all -1.
    if k < config.min_themes:
        raise DegenerateClusteringError(
            f"Theme map clustering was degenerate: k={k} themes (minimum "
            f"{config.min_themes}) over {n_datasets} datasets, with "
            f"{noise_fraction:.1%} noise. The validated corpora gave k=20-40 at "
            "30.9-37.4% noise, so this corpus has no theme structure the fixed "
            "configuration can find. No artifacts were written. Do not tune the "
            "parameters to force a result.",
            k=k,
            noise_fraction=noise_fraction,
        )

    themed = labels != -1
    silhouette = float(silhouette_score(coords[themed], labels[themed]))

    return ClusterResult(
        coords=coords.astype(np.float32),
        labels=labels,
        probabilities=probabilities,
        k=k,
        noise_fraction=noise_fraction,
        silhouette=silhouette,
        explained_variance_ratio_cumulative_at_50=explained,
    )


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def build_provenance(
    result: ClusterResult,
    *,
    estimated_cost_usd: float,
    config: ThemeMapConfig | None = None,
    timestamp: str | None = None,
) -> Provenance:
    """Record what produced a map, including the thread count (spec 5.8).

    ``OMP_NUM_THREADS`` is captured because t-SNE output is corpus-size dependent across
    thread counts: bit-identical at n=793 and n=807 between 8 threads and 1, but not at
    n=2,065 (ARI 0.275, 15-nearest-neighbour overlap 0.526, k 40 -> 38). An identical
    thread count always reproduces exactly. This is a record of the run, **not** a
    promise that the run can be regenerated from the seed -- the coordinates and labels
    are persisted as artifacts for precisely that reason.

    ``hdbscan`` exposes no ``__version__`` attribute, so its version comes from
    :func:`importlib.metadata.version`.
    """
    config = config or ThemeMapConfig()

    return Provenance(
        embedding_model=config.embedding_model,
        seed=config.tsne_random_state,
        omp_num_threads=os.environ.get("OMP_NUM_THREADS"),
        sklearn_version=sklearn.__version__,
        numpy_version=np.__version__,
        hdbscan_version=_package_version("hdbscan"),
        n_datasets=int(result.labels.shape[0]),
        n_themes=result.k,
        noise_fraction=result.noise_fraction,
        estimated_cost_usd=float(estimated_cost_usd),
        timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
        config=config,
    )
