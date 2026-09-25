"""Shared contracts for the theme-map feature.

This module is an **addition** to the module list in spec section 8, which names five
modules (``__init__``, ``embed``, ``cluster``, ``naming``, ``artifacts``). This is a
sixth. It exists so that ``embed``, ``cluster``, ``naming`` and ``artifacts`` can share
the same dataclasses without importing one another: dependencies flow one way, from
every other module in the package into this one, and never back out.

Every numeric default on :class:`ThemeMapConfig` is fixed by spec section 5.2 and
justified individually in spec section 5.3. They are not free choices and they must not
be made adaptive: a configuration fitted on two corpora already collapsed to a 43.9%
catch-all cluster on a held-out third.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

Float32Array = NDArray[np.float32]


__all__ = [
    "Float32Array",
    "ThemeMapConfig",
    "Theme",
    "DatasetPoint",
    "Provenance",
    "ThemeMapResult",
]


@dataclass(frozen=True)
class ThemeMapConfig:
    """The fixed numeric pipeline of spec section 5.2.

    Do not re-derive these values, and do not scale any of them by corpus size. Spec
    section 5.3 states why each one is fixed.

    ``hdbscan_*`` fields are for the ``hdbscan`` package, not
    ``sklearn.cluster.HDBSCAN``: the two disagree on ``min_samples`` (sklearn counts the
    point itself), and the validated numbers were measured with ``hdbscan``.
    """

    # Embeddings (spec 5.1, 5.7)
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    # PCA (spec 5.2; 5.3 "svd_solver", "PCA at 50 components")
    pca_components: int = 50
    pca_svd_solver: str = "covariance_eigh"

    # t-SNE (spec 5.2; 5.3 "perplexity=30")
    tsne_perplexity: float = 30.0
    tsne_init: str = "pca"
    tsne_learning_rate: str | float = "auto"
    tsne_random_state: int = 42

    # HDBSCAN, run on the two t-SNE coordinates (spec 5.2; 5.3 "leaf", "2-D display space")
    hdbscan_min_cluster_size: int = 15
    hdbscan_min_samples: int = 3
    hdbscan_cluster_selection_method: str = "leaf"
    hdbscan_approx_min_span_tree: bool = False

    # Hard guards (spec section 7)
    min_datasets: int = 50
    min_themes: int = 3
    # Spec 5.2 step 1: the embedding matrix is asserted unit-norm, not
    # renormalised. text-embedding-3-small returns unit vectors, and float32
    # storage rounds them to within 5.7e-4. Renormalising perturbs at that level
    # and t-SNE amplifies it; see cluster.py for the measurement.
    unit_norm_tolerance: float = 1e-3


@dataclass(frozen=True)
class Theme:
    """One named cluster, as persisted in ``themes.json`` (spec section 6).

    ``keywords`` is the c-TF-IDF evidence that produced ``label``; it is kept beside the
    label so a user can audit the naming call (spec 5.6).

    ``mean_relevance`` and ``max_relevance`` are ``None`` when no dataset in the theme
    carries a relevance score. They are never coerced to ``0.0``.
    """

    id: int
    label: str
    description: str
    n: int
    keywords: list[str] = field(default_factory=list)
    representatives: list[str] = field(default_factory=list)
    mean_relevance: float | None = None
    max_relevance: float | None = None


@dataclass(frozen=True)
class DatasetPoint:
    """One dataset, as persisted in a row of ``theme_map.csv`` (spec section 6).

    ``theme_id`` is ``-1`` for an unthemed dataset and ``theme_prob`` is then ``0.0``.
    ``relevance_score`` is ``None`` for a dataset the run never scored; spec 9.5 requires
    that such a point be drawn differently from a genuinely low-scoring one, so the
    distinction between ``None`` and ``0.0`` must survive persistence.
    """

    geo_accession: str
    x: float
    y: float
    theme_id: int
    theme_prob: float
    relevance_score: float | None
    query_sim: float
    query_sim_rank: int


@dataclass(frozen=True)
class Provenance:
    """The contents of ``provenance.json`` (spec section 6).

    ``omp_num_threads`` is recorded because t-SNE output is corpus-size dependent across
    thread counts (spec 5.8): at n=2,065 an 8-thread and a 1-thread run differ. It is
    ``None`` when the environment variable is unset. Regeneration from ``seed`` alone is
    **not** promised, and no code or document may claim it.
    """

    embedding_model: str
    seed: int
    omp_num_threads: str | None
    sklearn_version: str
    numpy_version: str
    hdbscan_version: str
    n_datasets: int
    n_themes: int
    noise_fraction: float
    estimated_cost_usd: float
    timestamp: str
    config: ThemeMapConfig


@dataclass(frozen=True, eq=False)
class ThemeMapResult:
    """Everything one theme-map build produces.

    ``embeddings`` is the float32 n x 1536 matrix persisted as ``embeddings.npy``; its
    row order matches ``points``.

    ``eq`` is disabled because a generated ``__eq__`` over a numpy array returns an array
    rather than a bool. Compare the fields instead.
    """

    points: list[DatasetPoint]
    themes: list[Theme]
    embeddings: Float32Array
    provenance: Provenance
