"""Opt-in regression test against the four real embedding corpora (spec section 10.2).

This is the only evidence that the pipeline in :mod:`uorca.identification.theme_map.cluster`
is the pipeline that spec section 5.2 validated. Everything else in the unit suite runs on
synthetic vectors and would pass just as happily on a subtly different configuration.

The ``.npy`` files live under ``docs/``, which ``.gitignore`` excludes, so they may simply
be absent on a given machine. The test skips in that case rather than failing. It makes no
API call: the embeddings were paid for once, on 2026-08-13, for $0.0272.

Spec section 10.2 sets the acceptance bands: k between 15 and 45, noise below 45%, and
largest cluster below 15%. This module asserts those bands on all four corpora, and
additionally asserts the spec 5.2 table EXACTLY on the two corpora where t-SNE output is
known to be thread-count independent (n=793 and n=807; spec 5.8). At n=1,199 and n=2,065
only the bands are asserted, because spec 5.8 records that output at n=2,065 is not
bit-identical across ``OMP_NUM_THREADS``.

Measured on this implementation, 2026-09-17, ``OMP_NUM_THREADS`` unset. All four rows of
the spec 5.2 table reproduce exactly:

===============  ===========================  ==========================
corpus           this module                  spec 5.2 table
===============  ===========================  ==========================
hcm_2065         k=40, noise 35.0%, max 4.4%  k=40, noise 35.0%, max 4.4%
npc_fixed_793    k=20, noise 30.9%, max 7.8%  k=20, noise 30.9%, max 7.8%
npc_807          k=20, noise 34.9%, max 7.7%  k=20, noise 34.9%, max 7.7%
pparg_1199       k=25, noise 37.4%, max 9.1%  k=25, noise 37.4%, max 9.1%
===============  ===========================  ==========================

History worth keeping: an earlier revision of this pipeline L2-normalised the embedding
matrix, because spec 5.2 step 1 said "L2-normalise". The script that produced the spec
table (``exp_d_final.py``) never did, although its own docstring claimed it. Renormalising
already-unit vectors perturbs them at the 1e-4 level and t-SNE amplifies that: k fell to
37/19/19/22 and noise rose to 41.1/34.4/39.8/42.1%. Spec 5.2 step 1 was corrected on
2026-09-17 to assert unit norm rather than impose it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from uorca.identification.theme_map.cluster import cluster_embeddings

EMBEDDINGS_DIR = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "theme_map_validation"
    / "real_embeddings"
)

# corpus key -> (n, k, noise %, largest cluster %) measured in spec section 5.2.
MEASURED: dict[str, tuple[int, int, float, float]] = {
    "hcm_2065": (2065, 40, 35.0, 4.4),
    "npc_fixed_793": (793, 20, 30.9, 7.8),
    "npc_807": (807, 20, 34.9, 7.7),
    "pparg_1199": (1199, 25, 37.4, 9.1),
}


# Spec 5.8: t-SNE output at these two sizes is bit-identical across OMP_NUM_THREADS 8
# versus 1. At n=2,065 it is not (ARI 0.275, 15-NN overlap 0.526, k 40 -> 38).
THREAD_INDEPENDENT = frozenset({"npc_fixed_793", "npc_807"})


def _embedding_path(key: str) -> Path:
    return EMBEDDINGS_DIR / f"emb_{key}.npy"


def _all_corpora_present() -> bool:
    return all(_embedding_path(key).is_file() for key in MEASURED)


requires_real_embeddings = pytest.mark.skipif(
    not _all_corpora_present(),
    reason=(
        "Real embedding corpora are absent from "
        f"{EMBEDDINGS_DIR} (docs/ is gitignored). Spec section 10.2 makes this test "
        "opt-in for that reason."
    ),
)


@pytest.mark.slow
@requires_real_embeddings
@pytest.mark.parametrize("key", sorted(MEASURED))
def test_validated_configuration_reproduces_the_measured_band(key: str) -> None:
    """Spec 10.2: k in [15, 45], noise < 45%, largest cluster < 15%."""
    expected_n, _expected_k, _expected_noise, _expected_largest = MEASURED[key]

    embeddings = np.load(_embedding_path(key))
    assert embeddings.shape[0] == expected_n, (
        f"{key} has {embeddings.shape[0]} rows, expected {expected_n}; the corpus on "
        "disk is not the one spec section 5.2 was measured on."
    )

    result = cluster_embeddings(embeddings)

    sizes = np.array(
        [int((result.labels == theme_id).sum()) for theme_id in range(result.k)]
    )
    largest_fraction = float(sizes.max()) / embeddings.shape[0]

    assert 15 <= result.k <= 45, f"{key}: k={result.k} outside [15, 45]"

    # Spec 5.8: n=793 and n=807 are bit-identical across OMP_NUM_THREADS 8 versus 1, so
    # the spec 5.2 row can be asserted exactly there. n=1,199 is untested for thread
    # independence and n=2,065 is known NOT to be, so those two get the bands only.
    if key in THREAD_INDEPENDENT:
        assert result.k == _expected_k, (
            f"{key}: k={result.k}, spec 5.2 measured {_expected_k}. This corpus is "
            "thread-count independent, so a difference is a real change to the pipeline."
        )
        assert abs(100 * result.noise_fraction - _expected_noise) < 0.05, (
            f"{key}: noise {100 * result.noise_fraction:.1f}%, spec 5.2 measured "
            f"{_expected_noise}%."
        )
        assert abs(100 * largest_fraction - _expected_largest) < 0.05, (
            f"{key}: largest cluster {100 * largest_fraction:.1f}%, spec 5.2 measured "
            f"{_expected_largest}%."
        )
    assert result.noise_fraction < 0.45, (
        f"{key}: noise {result.noise_fraction:.1%} is not below 45%"
    )
    assert largest_fraction < 0.15, (
        f"{key}: largest cluster {largest_fraction:.1%} is not below 15% -- this is the "
        'catch-all cluster that cluster_selection_method="leaf" exists to prevent.'
    )

    # Row alignment is what lets embeddings.npy and theme_map.csv be read together.
    assert result.coords.shape == (expected_n, 2)
    assert result.labels.shape == (expected_n,)
    assert result.probabilities.shape == (expected_n,)

    # Spec 5.3: PCA at 50 components retains 55.2-60.0% on real embeddings. The TF-IDF
    # surrogate claimed 12.5-19.9% and was badly wrong; this guards against a regression
    # back to the surrogate's world.
    assert 0.5 < result.explained_variance_ratio_cumulative_at_50 < 0.7


@pytest.mark.slow
@requires_real_embeddings
def test_leaf_selection_prevents_a_catch_all_cluster_on_every_real_corpus() -> None:
    """Spec 5.3: ``eom`` produced a catch-all in 39-61% of configurations; ``leaf`` in 0%."""
    for key in sorted(MEASURED):
        embeddings = np.load(_embedding_path(key))
        result = cluster_embeddings(embeddings)
        sizes = np.array(
            [int((result.labels == theme_id).sum()) for theme_id in range(result.k)]
        )
        assert float(sizes.max()) / embeddings.shape[0] < 0.25, (
            f"{key}: a cluster holds more than a quarter of the corpus."
        )
