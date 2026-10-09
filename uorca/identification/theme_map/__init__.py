"""Theme map for an identification run (spec ``docs/superpowers/specs/2026-09-16-theme-map-design.md``).

Embeds every valid dataset of a run, projects it to two dimensions, clusters it, names
the clusters with one structured LLM call, and persists four artifacts under
``<run_dir>/theme_map/`` for the GUI to read.

The public surface is the dataclasses re-exported here plus :func:`build_theme_map`.
Nothing in this package degrades quietly: every failure condition in spec section 7
raises, and a failed build writes no artifacts.

**This module keeps its import surface light on purpose.** ``uorca/gui/components``
imports :mod:`.artifacts`, which executes this file, and the Streamlit app must not pay
for sklearn, hdbscan and pydantic-ai on every rerun. :mod:`.embed`, :mod:`.cluster` and
:mod:`.naming` are therefore imported inside :func:`build_theme_map` rather than at
module scope.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .artifacts import (
    ARTIFACT_FILENAMES,
    THEME_MAP_DIRNAME,
    ThemeMapArtifactCorrupt,
    ThemeMapArtifactError,
    ThemeMapArtifactsMissing,
    ThemeMapBuildFailed,
    artifacts_exist,
    read_artifacts,
    read_failure,
    theme_map_dir,
    write_artifacts,
    write_failure,
)
from .types import (
    DatasetPoint,
    Provenance,
    Theme,
    ThemeMapConfig,
    ThemeMapResult,
)

logger = logging.getLogger(__name__)

#: The results table an identification run writes, and the theme map's only population.
RESULTS_CSV_FILENAME = "Dataset_identification_result.csv"

__all__ = [
    # types
    "ThemeMapConfig",
    "Theme",
    "DatasetPoint",
    "Provenance",
    "ThemeMapResult",
    # artifacts
    "THEME_MAP_DIRNAME",
    "ARTIFACT_FILENAMES",
    "theme_map_dir",
    "artifacts_exist",
    "write_artifacts",
    "read_artifacts",
    "read_failure",
    "write_failure",
    "ThemeMapArtifactError",
    "ThemeMapArtifactsMissing",
    "ThemeMapBuildFailed",
    "ThemeMapArtifactCorrupt",
    # entry points
    "RESULTS_CSV_FILENAME",
    "BuildEstimate",
    "estimate_theme_map_build",
    "build_theme_map",
]


@dataclass(frozen=True)
class BuildEstimate:
    """What a theme-map build would cost, priced from a real token count (spec 9.7).

    ``n_datasets`` is the size of the **embedding corpus** — the valid rows that survive
    the two preparation filters of spec 5.1 — not the raw valid-row count. Those differ
    by 9.3-10.5%, because sub-series of one super-series are dropped.
    """

    n_datasets: int
    n_tokens: int
    cost_usd: float


def _read_results(run_dir: Path) -> "object":
    """Load the run's results CSV, or raise.

    A missing results file is a hard failure: there is no map without a population, and
    returning ``None`` here would let a caller skip the step silently.
    """
    import pandas as pd

    csv_path = run_dir / RESULTS_CSV_FILENAME
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"{csv_path} does not exist, so there is no population to map. "
            f"{RESULTS_CSV_FILENAME} is written at the end of an identification run; "
            "this run has not finished, or this is not a run directory."
        )
    return pd.read_csv(csv_path)


def _corpus(run_dir: Path, config: ThemeMapConfig):
    """The shared front half of a build and an estimate.

    Reads the results CSV, builds the embedding corpus, applies the corpus-size guard and
    reads the research query. Nothing here spends anything.
    """
    from .cluster import CorpusTooSmallError
    from .embed import build_texts, read_query

    frame = _read_results(run_dir)
    accessions, texts = build_texts(frame)  # type: ignore[arg-type]

    # Guard 1 of spec section 7, hoisted ahead of the spend. cluster_embeddings applies
    # it again on its own input; this copy exists so a tiny corpus costs nothing.
    if len(accessions) < config.min_datasets:
        raise CorpusTooSmallError(
            f"Theme map corpus is too small: {len(accessions)} datasets after the spec "
            f"5.1 filters, minimum {config.min_datasets}. Three themes at "
            f"min_cluster_size={config.hdbscan_min_cluster_size} need "
            f"{config.min_themes * config.hdbscan_min_cluster_size} clustered datasets "
            "and noise runs at 30.9-37.4%, so a run needs roughly 70 valid datasets "
            "before three themes are arithmetically possible. Widen the identification "
            "query; do not lower this guard."
        )

    # Read before any spend. Spec 5.7 forbids a hardcoded fallback, so a run whose query
    # cannot be read can never produce a map — and must not pay to discover that.
    query = read_query(run_dir)

    return frame, accessions, texts, query


def estimate_theme_map_build(
    run_dir: Path | str, *, config: ThemeMapConfig | None = None
) -> BuildEstimate:
    """Price a theme-map build without calling any API (spec 9.7, spec section 8).

    The count comes from the real ``cl100k_base`` tokeniser, not from a guess, and it
    includes the one extra document that :func:`~.embed.query_similarity` embeds.

    Raises:
        FileNotFoundError: The run has written no results CSV.
        CorpusTooSmallError: Fewer than ``config.min_datasets`` datasets survive the spec
            5.1 filters.
        ThemeMapQueryMissing: The run's research query cannot be read.
    """
    from .embed import estimate_cost

    settings = config or ThemeMapConfig()
    _, accessions, texts, query = _corpus(Path(run_dir), settings)
    tokens, cost = estimate_cost([*texts, query])
    return BuildEstimate(n_datasets=len(accessions), n_tokens=tokens, cost_usd=cost)


def build_theme_map(
    run_dir: Path | str, *, config: ThemeMapConfig | None = None
) -> ThemeMapResult:
    """Build the theme map for an identification run directory.

    Spec section 8 gives this function three callers and one code path: the end of
    ``dataset_identification.py``, the ``Build theme map`` button on the Identify page,
    and ``python -m uorca.identification.theme_map``.

    The order of the steps is fixed:

    results CSV -> :func:`~.embed.build_texts` -> corpus-size guard ->
    :func:`~.embed.read_query` -> :func:`~.embed.check_provider` ->
    :func:`~.embed.estimate_cost` -> :func:`~.embed.embed_texts` ->
    :func:`~.cluster.cluster_embeddings` -> :func:`~.naming.ctfidf` ->
    :func:`~.naming.select_representatives` -> :func:`~.naming.name_themes` ->
    :func:`~.embed.query_similarity` -> :func:`~.cluster.build_provenance` ->
    :func:`~.artifacts.write_artifacts`.

    Everything that cannot spend money runs first, and ``check_provider`` runs before
    anything that can. Every failure raises and nothing is written until every step has
    succeeded, so a failed build leaves no partial ``theme_map/`` behind (spec section 6).

    Args:
        run_dir: The identification run directory holding
            ``Dataset_identification_result.csv`` and ``identification_metadata.json``.
        config: The fixed numeric pipeline. Defaults to :class:`ThemeMapConfig`, whose
            values are fixed by spec section 5.2 and must not be re-derived or scaled.

    Returns:
        The built :class:`ThemeMapResult`, already persisted to ``<run_dir>/theme_map/``.

    Raises:
        FileNotFoundError: The results CSV or the bundled naming prompt is absent.
        KeyError: The results CSV is missing a required column.
        ThemeMapProviderError: The provider is Bedrock, or ``OPENAI_API_KEY`` is absent.
        ThemeMapQueryMissing: The run's research query cannot be read.
        CorpusTooSmallError: Fewer than ``config.min_datasets`` datasets.
        DegenerateClusteringError: Fewer than ``config.min_themes`` themes.
        ThemeNamingError: The naming call failed or returned something unusable.
    """
    import numpy as np
    import pandas as pd

    from . import cluster as cluster_module
    from . import embed as embed_module
    from . import naming as naming_module

    settings = config or ThemeMapConfig()
    run_path = Path(run_dir)

    # --- 1. Population, corpus, guards and query. Nothing here spends. -------------
    frame, accessions, texts, query = _corpus(run_path, settings)
    assert isinstance(frame, pd.DataFrame)  # _read_results returns a frame or raises

    titles = _column_by_accession(frame, "Title", accessions, default="")
    relevance = _relevance_by_accession(frame, accessions)

    # --- 2. Provider guard, then the spend (spec section 7) ------------------------
    embed_module.check_provider()
    tokens, cost = embed_module.estimate_cost([*texts, query])
    logger.info(
        "Theme map: embedding %d datasets (%d tokens, about $%.4f)",
        len(texts),
        tokens,
        cost,
    )
    embeddings = embed_module.embed_texts(texts, config=settings)

    # --- 3. The numeric pipeline (spec 5.2). Raises on a degenerate result. --------
    clustered = cluster_module.cluster_embeddings(embeddings, config=settings)
    labels = clustered.labels
    if labels.shape[0] != len(accessions):
        raise ValueError(
            f"Clustering returned {labels.shape[0]} labels for {len(accessions)} "
            "datasets; the map would be misaligned with the corpus."
        )
    theme_ids = sorted({int(label) for label in np.unique(labels)} - {-1})

    # --- 4. Evidence: c-TF-IDF keywords and representative titles (spec 5.4, 5.5) --
    texts_by_theme = {
        theme_id: [texts[i] for i in range(len(texts)) if int(labels[i]) == theme_id]
        for theme_id in theme_ids
    }
    keywords = naming_module.ctfidf(texts_by_theme, texts)
    representatives = {
        theme_id: naming_module.select_representatives(
            theme_id, accessions, titles, labels, clustered.probabilities
        )
        for theme_id in theme_ids
    }

    # --- 5. One structured naming call for every theme (spec 5.6, decision 10) -----
    namings = naming_module.name_themes(
        [
            naming_module.ThemeNamingInput(
                theme_id=theme_id,
                n=len(texts_by_theme[theme_id]),
                keywords=keywords.get(theme_id, []),
                representatives=representatives[theme_id],
            )
            for theme_id in theme_ids
        ]
    )
    named = {naming.theme_id: naming for naming in namings}

    # --- 6. QuerySim: one extra API call, a sortable column and nothing more -------
    query_sim, query_sim_rank = embed_module.query_similarity(
        query, embeddings, config=settings
    )

    # --- 7. Assemble, in corpus row order so every artifact stays aligned ----------
    points = [
        DatasetPoint(
            geo_accession=accessions[index],
            x=float(clustered.coords[index][0]),
            y=float(clustered.coords[index][1]),
            theme_id=int(labels[index]),
            # HDBSCAN already reports 0.0 for an unthemed point; passed straight through.
            theme_prob=float(clustered.probabilities[index]),
            relevance_score=relevance[index],
            query_sim=query_sim[index],
            query_sim_rank=query_sim_rank[index],
        )
        for index in range(len(accessions))
    ]

    themes = []
    for theme_id in theme_ids:
        scores = [
            point.relevance_score
            for point in points
            if point.theme_id == theme_id and point.relevance_score is not None
        ]
        themes.append(
            Theme(
                id=theme_id,
                label=named[theme_id].label,
                description=named[theme_id].description,
                n=len(texts_by_theme[theme_id]),
                keywords=list(keywords.get(theme_id, [])),
                representatives=list(representatives[theme_id]),
                # None, never 0.0: an unscored theme is not a zero-scoring one.
                mean_relevance=(sum(scores) / len(scores)) if scores else None,
                max_relevance=max(scores) if scores else None,
            )
        )

    provenance = cluster_module.build_provenance(
        clustered, estimated_cost_usd=cost, config=settings
    )
    result = ThemeMapResult(
        points=points,
        themes=themes,
        embeddings=np.asarray(embeddings, dtype=np.float32),
        provenance=provenance,
    )

    # --- 8. Publish, only now that every step has succeeded (spec section 6) -------
    write_artifacts(run_path, result)
    logger.info(
        "Theme map written to %s: %d datasets, %d themes, %.1f%% unthemed",
        theme_map_dir(run_path),
        provenance.n_datasets,
        provenance.n_themes,
        provenance.noise_fraction * 100,
    )
    return result


# ---------------------------------------------------------------------------
# Small joins from the results CSV onto the corpus row order
# ---------------------------------------------------------------------------


def _column_by_accession(frame, column: str, accessions: list[str], *, default: str) -> list[str]:
    """One text column, re-ordered onto the corpus row order.

    A missing column raises rather than returning blanks: an unnamed representative
    title would silently degrade the naming call's evidence.
    """
    if column not in frame.columns:
        raise KeyError(
            f"{RESULTS_CSV_FILENAME} has no '{column}' column, which the theme map needs."
        )
    lookup = dict(zip(frame["GEO_Accession"].astype(str), frame[column]))
    values: list[str] = []
    for accession in accessions:
        value = lookup.get(accession, default)
        values.append(default if value is None or value != value else str(value))
    return values


def _relevance_by_accession(frame, accessions: list[str]) -> list[float | None]:
    """The relevance score per corpus row, with an unscored dataset staying ``None``.

    Spec 9.5 draws an unscored dataset differently from a low-scoring one, so a missing
    score is never coerced to ``0.0``. A results CSV with no ``RelevanceScore`` column at
    all is an older run, and every point is then unscored.
    """
    if "RelevanceScore" not in frame.columns:
        logger.warning(
            "%s has no RelevanceScore column; every point will be drawn as unscored.",
            RESULTS_CSV_FILENAME,
        )
        return [None] * len(accessions)

    lookup = dict(zip(frame["GEO_Accession"].astype(str), frame["RelevanceScore"]))
    scores: list[float | None] = []
    for accession in accessions:
        raw = lookup.get(accession)
        if raw is None or raw != raw or (isinstance(raw, str) and not raw.strip()):
            scores.append(None)
        else:
            scores.append(float(raw))
    return scores
