"""Read and write the four theme-map artifacts (spec section 6).

The artifacts live in a ``theme_map/`` subdirectory of an identification run directory:

===================  =========================================================
``embeddings.npy``   float32, n x 1536, row order matching ``theme_map.csv``
``theme_map.csv``    one row per dataset
``themes.json``      one record per theme
``provenance.json``  model, seed, thread count, versions, cost, timestamp, config
===================  =========================================================

Spec section 6 requires that the set be written **at the end, after every step
succeeds**, and that a failed build leave no partial ``theme_map/`` directory behind so
that a retry starts clean. :func:`write_artifacts` therefore stages the whole set in a
sibling temporary directory and renames it into place in one step. A crash part way
through leaves the staging directory, which is removed, and the real directory
untouched.

The one deliberate exception is :func:`write_failure`, which writes ``FAILED.txt`` into
``theme_map/`` so that the GUI can show *why* a build failed rather than silently
offering the build button again.

Nothing here is forgiving: a missing or incomplete artifact set raises. It never returns
``None`` for a caller to skip over.
"""

from __future__ import annotations

import csv
import json
import shutil
import tempfile
import uuid
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .types import (
    DatasetPoint,
    Float32Array,
    Provenance,
    Theme,
    ThemeMapConfig,
    ThemeMapResult,
)

__all__ = [
    "ThemeMapArtifactError",
    "ThemeMapArtifactsMissing",
    "ThemeMapBuildFailed",
    "ThemeMapArtifactCorrupt",
    "THEME_MAP_DIRNAME",
    "ARTIFACT_FILENAMES",
    "FAILURE_FILENAME",
    "theme_map_dir",
    "artifacts_exist",
    "write_artifacts",
    "read_artifacts",
    "read_failure",
    "write_failure",
]


THEME_MAP_DIRNAME = "theme_map"
EMBEDDINGS_FILENAME = "embeddings.npy"
POINTS_FILENAME = "theme_map.csv"
THEMES_FILENAME = "themes.json"
PROVENANCE_FILENAME = "provenance.json"
FAILURE_FILENAME = "FAILED.txt"

#: The four files that constitute a complete artifact set (spec section 6).
ARTIFACT_FILENAMES = (
    EMBEDDINGS_FILENAME,
    POINTS_FILENAME,
    THEMES_FILENAME,
    PROVENANCE_FILENAME,
)

#: Column order of ``theme_map.csv``, fixed by spec section 6.
CSV_COLUMNS = (
    "GEO_Accession",
    "x",
    "y",
    "ThemeId",
    "ThemeProb",
    "RelevanceScore",
    "QuerySim",
    "QuerySimRank",
)

_STAGING_PREFIX = ".theme_map.staging."


class ThemeMapArtifactError(RuntimeError):
    """Base class for every artifact-layer failure."""


class ThemeMapArtifactsMissing(ThemeMapArtifactError):
    """No theme map has been built for this run directory."""


class ThemeMapBuildFailed(ThemeMapArtifactError):
    """A theme-map build was attempted for this run and failed.

    Carries the recorded failure message so the caller can show it verbatim.
    """

    def __init__(self, run_dir: Path, message: str) -> None:
        self.run_dir = Path(run_dir)
        self.failure_message = message
        super().__init__(
            f"The theme map build for {self.run_dir} failed and wrote no artifacts. "
            f"Recorded failure: {message}"
        )


class ThemeMapArtifactCorrupt(ThemeMapArtifactError):
    """The ``theme_map/`` directory exists but is incomplete or unreadable."""


def theme_map_dir(run_dir: Path | str) -> Path:
    """The ``theme_map/`` directory for an identification run directory."""
    return Path(run_dir) / THEME_MAP_DIRNAME


def artifacts_exist(run_dir: Path | str) -> bool:
    """True only when all four artifacts of spec section 6 are present.

    A missing directory is False. A directory holding only ``FAILED.txt`` is False. A
    partial set is False, because spec section 6 defines the artifacts as a set.
    """
    directory = theme_map_dir(run_dir)
    if not directory.is_dir():
        return False
    return all((directory / name).is_file() for name in ARTIFACT_FILENAMES)


def read_failure(run_dir: Path | str) -> str | None:
    """The recorded failure message for this run, or ``None`` if there is none.

    ``None`` here means "no ``FAILED.txt`` on disk", which is a genuine absence rather
    than a swallowed error.
    """
    failure_path = theme_map_dir(run_dir) / FAILURE_FILENAME
    if not failure_path.is_file():
        return None
    return failure_path.read_text(encoding="utf-8").strip()


def write_failure(run_dir: Path | str, message: str) -> None:
    """Record why a theme-map build failed, for the GUI to display.

    This is the only writer that leaves something in ``theme_map/`` without a complete
    artifact set. It does not create a partial artifact set: ``artifacts_exist`` stays
    False and ``read_artifacts`` raises :class:`ThemeMapBuildFailed`.
    """
    directory = theme_map_dir(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / FAILURE_FILENAME).write_text(message.rstrip() + "\n", encoding="utf-8")


def write_artifacts(run_dir: Path | str, result: ThemeMapResult) -> None:
    """Write the four artifacts as one atomic set (spec section 6).

    Everything is staged in a sibling temporary directory and renamed into place only
    after every file is written. If any step raises, the staging directory is removed
    and any previously published ``theme_map/`` is left exactly as it was.
    """
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    destination = theme_map_dir(run_path)

    staging = Path(tempfile.mkdtemp(prefix=_STAGING_PREFIX, dir=run_path))
    superseded = run_path / f"{_STAGING_PREFIX}superseded.{uuid.uuid4().hex}"
    published = False
    try:
        _write_embeddings(staging / EMBEDDINGS_FILENAME, result.embeddings)
        _write_points(staging / POINTS_FILENAME, result.points)
        _write_themes(staging / THEMES_FILENAME, result.themes)
        _write_provenance(staging / PROVENANCE_FILENAME, result.provenance)

        # Publish: move any existing set aside, then swing the new one into place.
        if destination.exists():
            destination.rename(superseded)
        staging.rename(destination)
        published = True
    finally:
        if not published:
            shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(superseded, ignore_errors=True)


def read_artifacts(run_dir: Path | str) -> ThemeMapResult:
    """Load a complete artifact set, or raise.

    Raises :class:`ThemeMapBuildFailed` when only a recorded failure is present,
    :class:`ThemeMapArtifactsMissing` when nothing has been built, and
    :class:`ThemeMapArtifactCorrupt` when the set is incomplete.
    """
    run_path = Path(run_dir)
    directory = theme_map_dir(run_path)

    if not artifacts_exist(run_path):
        failure = read_failure(run_path)
        if failure is not None:
            raise ThemeMapBuildFailed(run_path, failure)
        if not directory.is_dir():
            raise ThemeMapArtifactsMissing(
                f"No {THEME_MAP_DIRNAME}/ directory in {run_path}. "
                "Build the theme map for this run first."
            )
        present = sorted(p.name for p in directory.iterdir())
        missing = [n for n in ARTIFACT_FILENAMES if not (directory / n).is_file()]
        raise ThemeMapArtifactCorrupt(
            f"Incomplete theme map artifacts in {directory}: missing {missing}; "
            f"present {present}. The artifacts are written as a set, so this directory "
            "should be deleted and the theme map rebuilt."
        )

    embeddings = _read_embeddings(directory / EMBEDDINGS_FILENAME)
    points = _read_points(directory / POINTS_FILENAME)
    themes = _read_themes(directory / THEMES_FILENAME)
    provenance = _read_provenance(directory / PROVENANCE_FILENAME)

    if embeddings.shape[0] != len(points):
        raise ThemeMapArtifactCorrupt(
            f"{EMBEDDINGS_FILENAME} has {embeddings.shape[0]} rows but "
            f"{POINTS_FILENAME} has {len(points)}; they must be row-aligned."
        )

    return ThemeMapResult(
        points=points, themes=themes, embeddings=embeddings, provenance=provenance
    )


# --------------------------------------------------------------------------------------
# Writers. Each one is a module-level function so a failure in any single step can be
# exercised in isolation by the tests.
# --------------------------------------------------------------------------------------


def _write_embeddings(path: Path, embeddings: Float32Array) -> None:
    array = np.asarray(embeddings)
    if array.ndim != 2:
        raise ValueError(
            f"embeddings must be a 2-D n x d matrix, got shape {array.shape}"
        )
    np.save(path, array.astype(np.float32, copy=False), allow_pickle=False)


def _write_points(path: Path, points: list[DatasetPoint]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        for point in points:
            writer.writerow(
                [
                    point.geo_accession,
                    repr(float(point.x)),
                    repr(float(point.y)),
                    int(point.theme_id),
                    repr(float(point.theme_prob)),
                    # An unscored dataset is an empty cell, never 0.0 and never NaN.
                    "" if point.relevance_score is None else repr(float(point.relevance_score)),
                    repr(float(point.query_sim)),
                    int(point.query_sim_rank),
                ]
            )


def _write_themes(path: Path, themes: list[Theme]) -> None:
    payload = [asdict(theme) for theme in themes]
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_provenance(path: Path, provenance: Provenance) -> None:
    path.write_text(
        json.dumps(asdict(provenance), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


# --------------------------------------------------------------------------------------
# Readers
# --------------------------------------------------------------------------------------


def _read_embeddings(path: Path) -> Float32Array:
    return np.load(path, allow_pickle=False)


def _read_points(path: Path) -> list[DatasetPoint]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or tuple(reader.fieldnames) != CSV_COLUMNS:
            raise ThemeMapArtifactCorrupt(
                f"{path} has columns {reader.fieldnames}, expected {list(CSV_COLUMNS)}."
            )
        points: list[DatasetPoint] = []
        for row_number, row in enumerate(reader, start=2):
            try:
                points.append(
                    DatasetPoint(
                        geo_accession=row["GEO_Accession"],
                        x=float(row["x"]),
                        y=float(row["y"]),
                        theme_id=int(row["ThemeId"]),
                        theme_prob=float(row["ThemeProb"]),
                        relevance_score=_optional_float(row["RelevanceScore"]),
                        query_sim=float(row["QuerySim"]),
                        query_sim_rank=int(row["QuerySimRank"]),
                    )
                )
            except (TypeError, ValueError) as exc:
                raise ThemeMapArtifactCorrupt(
                    f"{path} line {row_number} is not a valid theme map row: {exc}"
                ) from exc
    return points


def _read_themes(path: Path) -> list[Theme]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ThemeMapArtifactCorrupt(
            f"{path} must hold a list of theme records, got {type(payload).__name__}."
        )
    try:
        return [
            Theme(
                id=int(record["id"]),
                label=record["label"],
                description=record["description"],
                n=int(record["n"]),
                keywords=list(record["keywords"]),
                representatives=list(record["representatives"]),
                mean_relevance=_optional_float(record["mean_relevance"]),
                max_relevance=_optional_float(record["max_relevance"]),
            )
            for record in payload
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise ThemeMapArtifactCorrupt(f"{path} holds a malformed theme record: {exc}") from exc


def _read_provenance(path: Path) -> Provenance:
    payload = json.loads(path.read_text(encoding="utf-8"))
    try:
        config = ThemeMapConfig(**payload["config"])
        return Provenance(
            embedding_model=payload["embedding_model"],
            seed=int(payload["seed"]),
            omp_num_threads=payload["omp_num_threads"],
            sklearn_version=payload["sklearn_version"],
            numpy_version=payload["numpy_version"],
            hdbscan_version=payload["hdbscan_version"],
            n_datasets=int(payload["n_datasets"]),
            n_themes=int(payload["n_themes"]),
            noise_fraction=float(payload["noise_fraction"]),
            estimated_cost_usd=float(payload["estimated_cost_usd"]),
            timestamp=payload["timestamp"],
            config=config,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ThemeMapArtifactCorrupt(f"{path} is malformed: {exc}") from exc


def _optional_float(value: object) -> float | None:
    """Empty cell or JSON ``null`` stays ``None``; everything else must parse.

    Spec 9.5 turns on this distinction: an unscored dataset is drawn differently from a
    dataset scored 0.0, so the two must never be conflated on the way back off disk.
    """
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    return float(value)  # type: ignore[arg-type]  # raises on anything unparseable
