"""Attach UORCA results produced outside the GUI to a project.

Results written by ``uorca run local|slurm``, by site scripts, or copied from an HPC carry no
project record, so the Explore page (which lists a project's completed pipeline runs) cannot
open them. This module inspects such a folder and records it as a completed pipeline run.

Pure Python: no Streamlit import, so the CLI and the GUI share it.

Accepted layouts (``path`` is what the user passes):

(a) analysis folder:   ``path/GSE*/RNAseqAnalysis/`` (+ ``path/GSE*/metadata/analysis_info.json``)
(b) arm/run folder:    ``path/analysis/`` laid out as (a), optionally ``path/identify/``
(c) one dataset:       ``path/RNAseqAnalysis/``

A dataset is *complete* only when ``analysis_info.json`` says ``"analysis_success": true``, its
``unique_groups`` is not null, and ``RNAseqAnalysis/CPM.csv`` exists: the same rule the Explorer
(``ResultsIntegrator.load_analysis_info``/``find_data_files``) uses to decide what it loads. A
dataset with DEG tables but not complete is *partial* (e.g. DEG tables written, then the R step
failed). Everything else that was found is *failed*.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from .manager import ProjectManager
from .models import Project, RunEntry
from .paths import resolve_local_path

Layout = Literal["analysis_folder", "arm_folder", "dataset_folder"]

LAYOUT_LABELS: dict[str, str] = {
    "analysis_folder": "Analysis folder (one sub-folder per dataset)",
    "arm_folder": "Run folder (analysis/ and identify/)",
    "dataset_folder": "Single dataset folder",
}

# Sub-folders of an analysis folder that are never datasets.
_NON_DATASET_DIRS = {"job_status", "logs", "identify", "analysis", "tmp", "scratch"}
_STATUS_SUFFIX = "_status.json"


class ImportResultsError(ValueError):
    """The path cannot be imported (missing, holds no analysis, nothing complete)."""


class DuplicateImportError(ImportResultsError):
    """The results folder is already recorded in the project."""


@dataclass
class ImportCandidate:
    """What :func:`inspect_results` found at a path."""

    source: Path                      # the resolved path the user gave
    layout: Layout
    analysis_dir: Path                # folder handed to the Explorer (becomes output_path)
    datasets: list[str]               # every dataset found (folders + job_status records)
    complete: list[str]
    partial: list[str]
    failed: list[str]
    research_question: Optional[str] = None
    identify_dir: Optional[Path] = None
    warnings: list[str] = field(default_factory=list)

    @property
    def importable(self) -> bool:
        return bool(self.complete)

    def dataset_status(self, accession: str) -> str:
        if accession in self.complete:
            return "complete"
        if accession in self.partial:
            return "partial"
        return "failed"


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------


def _is_dataset_dir(path: Path) -> bool:
    return (path / "RNAseqAnalysis").is_dir() or (path / "metadata" / "analysis_info.json").is_file()


def _dataset_dirs(analysis_dir: Path) -> list[Path]:
    return sorted(
        p for p in analysis_dir.iterdir()
        if p.is_dir() and p.name not in _NON_DATASET_DIRS and _is_dataset_dir(p)
    )


def _read_info(dataset_dir: Path, warnings: list[str]) -> Optional[dict]:
    """Return the dataset's analysis_info.json as a dict (None when absent or unreadable)."""
    for info_path in (dataset_dir / "metadata" / "analysis_info.json", dataset_dir / "analysis_info.json"):
        if not info_path.is_file():
            continue
        try:
            data = json.loads(info_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            warnings.append(f"{dataset_dir.name}: unreadable {info_path.name} ({exc}); counted as not complete")
            return None
        if not isinstance(data, dict):
            warnings.append(f"{dataset_dir.name}: {info_path.name} is not a JSON object; counted as not complete")
            return None
        return data
    return None


def _classify(dataset_dir: Path, warnings: list[str]) -> str:
    """complete / partial / failed, matching what ResultsIntegrator will actually load."""
    rnaseq = dataset_dir / "RNAseqAnalysis"
    info = _read_info(dataset_dir, warnings)
    success = info is not None and info.get("analysis_success") is True
    has_groups = info is not None and info.get("unique_groups") is not None
    has_cpm = (rnaseq / "CPM.csv").is_file()
    if success and has_groups and has_cpm:
        return "complete"
    if success and not has_groups:
        warnings.append(
            f"{dataset_dir.name}: analysis_info.json has unique_groups null; the Explorer skips "
            "this dataset, so it is counted as not complete"
        )
    if success and not has_cpm:
        warnings.append(f"{dataset_dir.name}: analysis_success is true but RNAseqAnalysis/CPM.csv is missing")
    has_deg = rnaseq.is_dir() and any(rnaseq.glob("*/DEG.csv"))
    return "partial" if has_deg else "failed"


def _job_status_accessions(analysis_dir: Path) -> list[str]:
    status_dir = analysis_dir / "job_status"
    if not status_dir.is_dir():
        return []
    return sorted(p.name[: -len(_STATUS_SUFFIX)] for p in status_dir.glob(f"*{_STATUS_SUFFIX}"))


def _read_research_question(analysis_dir: Path, warnings: list[str]) -> Optional[str]:
    rq_path = analysis_dir / "research_question.json"
    if not rq_path.is_file():
        return None
    try:
        data = json.loads(rq_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        warnings.append(f"Unreadable research_question.json ({exc})")
        return None
    rq = data.get("research_question") if isinstance(data, dict) else None
    if isinstance(rq, str) and rq.strip():
        return rq.strip()
    warnings.append("research_question.json has no 'research_question' text")
    return None


def inspect_results(path: str | Path) -> ImportCandidate:
    """Inspect a results folder and report what can be imported.

    Raises:
        ImportResultsError: the path does not exist, is not a folder, or holds no analysis.
    """
    raw = str(path).strip()
    if not raw:
        raise ImportResultsError("No path given.")
    source = Path(raw).expanduser().resolve()
    if not source.exists():
        raise ImportResultsError(f"Path does not exist: {source}")
    if not source.is_dir():
        raise ImportResultsError(f"Path is not a folder: {source}")

    layout: Layout
    identify_dir: Optional[Path] = None
    if (source / "RNAseqAnalysis").is_dir():
        layout = "dataset_folder"
        analysis_dir = source
        dataset_dirs = [source]
    elif (source / "analysis").is_dir() and _dataset_dirs(source / "analysis"):
        layout = "arm_folder"
        analysis_dir = source / "analysis"
        dataset_dirs = _dataset_dirs(analysis_dir)
        identify_dir = source / "identify" if (source / "identify").is_dir() else None
    else:
        layout = "analysis_folder"
        analysis_dir = source
        dataset_dirs = _dataset_dirs(source)
        sibling = source.parent / "identify"
        if source.name == "analysis" and sibling.is_dir():
            identify_dir = sibling

    if not dataset_dirs:
        raise ImportResultsError(
            f"No UORCA analysis found in {source}. Expected one of: "
            "<folder>/GSE*/RNAseqAnalysis/, <folder>/analysis/GSE*/RNAseqAnalysis/, "
            "or <folder>/RNAseqAnalysis/."
        )

    warnings: list[str] = []
    status: dict[str, str] = {d.name: _classify(d, warnings) for d in dataset_dirs}
    if layout != "dataset_folder":
        for acc in _job_status_accessions(analysis_dir):
            status.setdefault(acc, "failed")  # submitted, but no dataset folder was written

    datasets = sorted(status)
    return ImportCandidate(
        source=source,
        layout=layout,
        analysis_dir=analysis_dir,
        datasets=datasets,
        complete=[a for a in datasets if status[a] == "complete"],
        partial=[a for a in datasets if status[a] == "partial"],
        failed=[a for a in datasets if status[a] == "failed"],
        research_question=_read_research_question(analysis_dir, warnings),
        identify_dir=identify_dir,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------


def default_project_name(candidate: ImportCandidate) -> str:
    """Project name for a path: the folder name (the run folder's, for <run>/analysis)."""
    if candidate.source.name == "analysis" and candidate.layout == "analysis_folder":
        return candidate.source.parent.name
    return candidate.source.name


def _same_path(recorded: str, target: Path) -> bool:
    if not recorded:
        return False
    p = resolve_local_path(recorded)
    try:
        return p.resolve() == target.resolve()
    except OSError:
        return False


def find_imported_run(project: Project, candidate: ImportCandidate) -> Optional[RunEntry]:
    """The pipeline run in ``project`` that already records ``candidate``'s analysis folder."""
    for run in project.get_runs_by_type("pipeline"):
        if _same_path(run.output_path, candidate.analysis_dir):
            return run
    return None


def _unique_id(project: Project, base: str) -> str:
    taken = {r.id for r in project.runs}
    if base not in taken:
        return base
    n = 2
    while f"{base}_{n}" in taken:
        n += 1
    return f"{base}_{n}"


def _require_importable(candidate: ImportCandidate) -> None:
    """Raise ImportResultsError when no dataset is complete."""
    if not candidate.importable:
        raise ImportResultsError(
            f"No complete dataset in {candidate.analysis_dir} (analysis_success true, "
            f"unique_groups set, and RNAseqAnalysis/CPM.csv): {len(candidate.partial)} partial, "
            f"{len(candidate.failed)} failed. There is nothing to explore."
        )


def import_results(
    project: Project,
    candidate: ImportCandidate,
    note: Optional[str] = None,
    now: Optional[datetime] = None,
) -> list[RunEntry]:
    """Append the candidate to ``project.runs`` as completed run(s); the caller saves.

    Appends a ``pipeline`` run and, when an identify folder exists, an ``identification`` run
    that the pipeline run links to through ``input_from``. Returns the new entries.

    Raises:
        DuplicateImportError: the analysis folder is already recorded in this project.
        ImportResultsError: no dataset is complete, so there is nothing to explore.
    """
    existing = find_imported_run(project, candidate)
    if existing is not None:
        raise DuplicateImportError(
            f"{candidate.analysis_dir} is already in project '{project.name}' (run {existing.id})."
        )
    _require_importable(candidate)

    now = now or datetime.now()
    stamp = now.strftime("%Y%m%d_%H%M%S")
    tag = re.sub(r"[^A-Za-z0-9]+", "_", default_project_name(candidate)).strip("_") or "results"
    rq = candidate.research_question or ""
    params_common = {
        "imported": True,
        "source": str(candidate.source),
        "imported_at": now.isoformat(timespec="seconds"),
        "layout": candidate.layout,
    }

    entries: list[RunEntry] = []
    id_entry: Optional[RunEntry] = None
    if candidate.identify_dir is not None:
        id_entry = RunEntry(
            id=_unique_id(project, f"identify_{stamp}_imported_{tag}"),
            type="identification",
            status="completed",
            research_question=rq,
            output_path=str(candidate.identify_dir),
            timestamp=now.isoformat(),
            params=dict(params_common),
            note=note,
        )
        project.runs.append(id_entry)
        entries.append(id_entry)

    pipe_entry = RunEntry(
        id=_unique_id(project, f"pipeline_{stamp}_imported_{tag}"),
        type="pipeline",
        status="completed",
        research_question=rq,
        output_path=str(candidate.analysis_dir),
        timestamp=now.isoformat(),
        datasets=list(candidate.datasets),
        summary={
            "submitted": len(candidate.datasets),
            "succeeded": len(candidate.complete),
            "failed": len(candidate.failed),
            "partial": len(candidate.partial),
        },
        params=dict(params_common),
        input_from=id_entry.id if id_entry else None,
        note=note,
    )
    project.runs.append(pipe_entry)
    entries.append(pipe_entry)
    project.updated = now.isoformat()
    return entries


def import_into_named_project(
    mgr: ProjectManager,
    candidate: ImportCandidate,
    project_name: str,
    create: bool = False,
    note: Optional[str] = None,
) -> tuple[Project, list[RunEntry]]:
    """Load (or, with ``create``, create) ``project_name``, import, and save.

    Raises:
        ImportResultsError: the project does not exist and ``create`` is False, or another
            project already uses the same file name; plus everything :func:`import_results` raises.
    """
    name = project_name.strip()
    if not name:
        raise ImportResultsError("Project name is empty.")
    slug = mgr.get_project_slug(name)
    if not slug:
        raise ImportResultsError(f"Project name '{name}' has no letters or digits.")
    if mgr.project_exists(name):
        project = mgr.load_project(slug)
        if project.name != name:
            raise ImportResultsError(
                f"Project '{project.name}' already uses the file name '{slug}'; choose another name."
            )
    elif create:
        # Check before create_project writes the YAML, so a failed import leaves no empty project.
        _require_importable(candidate)
        project = mgr.create_project(name)
    else:
        raise ImportResultsError(f"Project '{name}' does not exist (create it, or pass --create).")

    entries = import_results(project, candidate, note=note)
    mgr.save_project(slug, project)
    return project, entries


def _explore_project_names(candidate: ImportCandidate) -> list[str]:
    """Candidate project names for ``uorca explore <path>``, most preferred first.

    The folder name, then ``<parent>/<folder>`` (so ``q1/openai`` and ``q2/openai`` differ).
    """
    anchor = candidate.source
    if candidate.layout == "analysis_folder" and anchor.name == "analysis":
        anchor = anchor.parent
    names = [anchor.name]
    if anchor.parent.name:
        names.append(f"{anchor.parent.name}/{anchor.name}")
    return names


def find_project_recording(
    mgr: ProjectManager, candidate: ImportCandidate
) -> Optional[Project]:
    """The first project (by slug) whose pipeline runs already record ``candidate``."""
    for project in mgr.list_projects():
        if find_imported_run(project, candidate) is not None:
            return project
    return None


def ensure_imported(
    mgr: ProjectManager, path: str | Path
) -> tuple[Project, Optional[list[RunEntry]]]:
    """Import ``path`` for ``uorca explore <path>``, once, into a project of its own.

    If any project already records this folder, that project is returned and nothing is
    imported. Otherwise a **new** project is created, named after the folder; when that name
    is taken it falls back to ``<parent>/<folder>`` and then a numeric suffix. An existing
    project is never appended to, so folders that share a name (``q1/openai``, ``q2/openai``)
    and projects the user built in the GUI stay separate. Returns the project and the new
    entries (``None`` when the folder was already recorded).
    """
    candidate = inspect_results(path)
    recorded = find_project_recording(mgr, candidate)
    if recorded is not None:
        return recorded, None

    names = _explore_project_names(candidate)
    name = next((n for n in names if not mgr.project_exists(n)), None)
    if name is None:
        base = names[-1]
        n = 2
        while mgr.project_exists(f"{base} ({n})"):
            n += 1
        name = f"{base} ({n})"
    project, entries = import_into_named_project(mgr, candidate, name, create=True)
    return project, entries
