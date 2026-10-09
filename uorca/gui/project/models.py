"""Project data models for UORCA GUI."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class SlurmConfig:
    """SLURM job configuration."""

    partition: str = ""
    constraint: str = ""
    cpus_per_task: int = 12
    memory: str = "16G"
    time_limit: str = "6:00:00"
    max_parallel: int = 6

    def to_dict(self) -> dict[str, Any]:
        return {
            "partition": self.partition,
            "constraint": self.constraint,
            "cpus_per_task": self.cpus_per_task,
            "memory": self.memory,
            "time_limit": self.time_limit,
            "max_parallel": self.max_parallel,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SlurmConfig":
        cfg = cls()
        cfg.partition = data.get("partition", "")
        cfg.constraint = data.get("constraint", "")
        cfg.cpus_per_task = data.get("cpus_per_task", 12)
        cfg.memory = data.get("memory", "16G")
        cfg.time_limit = data.get("time_limit", "6:00:00")
        cfg.max_parallel = data.get("max_parallel", 6)
        return cfg


@dataclass
class ScoringConfig:
    """Per-project tuning of the identification scoring pipeline."""

    biology_weight: float = 0.8
    design_min: int = 0
    stage1_biology_threshold: int = 3

    def to_dict(self) -> dict[str, Any]:
        return {
            "biology_weight": self.biology_weight,
            "design_min": self.design_min,
            "stage1_biology_threshold": self.stage1_biology_threshold,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScoringConfig":
        cfg = cls()
        cfg.biology_weight = float(data.get("biology_weight", 0.8))
        cfg.design_min = int(data.get("design_min", 0))
        cfg.stage1_biology_threshold = int(data.get("stage1_biology_threshold", 3))
        return cfg


@dataclass
class HpcConfig:
    """HPC connection and resource configuration."""

    host_alias: str = ""
    container_image: str = ""
    resource_dir: str = ""
    remote_working_dir: str = ""
    slurm: SlurmConfig = field(default_factory=SlurmConfig)

    def to_dict(self) -> dict[str, Any]:
        return {
            "host_alias": self.host_alias,
            "container_image": self.container_image,
            "resource_dir": self.resource_dir,
            "remote_working_dir": self.remote_working_dir,
            "slurm": self.slurm.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HpcConfig":
        cfg = cls()
        cfg.host_alias = data.get("host_alias", "")
        cfg.container_image = data.get("container_image", "")
        cfg.resource_dir = data.get("resource_dir", "")
        cfg.remote_working_dir = data.get("remote_working_dir", "")
        slurm_data = data.get("slurm", {})
        cfg.slurm = SlurmConfig.from_dict(slurm_data)
        return cfg


@dataclass
class RunEntry:
    """A single workflow run (identification or pipeline)."""

    id: str = ""
    type: str = ""  # "identification" | "pipeline"
    status: str = ""  # "pending" | "running" | "completed" | "failed"
    research_question: str = ""
    output_path: str = ""
    summary: dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[str] = field(default_factory=lambda: datetime.now().isoformat())
    params: Dict[str, Any] = field(default_factory=dict)
    input_from: Optional[str] = None
    target: Optional[str] = None
    datasets: List[str] = field(default_factory=list)
    custom_dataset_ids: List[str] = field(default_factory=list)
    note: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "status": self.status,
            "research_question": self.research_question,
            "output_path": self.output_path,
            "summary": self.summary,
            "timestamp": self.timestamp,
            "params": self.params,
            "input_from": self.input_from,
            "target": self.target,
            "datasets": self.datasets,
            "custom_dataset_ids": self.custom_dataset_ids,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunEntry":
        entry = cls()
        entry.id = data.get("id", "")
        entry.type = data.get("type", "")
        entry.status = data.get("status", "")
        entry.research_question = data.get("research_question", "")
        entry.output_path = data.get("output_path", "")
        entry.summary = data.get("summary", {})
        entry.timestamp = data.get("timestamp")
        entry.params = data.get("params", {})
        entry.input_from = data.get("input_from")
        entry.target = data.get("target")
        entry.datasets = data.get("datasets", [])
        entry.custom_dataset_ids = data.get("custom_dataset_ids") or []
        entry.note = data.get("note")
        return entry


@dataclass
class CustomDataset:
    """User-provided FASTQ dataset registered to a project."""

    id: str = ""                     # slug-style; deterministic from label
    label: str = ""                  # human-readable
    fastq_dir: str = ""              # absolute HPC path
    metadata_filename: str = ""      # filename under per-project staging dir
    organism: Optional[str] = None
    description: Optional[str] = None
    created: Optional[str] = field(default_factory=lambda: datetime.now().isoformat())
    last_validated: Optional[str] = None
    validation_status: str = "unvalidated"
    validation_message: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "fastq_dir": self.fastq_dir,
            "metadata_filename": self.metadata_filename,
            "organism": self.organism,
            "description": self.description,
            "created": self.created,
            "last_validated": self.last_validated,
            "validation_status": self.validation_status,
            "validation_message": self.validation_message,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CustomDataset":
        cd = cls()
        cd.id = data.get("id", "")
        cd.label = data.get("label", "")
        cd.fastq_dir = data.get("fastq_dir", "")
        cd.metadata_filename = data.get("metadata_filename", "")
        cd.organism = data.get("organism")
        cd.description = data.get("description")
        cd.created = data.get("created", cd.created)
        cd.last_validated = data.get("last_validated")
        cd.validation_status = data.get("validation_status", "unvalidated")
        cd.validation_message = data.get("validation_message")
        return cd


@dataclass
class Project:
    """A UORCA project, persisted as YAML."""

    name: str = ""
    description: str = ""
    default_research_question: str = ""
    hpc: HpcConfig = field(default_factory=HpcConfig)
    runs: list[RunEntry] = field(default_factory=list)
    custom_datasets: list[CustomDataset] = field(default_factory=list)
    scoring_config: ScoringConfig = field(default_factory=ScoringConfig)
    created: Optional[str] = field(default_factory=lambda: datetime.now().isoformat())
    updated: Optional[str] = None

    def __post_init__(self) -> None:
        if self.updated is None:
            self.updated = self.created

    def get_runs_by_type(self, run_type: str) -> list[RunEntry]:
        """Return all runs of a given type."""
        return [r for r in self.runs if r.type == run_type]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "default_research_question": self.default_research_question,
            "hpc": self.hpc.to_dict(),
            "runs": [r.to_dict() for r in self.runs],
            "custom_datasets": [d.to_dict() for d in self.custom_datasets],
            "scoring_config": self.scoring_config.to_dict(),
            "created": self.created,
            "updated": self.updated,
        }

    def save(self, path: Path) -> None:
        """Serialize project to YAML at the given path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)

    @classmethod
    def load(cls, path: Path) -> "Project":
        """Deserialize project from YAML at the given path."""
        path = Path(path)
        with path.open("r") as f:
            data = yaml.safe_load(f)
        project = cls()
        project.name = data.get("name", "")
        project.description = data.get("description", "")
        project.default_research_question = data.get("default_research_question", "")
        hpc_data = data.get("hpc", {})
        project.hpc = HpcConfig.from_dict(hpc_data)
        project.runs = [RunEntry.from_dict(r) for r in data.get("runs", [])]
        project.custom_datasets = [
            CustomDataset.from_dict(d) for d in data.get("custom_datasets") or []
        ]
        project.scoring_config = ScoringConfig.from_dict(data.get("scoring_config") or {})
        project.created = data.get("created")
        project.updated = data.get("updated")
        return project
