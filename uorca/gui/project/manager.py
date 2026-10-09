"""ProjectManager: CRUD operations for UORCA projects."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import yaml

from .models import HpcConfig, Project


def _slugify(name: str) -> str:
    """Convert a project name to a filesystem-safe slug."""
    slug = name.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = slug.strip("-")
    return slug


class ProjectManager:
    """Manages project creation, loading, saving, and listing."""

    def __init__(
        self,
        projects_dir: Optional[Path] = None,
        config_path: Optional[Path] = None,
    ) -> None:
        self.projects_dir = Path(projects_dir) if projects_dir else Path.home() / ".uorca" / "projects"
        self.config_path = Path(config_path) if config_path else Path.home() / ".uorca" / "config.yaml"
        self.projects_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # User defaults
    # ------------------------------------------------------------------

    def get_user_defaults(self) -> HpcConfig:
        """Load HPC defaults from the user config file, if present."""
        if not self.config_path.exists():
            return HpcConfig()
        with self.config_path.open("r") as f:
            data = yaml.safe_load(f) or {}
        hpc_data = data.get("hpc", {})
        return HpcConfig.from_dict(hpc_data)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create_project(self, name: str, description: str = "") -> Project:
        """Create a new project, persisting it to disk immediately."""
        slug = _slugify(name)
        project_dir = self.projects_dir / slug
        project_dir.mkdir(parents=True, exist_ok=True)

        # Apply user defaults for HPC config
        hpc_defaults = self.get_user_defaults()

        project = Project(
            name=name,
            description=description,
            hpc=hpc_defaults,
        )
        project.save(project_dir / "project.yaml")
        return project

    def load_project(self, slug: str) -> Project:
        """Load a project by its slug."""
        yaml_path = self.projects_dir / slug / "project.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(f"Project '{slug}' not found at {yaml_path}")
        return Project.load(yaml_path)

    def save_project(self, slug: str, project: Project) -> None:
        """Persist an existing project back to disk."""
        project_dir = self.projects_dir / slug
        project_dir.mkdir(parents=True, exist_ok=True)
        project.save(project_dir / "project.yaml")

    def get_project_slug(self, name: str) -> str:
        """Return the filesystem-safe slug for a project name."""
        return _slugify(name)

    def list_projects(self) -> list[Project]:
        """Return all projects found in the projects directory."""
        projects: list[Project] = []
        for yaml_path in sorted(self.projects_dir.glob("*/project.yaml")):
            try:
                projects.append(Project.load(yaml_path))
            except Exception:
                # Skip malformed project files
                pass
        return projects
