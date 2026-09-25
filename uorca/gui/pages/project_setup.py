"""Project Setup page — create and edit UORCA projects."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

import streamlit as st

from uorca.ai_provider import (
    DEFAULT_OPENAI_MODEL,
    _load_ai_config,
    get_model_name,
    save_ai_config,
)
from uorca.gui.project.manager import ProjectManager
from uorca.gui.project.models import HpcConfig, Project, SlurmConfig


def _set_active_project(mgr: ProjectManager, project: Project) -> None:
    """Set a project as active in session state."""
    slug = mgr.get_project_slug(project.name)
    st.session_state["active_project"] = project
    st.session_state["active_project_name"] = project.name
    st.session_state["active_project_slug"] = slug


def _render_hpc_fields(
    prefix: str,
    defaults: HpcConfig,
) -> HpcConfig:
    """Render HPC config fields and return the filled HpcConfig.

    prefix is used to make widget keys unique between create/edit forms.
    """
    st.subheader("HPC Configuration")
    col1, col2 = st.columns(2)
    with col1:
        host_alias = st.text_input(
            "Host Alias",
            value=defaults.host_alias,
            key=f"{prefix}_host_alias",
            help="SSH alias for the HPC cluster (e.g. 'sherlock')",
        )
        remote_working_dir = st.text_input(
            "Remote Working Directory",
            value=defaults.remote_working_dir,
            key=f"{prefix}_remote_working_dir",
            help="Absolute path on the HPC where runs will be staged",
        )
    with col2:
        container_image = st.text_input(
            "Container Image",
            value=defaults.container_image,
            key=f"{prefix}_container_image",
            help="Singularity/Apptainer image path or Docker URL",
        )
        resource_dir = st.text_input(
            "Resource Directory",
            value=defaults.resource_dir,
            key=f"{prefix}_resource_dir",
            help="Path to reference genome indices and annotation files on HPC",
        )

    with st.expander("SLURM Settings", expanded=False):
        s = defaults.slurm
        scol1, scol2, scol3 = st.columns(3)
        with scol1:
            partition = st.text_input(
                "Partition",
                value=s.partition,
                key=f"{prefix}_partition",
                help="SLURM partition name (leave blank to use cluster default)",
            )
            constraint = st.text_input(
                "Constraint",
                value=s.constraint,
                key=f"{prefix}_constraint",
                help="SLURM --constraint flag value (optional)",
            )
        with scol2:
            cpus_per_task = st.number_input(
                "CPUs per Task",
                min_value=1,
                max_value=128,
                value=s.cpus_per_task,
                key=f"{prefix}_cpus_per_task",
            )
            memory = st.text_input(
                "Memory",
                value=s.memory,
                key=f"{prefix}_memory",
                help="e.g. 16G, 32000M",
            )
        with scol3:
            time_limit = st.text_input(
                "Time Limit",
                value=s.time_limit,
                key=f"{prefix}_time_limit",
                help="e.g. 6:00:00",
            )
            max_parallel = st.number_input(
                "Max Parallel Jobs",
                min_value=1,
                max_value=100,
                value=s.max_parallel,
                key=f"{prefix}_max_parallel",
            )

    slurm = SlurmConfig(
        partition=partition,
        constraint=constraint,
        cpus_per_task=int(cpus_per_task),
        memory=memory,
        time_limit=time_limit,
        max_parallel=int(max_parallel),
    )
    hpc = HpcConfig(
        host_alias=host_alias,
        container_image=container_image,
        resource_dir=resource_dir,
        remote_working_dir=remote_working_dir,
        slurm=slurm,
    )
    return hpc


def _render_create_form(mgr: ProjectManager) -> None:
    """Render the Create New Project form."""
    st.subheader("Create New Project")
    defaults = mgr.get_user_defaults()

    with st.form("create_project_form"):
        name = st.text_input(
            "Project Name *",
            placeholder="e.g. Heart Failure RNA-seq",
            help="Required. Used as the project identifier.",
        )
        description = st.text_area(
            "Description",
            placeholder="Brief description of the research goal.",
            height=80,
        )
        default_rq = st.text_area(
            "Default Research Question",
            placeholder="e.g. What genes are differentially expressed in heart failure?",
            height=80,
            help="Pre-fills the research question when running dataset identification.",
        )

        hpc = _render_hpc_fields("create", defaults)

        submitted = st.form_submit_button("Create Project", type="primary")

    if submitted:
        if not name.strip():
            st.error("Project Name is required.")
            return

        # Check for duplicate
        existing = [p.name for p in mgr.list_projects()]
        if name.strip() in existing:
            st.error(f"A project named '{name.strip()}' already exists. Choose a different name.")
            return

        project = Project(
            name=name.strip(),
            description=description.strip(),
            default_research_question=default_rq.strip(),
            hpc=hpc,
        )
        slug = mgr.get_project_slug(project.name)
        mgr.save_project(slug, project)
        _set_active_project(mgr, project)
        st.success(f"Project **{project.name}** created and set as active!")


def _render_edit_form(mgr: ProjectManager) -> None:
    """Render the Edit Existing Project form."""
    projects = mgr.list_projects()
    if not projects:
        st.info("No projects found. Switch to 'Create New Project' to make one.")
        return

    project_names = [p.name for p in projects]
    active_name = st.session_state.get("active_project_name")
    default_index = project_names.index(active_name) if active_name in project_names else 0

    selected_name = st.selectbox("Select Project to Edit", project_names, index=default_index)
    project: Optional[Project] = None
    for p in projects:
        if p.name == selected_name:
            project = p
            break

    if project is None:
        return

    st.subheader(f"Editing: {project.name}")

    with st.form("edit_project_form"):
        description = st.text_area(
            "Description",
            value=project.description,
            height=80,
        )
        default_rq = st.text_area(
            "Default Research Question",
            value=project.default_research_question,
            height=80,
        )

        hpc = _render_hpc_fields("edit", project.hpc)

        saved = st.form_submit_button("Save Changes", type="primary")

    if saved:
        project.description = description.strip()
        project.default_research_question = default_rq.strip()
        project.hpc = hpc
        project.updated = datetime.now().isoformat()
        slug = mgr.get_project_slug(project.name)
        mgr.save_project(slug, project)
        # Refresh active project if this is the one being edited
        if st.session_state.get("active_project_name") == project.name:
            _set_active_project(mgr, project)
        st.success(f"Project **{project.name}** saved!")

    # Run history
    if project.runs:
        with st.expander(f"Run History ({len(project.runs)} runs)", expanded=False):
            for run in sorted(project.runs, key=lambda r: r.timestamp or "", reverse=True):
                ts = run.timestamp or "—"
                st.markdown(
                    f"- **{run.id}** &nbsp; `{run.type}` &nbsp; `{run.status}` &nbsp; {ts[:19]}"
                )
    else:
        st.caption("No runs recorded for this project yet.")


def _render_user_defaults(mgr: ProjectManager) -> None:
    """Render the User Defaults section."""
    st.divider()
    st.subheader("User Defaults")
    st.caption(
        "These are the default HPC settings applied when creating new projects. "
        "Edit the config file to change them."
    )
    defaults = mgr.get_user_defaults()
    st.code(json.dumps(defaults.to_dict(), indent=2), language="json")
    st.caption(f"Config file: `{mgr.config_path}`")

    _render_ai_provider_section()


def _render_ai_provider_section() -> None:
    """Render the AI Provider configuration section."""
    st.divider()
    st.subheader("AI Provider")

    # Load current config to pre-populate fields
    current_config = _load_ai_config()
    ai_cfg = current_config.get("ai_provider", {})
    current_provider = ai_cfg.get("provider", "openai")
    openai_cfg = ai_cfg.get("openai", {})
    bedrock_cfg = ai_cfg.get("bedrock", {})

    provider = st.radio(
        "LLM Provider",
        ["openai", "bedrock"],
        index=0 if current_provider != "bedrock" else 1,
        horizontal=True,
    )

    if provider == "openai":
        openai_model = st.text_input(
            "Model Name",
            value=openai_cfg.get("model", DEFAULT_OPENAI_MODEL),
            key="ai_openai_model",
        )
        st.caption("Requires OPENAI_API_KEY environment variable.")

        if st.button("Save AI Config", key="save_ai_config_openai"):
            save_ai_config(provider="openai", openai_model=openai_model)
            st.success("AI provider config saved.")

    else:  # bedrock
        col1, col2 = st.columns(2)
        with col1:
            bedrock_model = st.text_input(
                "Model ID",
                value=bedrock_cfg.get("model", "anthropic.claude-sonnet-4-5-20250929-v1:0"),
                key="ai_bedrock_model",
            )
            bedrock_region = st.text_input(
                "Region",
                value=bedrock_cfg.get("region", "ap-southeast-2"),
                key="ai_bedrock_region",
            )
        with col2:
            bedrock_prefix = st.text_input(
                "Model Prefix",
                value=bedrock_cfg.get("model_prefix", "au."),
                key="ai_bedrock_prefix",
            )
            bedrock_profile = st.text_input(
                "AWS Profile",
                value=bedrock_cfg.get("profile", ""),
                key="ai_bedrock_profile",
            )
        st.caption("Requires AWS SSO login.")

        if st.button("Save AI Config", key="save_ai_config_bedrock"):
            save_ai_config(
                provider="bedrock",
                bedrock_config={
                    "model": bedrock_model,
                    "region": bedrock_region,
                    "model_prefix": bedrock_prefix,
                    "profile": bedrock_profile or None,
                },
            )
            st.success("AI provider config saved.")

    st.caption(f"Current model: {get_model_name()}")


def page() -> None:
    st.title("Project Setup")

    mgr = ProjectManager()

    mode = st.radio(
        "Action",
        ["Create New Project", "Edit Existing Project"],
        horizontal=True,
        label_visibility="collapsed",
    )

    st.divider()

    if mode == "Create New Project":
        _render_create_form(mgr)
    else:
        _render_edit_form(mgr)

    _render_user_defaults(mgr)
