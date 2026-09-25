"""Streamlit component for the Identify > My Datasets tab.

Renders a registry table of project.custom_datasets and an add/edit form.
Validation is delegated to uorca.gui.hpc.custom_dataset_validator.
"""

from __future__ import annotations

import re
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from uorca.gui.hpc.custom_dataset_validator import (
    remote_file_exists,
    stage_metadata_csv,
    validate_fastq_dir,
)
from uorca.gui.hpc.ssh_manager import close as ssh_close, connect as ssh_connect
from uorca.gui.project.manager import ProjectManager
from uorca.gui.project.models import CustomDataset, Project


_STATUS_BADGE = {
    "valid": "🟢 valid",
    "unvalidated": "🟡 unvalidated",
    "missing": "🔴 missing",
    "invalid": "🔴 invalid",
}


def _slugify(label: str) -> str:
    """Match uorca.gui.project.manager._slugify."""
    s = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
    return s


def _save_project(project: Project) -> None:
    mgr = ProjectManager()
    slug = mgr.get_project_slug(project.name)
    mgr.save_project(slug, project)
    st.session_state["active_project"] = project


def _render_registry_table(project: Project) -> None:
    if not project.custom_datasets:
        st.info("No custom datasets registered. Add one below.")
        return

    rows = []
    for cd in project.custom_datasets:
        rows.append({
            "Label": cd.label,
            "Status": _STATUS_BADGE.get(cd.validation_status, cd.validation_status),
            "FASTQ directory": cd.fastq_dir,
            "Organism": cd.organism or "let AI infer",
            "Last validated": cd.last_validated or "—",
            "Slug": cd.id,  # used for action keys; hidden via column_config below
        })

    df = pd.DataFrame(rows)
    st.dataframe(
        df.drop(columns=["Slug"]),
        use_container_width=True,
        hide_index=True,
    )

    st.caption("Per-row actions:")
    for cd in project.custom_datasets:
        c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
        with c1:
            st.markdown(f"**{cd.label}** &nbsp;·&nbsp; `{cd.id}`")
        with c2:
            if st.button("Re-validate", key=f"reval_{cd.id}"):
                _revalidate_one(project, cd)
        with c3:
            if st.button("Edit", key=f"edit_{cd.id}"):
                st.session_state["custom_dataset_edit_id"] = cd.id
                st.rerun()
        with c4:
            if st.button("Remove", key=f"remove_{cd.id}", type="secondary"):
                _remove_one(project, cd)


def render_my_datasets(project: Project) -> None:
    """Public entry point — called from uorca/gui/pages/identify.py."""
    st.subheader("Registered custom datasets")
    _render_registry_table(project)
    st.divider()
    _render_add_form(project)


def _render_add_form(project: Project) -> None:
    """Form to register a new custom dataset (or edit an existing one)."""

    edit_id: Optional[str] = st.session_state.get("custom_dataset_edit_id")
    editing: Optional[CustomDataset] = None
    if edit_id:
        editing = next(
            (d for d in project.custom_datasets if d.id == edit_id), None
        )

    title = "Edit custom dataset" if editing else "➕ Add a custom dataset"

    with st.expander(title, expanded=editing is not None):
        with st.form("custom_dataset_form"):
            label = st.text_input(
                "Label *",
                value=editing.label if editing else "",
                placeholder="e.g. GATA4 cohort (n=16)",
            )
            generated_slug = _slugify(label) if label else ""
            st.caption(f"Slug: `{generated_slug or '(empty)'}`")

            fastq_dir = st.text_input(
                "FASTQ directory on HPC *",
                value=editing.fastq_dir if editing else "",
                placeholder="/data/.../my_project/fastqs",
                help="Must contain paired-end *.fastq.gz files. Single-end is not supported.",
            )

            uploaded = st.file_uploader(
                "Metadata CSV *" if not editing else "Replace metadata CSV (optional)",
                type=["csv"],
            )

            org_choices = [
                "Let AI infer",
                "Homo sapiens",
                "Mus musculus",
                "Canis familiaris",
                "Macaca mulatta",
                "Danio rerio",
            ]
            org_default = editing.organism if (editing and editing.organism) else "Let AI infer"
            organism = st.selectbox(
                "Organism",
                org_choices,
                index=org_choices.index(org_default) if org_default in org_choices else 0,
            )

            description = st.text_input(
                "Description (optional)",
                value=editing.description or "" if editing else "",
                placeholder="GATA4 KO vs WT, iPSC-derived cardiomyocytes",
            )

            password = st.text_input(
                "HPC password",
                type="password",
                value=st.session_state.get("identify_hpc_password", ""),
                help="Used to validate the FASTQ directory and stage the metadata CSV.",
            )

            col_submit, col_cancel = st.columns([1, 1])
            with col_submit:
                submitted = st.form_submit_button(
                    "Validate & Save" if editing else "Validate & Add",
                    type="primary",
                    use_container_width=True,
                )
            with col_cancel:
                cancelled = st.form_submit_button(
                    "Cancel", use_container_width=True
                )

        if cancelled:
            st.session_state.pop("custom_dataset_edit_id", None)
            st.rerun()

        if submitted:
            _handle_add_or_edit_submit(
                project=project,
                editing=editing,
                label=label.strip(),
                fastq_dir=fastq_dir.strip(),
                uploaded=uploaded,
                organism=None if organism == "Let AI infer" else organism,
                description=description.strip() or None,
                password=password,
            )


def _handle_add_or_edit_submit(
    *,
    project: Project,
    editing: Optional[CustomDataset],
    label: str,
    fastq_dir: str,
    uploaded,  # streamlit UploadedFile or None
    organism: Optional[str],
    description: Optional[str],
    password: Optional[str],
) -> None:
    """Validate inputs, run SSH check, stage CSV, persist project YAML."""

    # 1. Client-side validation — surface errors loudly.
    if not label:
        st.error("Label is required.")
        return
    if not fastq_dir:
        st.error("FASTQ directory is required.")
        return
    if not editing and uploaded is None:
        st.error("Metadata CSV is required.")
        return
    if not project.hpc.host_alias:
        st.error("HPC host_alias is not set on this project. Configure it in Project Setup.")
        return
    if not project.hpc.remote_working_dir:
        st.error("HPC remote_working_dir is not set on this project. Configure it in Project Setup.")
        return

    new_id = _slugify(label)
    if not new_id:
        st.error("Label produces an empty slug. Use alphanumeric characters.")
        return

    # 2. Slug uniqueness — except when editing the same row.
    for existing in project.custom_datasets:
        if existing.id == new_id and (editing is None or editing.id != new_id):
            st.error(
                f"A custom dataset with slug `{new_id}` already exists. "
                "Use a different label."
            )
            return

    # 3. SSH validation + SFTP staging.
    mgr = ProjectManager()
    project_slug = mgr.get_project_slug(project.name)
    metadata_filename = f"{new_id}.csv"
    remote_dir = f"{project.hpc.remote_working_dir}/scratch/custom_metadata/{project_slug}"
    remote_csv_path = f"{remote_dir}/{metadata_filename}"

    client = None
    try:
        client = ssh_connect(project.hpc.host_alias, password=password or None)
    except Exception as e:
        st.error(f"SSH connection failed: {e}")
        return

    try:
        st.session_state["identify_hpc_password"] = password

        status, message = validate_fastq_dir(client, fastq_dir)
        if status != "valid":
            st.error(message)
            return

        if uploaded is not None:
            try:
                stage_metadata_csv(
                    client, BytesIO(uploaded.getvalue()), remote_csv_path
                )
            except Exception as e:
                st.error(f"Metadata CSV upload failed: {e}")
                return
        elif editing is not None:
            # Editing without re-uploading — verify the previous CSV still exists.
            existing_path = f"{remote_dir}/{editing.metadata_filename}"
            if not remote_file_exists(client, existing_path):
                st.error(
                    f"Existing metadata CSV is missing on HPC at {existing_path}. "
                    "Please re-upload."
                )
                return
            metadata_filename = editing.metadata_filename
    finally:
        if client is not None:
            try:
                ssh_close(client)
            except Exception:
                pass

    # 4. Persist to the project YAML.
    now = datetime.now().isoformat()
    if editing is None:
        cd = CustomDataset(
            id=new_id,
            label=label,
            fastq_dir=fastq_dir,
            metadata_filename=metadata_filename,
            organism=organism,
            description=description,
            created=now,
            last_validated=now,
            validation_status="valid",
            validation_message=None,
        )
        project.custom_datasets.append(cd)
    else:
        editing.label = label
        editing.fastq_dir = fastq_dir
        editing.metadata_filename = metadata_filename
        editing.organism = organism
        editing.description = description
        editing.last_validated = now
        editing.validation_status = "valid"
        editing.validation_message = None

    project.updated = now
    _save_project(project)
    st.session_state.pop("custom_dataset_edit_id", None)
    st.success(f"{'Saved' if editing else 'Added'} custom dataset `{new_id}`.")
    st.rerun()


def _revalidate_one(project: Project, cd: CustomDataset) -> None:
    if not project.hpc.host_alias:
        st.error("HPC host_alias is not set on this project.")
        return
    password = st.session_state.get("identify_hpc_password", "") or st.session_state.get(
        "run_hpc_password", ""
    )
    if not password:
        st.error(
            "Re-validation needs an HPC password. Open the Add form to enter one, "
            "or visit the Run page (which caches it for the session)."
        )
        return

    mgr = ProjectManager()
    project_slug = mgr.get_project_slug(project.name)
    remote_csv = (
        f"{project.hpc.remote_working_dir}/scratch/custom_metadata/"
        f"{project_slug}/{cd.metadata_filename}"
    )

    try:
        client = ssh_connect(project.hpc.host_alias, password=password)
    except Exception as e:
        st.error(f"SSH connection failed: {e}")
        return

    try:
        status, message = validate_fastq_dir(client, cd.fastq_dir)
        csv_present = remote_file_exists(client, remote_csv)
    finally:
        try:
            ssh_close(client)
        except Exception:
            pass

    now = datetime.now().isoformat()
    if status == "valid" and csv_present:
        cd.validation_status = "valid"
        cd.validation_message = None
        cd.last_validated = now
        st.success(f"Re-validated `{cd.id}` — OK.")
    elif status == "valid" and not csv_present:
        cd.validation_status = "invalid"
        cd.validation_message = f"Staged metadata CSV is missing at {remote_csv}"
        cd.last_validated = now
        st.error(cd.validation_message)
    else:
        cd.validation_status = status
        cd.validation_message = message
        cd.last_validated = now
        st.error(message or "Re-validation failed.")

    project.updated = now
    _save_project(project)
    st.rerun()


def _remove_one(project: Project, cd: CustomDataset) -> None:
    """Remove from registry; best-effort SFTP delete of staged CSV."""
    mgr = ProjectManager()
    project_slug = mgr.get_project_slug(project.name)
    remote_csv = (
        f"{project.hpc.remote_working_dir}/scratch/custom_metadata/"
        f"{project_slug}/{cd.metadata_filename}"
    )
    password = st.session_state.get("identify_hpc_password", "") or st.session_state.get(
        "run_hpc_password", ""
    )

    if password and project.hpc.host_alias:
        try:
            client = ssh_connect(project.hpc.host_alias, password=password)
            try:
                sftp = client.open_sftp()
                try:
                    sftp.remove(remote_csv)
                except IOError as e:
                    st.warning(
                        f"Could not delete staged CSV at {remote_csv}: {e}. "
                        "Removed from registry anyway; orphan left on HPC."
                    )
                finally:
                    sftp.close()
            finally:
                ssh_close(client)
        except Exception as e:
            st.warning(f"SSH cleanup failed: {e}. Removed from registry anyway.")
    else:
        st.warning(
            "No HPC password cached — staged CSV not removed from HPC. "
            "Removed from project registry."
        )

    project.custom_datasets = [
        d for d in project.custom_datasets if d.id != cd.id
    ]
    project.updated = datetime.now().isoformat()
    _save_project(project)
    st.success(f"Removed custom dataset `{cd.id}`.")
    st.rerun()
