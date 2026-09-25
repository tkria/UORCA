"""Run Pipeline page — submit and monitor HPC SLURM jobs."""

from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from uorca.gui.project.manager import ProjectManager
from uorca.gui.project.models import Project, RunEntry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_project() -> Project | None:
    return st.session_state.get("active_project")


def _get_project_slug(project: Project) -> str:
    mgr = ProjectManager()
    return mgr.get_project_slug(project.name)


def _save_project(project: Project) -> None:
    mgr = ProjectManager()
    slug = mgr.get_project_slug(project.name)
    mgr.save_project(slug, project)
    st.session_state["active_project"] = project


def _hpc_configured(project: Project) -> bool:
    """Return True if the project has a non-empty host_alias."""
    return bool(project.hpc.host_alias.strip())


def _get_ssh_username(host_alias: str) -> str:
    """Extract the username for *host_alias* from the local SSH config."""
    try:
        result = subprocess.run(
            ["ssh", "-G", host_alias],
            capture_output=True,
            text=True,
            timeout=10,
        )
        for line in result.stdout.splitlines():
            if line.startswith("user "):
                return line.split()[1]
    except Exception:
        pass
    return ""


def _load_csv_accessions(df: pd.DataFrame) -> list[str]:
    """Extract accession list from a dataframe.  Looks for an 'Accession' column."""
    if "Accession" in df.columns:
        return [str(v) for v in df["Accession"].dropna().unique()]
    # Fallback: first column
    return [str(v) for v in df.iloc[:, 0].dropna().unique()]


# ---------------------------------------------------------------------------
# Job submission
# ---------------------------------------------------------------------------


def _make_screen_name(project: Project) -> str:
    """Generate a named screen session identifier for this project."""
    date_str = datetime.now().strftime("%Y%m%d")
    slug = project.name.replace(" ", "_")
    return f"uorca_{slug}_{date_str}"


def _submit_jobs(
    project: Project,
    datasets: list[str],
    custom_ids: list[str],
    password: str | None,
) -> tuple[bool, str, str]:
    """Submit a mixed SLURM pipeline run via a named screen session.

    `datasets` is the list of GEO accessions (may be empty).
    `custom_ids` is the list of custom dataset IDs to include (may be empty).

    Returns (success, message, results_dir).
    """
    if not datasets and not custom_ids:
        return False, "Nothing queued for submission.", ""

    try:
        from uorca.gui.hpc.ssh_manager import close, connect, run_command
    except ImportError as exc:
        return False, f"HPC module unavailable: {exc}", ""

    hpc = project.hpc
    date_str = datetime.now().strftime("%Y%m%d")
    project_slug_for_path = project.name.replace(" ", "_")
    run_name = f"{date_str}_{project_slug_for_path}"
    screen_name = _make_screen_name(project)

    mgr = ProjectManager()
    project_slug = mgr.get_project_slug(project.name)

    remote_geo_csv = f"{hpc.remote_working_dir}/scratch/_gui_submit_{run_name}.csv"
    results_dir = f"{hpc.remote_working_dir}/results/{run_name}"

    # Build per-dataset user-config YAMLs.
    custom_lookup = {cd.id: cd for cd in project.custom_datasets}
    user_yaml_paths: list[tuple[str, dict]] = []
    for cid in custom_ids:
        cd = custom_lookup.get(cid)
        if cd is None:
            return False, f"Selected custom dataset `{cid}` is no longer in the registry.", ""
        if cd.validation_status in ("missing", "invalid"):
            return False, (
                f"Custom dataset `{cid}` has status {cd.validation_status} — "
                "fix it in Identify > My Datasets before submitting."
            ), ""

        remote_csv_path = (
            f"{hpc.remote_working_dir}/scratch/custom_metadata/"
            f"{project_slug}/{cd.metadata_filename}"
        )
        yaml_body: dict = {
            "mode": "user",
            "user_data": {
                "fastq_dir": cd.fastq_dir,
                "metadata_path": remote_csv_path,
            },
            "output_dir": results_dir,
            "resource_dir": hpc.resource_dir,
        }
        if cd.organism:
            yaml_body["user_data"]["organism"] = cd.organism
        if cd.description:
            yaml_body["user_data"]["description"] = cd.description

        user_yaml_paths.append(
            (
                f"{hpc.remote_working_dir}/scratch/_gui_submit_{run_name}__user_{cd.id}.yaml",
                yaml_body,
            )
        )

    client = None
    try:
        client = connect(hpc.host_alias, password=password or None)

        # Duplicate-protection
        _, screen_ls, _ = run_command(client, f"screen -ls | grep {screen_name} || true", timeout=10)
        if screen_name in screen_ls:
            return False, (
                f"A submission is already running for this project (screen: {screen_name}). "
                "Use Refresh Status to monitor."
            ), results_dir

        # Ensure directories exist
        run_command(client, f"mkdir -p {hpc.remote_working_dir}/scratch {results_dir}")

        # Write GEO CSV
        if datasets:
            csv_lines = "Accession,organism,RelevanceScore,Valid,DatasetSizeGB"
            for acc in datasets:
                csv_lines += f"\n{acc},Homo sapiens,0,Yes,0"
            run_command(client, f'printf "%s" "{csv_lines}" > {remote_geo_csv}')

        # Write each user-config YAML
        import yaml as _yaml
        for path, body in user_yaml_paths:
            yaml_str = _yaml.safe_dump(body, default_flow_style=False)
            # heredoc to avoid quoting headaches
            cmd = f"cat > {path} <<'__YAML_EOF__'\n{yaml_str}__YAML_EOF__\n"
            run_command(client, cmd, timeout=15)

        # Build a per-run slurm_config.yaml that overrides the project's SLURM
        # settings (partition, constraint, cpus, memory, time_limit,
        # max_parallel) on top of the cluster's checked-in defaults
        # (container image, max_storage_gb, ...). Pass this via --config so the
        # job actually uses what the user typed in Project Setup, not whatever
        # the host's slurm_config.yaml happened to have.
        _, base_yaml_str, _ = run_command(
            client,
            f"cat {hpc.remote_working_dir}/slurm_config.yaml 2>/dev/null || true",
            timeout=10,
        )
        try:
            base_cfg = _yaml.safe_load(base_yaml_str) or {}
        except Exception:
            base_cfg = {}
        if not isinstance(base_cfg, dict):
            base_cfg = {}

        ps = hpc.slurm
        slurm_section = base_cfg.setdefault("slurm", {})
        slurm_section["partition"] = ps.partition or slurm_section.get("partition", "")
        # Project's constraint wins when set; an empty project value falls
        # through to the host default (typing a blank should not silently
        # remove a cluster-default constraint — re-empty by removing the
        # value from the host config instead).
        if ps.constraint:
            slurm_section["constraint"] = ps.constraint
        slurm_section["cpus_per_task"] = ps.cpus_per_task
        slurm_section["memory"] = ps.memory
        slurm_section["time_limit"] = ps.time_limit
        rm_section = base_cfg.setdefault("resource_management", {})
        rm_section["max_parallel"] = ps.max_parallel

        merged_yaml = _yaml.safe_dump(base_cfg, default_flow_style=False)
        remote_run_config = (
            f"{hpc.remote_working_dir}/scratch/_gui_submit_{run_name}__slurm_config.yaml"
        )
        run_command(
            client,
            f"cat > {remote_run_config} <<'__YAML_EOF__'\n{merged_yaml}__YAML_EOF__\n",
            timeout=15,
        )

        # Build the multi-flag submit command (preamble joined with &&;
        # uorca arguments joined with spaces).
        preamble = f"cd {hpc.remote_working_dir} && source .env && "
        cmd_parts: list[str] = ["uv run uorca run slurm"]
        if datasets:
            cmd_parts.append(f"--input {remote_geo_csv}")
        for path, _ in user_yaml_paths:
            cmd_parts.append(f"--user-config {path}")
        cmd_parts.extend([
            f"--output_dir {results_dir}",
            f"--resource_dir {hpc.resource_dir}",
            f"--config {remote_run_config}",
            f"--max_parallel {hpc.slurm.max_parallel}",
        ])
        submit_cmd = preamble + " ".join(cmd_parts)

        screen_cmd = (
            f'screen -dmS {screen_name} bash -c '
            f'"{submit_cmd} > {results_dir}/submission.log 2>&1"'
        )
        exit_code, stdout, stderr = run_command(client, screen_cmd, timeout=15)
        if exit_code != 0:
            return False, f"Failed to start screen session: {stderr}", results_dir

        import time
        time.sleep(1)
        _, verify, _ = run_command(client, f"screen -ls | grep {screen_name} || true", timeout=10)
        if screen_name in verify:
            return True, (
                f"Submission started in screen session `{screen_name}` on {hpc.host_alias}. "
                f"GEO accessions: {len(datasets)}; custom datasets: {len(user_yaml_paths)}."
            ), results_dir

        # Screen exited fast — surface log tail rather than swallow.
        _, log_head, _ = run_command(
            client, f"head -20 {results_dir}/submission.log 2>/dev/null", timeout=10
        )
        return False, (
            f"Screen session exited unexpectedly. submission.log head:\n{log_head[:600]}"
        ), results_dir

    except Exception as exc:
        return False, f"SSH error: {exc}", ""
    finally:
        if client is not None:
            try:
                close(client)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Left panel — Submit
# ---------------------------------------------------------------------------


def _render_left_panel(project: Project) -> None:
    # Project context banner
    hpc = project.hpc
    hpc_ok = _hpc_configured(project)

    st.info(
        f"**Project:** {project.name}  \n"
        f"**HPC Host:** {hpc.host_alias or '*(not configured)*'}"
    )

    if not hpc_ok:
        st.warning(
            "HPC is not configured for this project. "
            "Please set a Host Alias in Project Setup before submitting jobs."
        )
        if st.button("Go to Project Setup", key="goto_setup"):
            st.switch_page(st.session_state["_pages"]["project-setup"])
        return

    # ------------------------------------------------------------------
    # Input source selection
    # ------------------------------------------------------------------
    st.subheader("Input Datasets")

    input_source = st.radio(
        "Input source",
        ["From identification run", "Upload CSV"],
        key="run_input_source",
        horizontal=True,
    )

    datasets: list[str] = []

    if input_source == "From identification run":
        # Offer a dropdown of previous identification runs
        id_runs = project.get_runs_by_type("identification")
        if not id_runs:
            st.info("No identification runs found. Run dataset identification first, or upload a CSV.")
        else:
            # Check for datasets forwarded from the Identify page (session state)
            pending: list[str] | None = st.session_state.get("pending_datasets")

            run_labels = [f"{r.id} ({r.status})" for r in reversed(id_runs)]
            selected_label = st.selectbox(
                "Select identification run",
                run_labels,
                key="run_id_run_select",
            )
            selected_run = list(reversed(id_runs))[run_labels.index(selected_label)]

            if pending:
                # Forwarded from Identify page this session
                st.success(f"{len(pending)} dataset(s) selected from Identify page.")
                datasets = pending
            elif selected_run.datasets:
                # Curated selection saved in the identification run
                datasets = selected_run.datasets
                st.info(f"{len(datasets)} dataset(s) from curated selection.")
            else:
                # No curated selection — show warning
                st.warning(
                    "No curated dataset selection found for this run. "
                    "Go to the **Identify** page and select datasets before submitting."
                )

    else:  # Upload CSV
        uploaded = st.file_uploader(
            "Upload CSV (must have an Accession column)",
            type=["csv"],
            key="run_csv_upload",
        )
        if uploaded is not None:
            try:
                df_up = pd.read_csv(uploaded)
                datasets = _load_csv_accessions(df_up)
            except Exception as exc:
                st.error(f"Could not parse CSV: {exc}")

    # ------------------------------------------------------------------
    # Dataset preview
    # ------------------------------------------------------------------
    if datasets:
        st.caption(f"**{len(datasets)} accession(s) queued:**")
        preview = datasets[:20]
        st.code(", ".join(preview) + ("..." if len(datasets) > 20 else ""))
    else:
        st.caption("No datasets selected.")

    # Custom datasets selection
    ticked_custom_ids = _render_custom_datasets_section(project)

    # Combined queue summary
    n_geo = len(datasets)
    n_custom = len(ticked_custom_ids)
    if n_geo + n_custom > 0:
        st.info(
            f"📦 **{n_geo} GEO accession(s) + {n_custom} custom dataset(s)** "
            "queued for submission."
        )
    else:
        st.warning("Nothing queued for submission.")

    # Stash for the submit handler
    st.session_state["run_ticked_custom_ids"] = ticked_custom_ids

    # ------------------------------------------------------------------
    # SLURM settings summary
    # ------------------------------------------------------------------
    with st.expander("SLURM Settings", expanded=False):
        s = hpc.slurm
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**Partition:** {s.partition or '*(default)*'}")
            st.markdown(f"**Constraint:** {getattr(s, 'constraint', '') or '*(none)*'}")
        with col2:
            st.markdown(f"**CPUs/Task:** {s.cpus_per_task}")
            st.markdown(f"**Memory:** {s.memory}")
        with col3:
            st.markdown(f"**Time Limit:** {s.time_limit}")
            st.markdown(f"**Max Parallel:** {s.max_parallel}")

    # ------------------------------------------------------------------
    # HPC Authentication
    # ------------------------------------------------------------------
    st.subheader("HPC Authentication")
    password = st.text_input(
        f"Password for `{hpc.host_alias}`",
        type="password",
        key="run_hpc_password",
        help="Leave blank to use SSH key authentication.",
    )

    # Quick connection test
    if st.button("Test Connection", key="test_ssh_conn"):
        with st.spinner("Connecting…"):
            try:
                from uorca.gui.hpc.ssh_manager import close, connect, run_command

                client = connect(hpc.host_alias, password=password or None)
                _, stdout, _ = run_command(client, "echo connected")
                close(client)
                if "connected" in stdout:
                    st.success("Connected")
                else:
                    st.warning("Connected but unexpected response.")
            except ImportError as exc:
                st.error(f"HPC module unavailable: {exc}")
            except Exception as exc:
                st.error(f"Connection failed: {exc}")

    # ------------------------------------------------------------------
    # Submit button
    # ------------------------------------------------------------------
    st.divider()
    custom_ids = st.session_state.get("run_ticked_custom_ids", [])
    n_geo = len(datasets)
    n_custom = len(custom_ids)
    n_total = n_geo + n_custom

    submit_label = (
        f"Submit {n_total} Job{'s' if n_total != 1 else ''}"
        if n_total else "Submit Jobs"
    )

    if st.button(submit_label, type="primary", disabled=(n_total == 0), key="run_submit_btn"):
        with st.spinner("Starting submission on HPC…"):
            success, message, actual_results_dir = _submit_jobs(
                project, datasets, custom_ids, password
            )

        if success:
            st.success(message)

            screen_name = _make_screen_name(project)
            now = datetime.now()
            date_str = now.strftime("%Y%m%d")
            time_str = now.strftime("%H%M%S")
            project_slug = project.name.replace(" ", "_")
            # Include time so multiple submissions on the same day produce
            # distinct RunEntry ids — otherwise Streamlit raises
            # StreamlitDuplicateElementKey on the per-run Cancel button and
            # the per-run dataset_status keys collide silently.
            run_id = f"pipeline_{date_str}_{time_str}_{project_slug}"
            entry = RunEntry(
                id=run_id,
                type="pipeline",
                status="running",
                output_path=actual_results_dir,
                timestamp=datetime.now().isoformat(),
                datasets=datasets,
                custom_dataset_ids=custom_ids,
                params={
                    "host_alias": hpc.host_alias,
                    "max_parallel": hpc.slurm.max_parallel,
                    "n_datasets": n_geo,
                    "n_custom_datasets": n_custom,
                    "screen_name": screen_name,
                },
            )
            project.runs.append(entry)
            project.updated = datetime.now().isoformat()
            _save_project(project)

            # Clear forwarded datasets
            st.session_state.pop("pending_datasets", None)
            st.rerun()
        else:
            st.error(f"Submission failed:\n\n{message}")


# ---------------------------------------------------------------------------
# Custom datasets section
# ---------------------------------------------------------------------------


def _render_custom_datasets_section(project: Project) -> list[str]:
    """Render the Custom datasets selection section.

    Returns the list of dataset IDs the user has ticked. Empty list if no
    customs are registered (in which case the section is hidden entirely).
    """
    if not project.custom_datasets:
        return []

    st.subheader("Custom datasets")
    st.caption("Tick which registered custom datasets to include in this submission.")

    ticked: list[str] = []
    for cd in project.custom_datasets:
        disabled = cd.validation_status in ("missing", "invalid")

        cols = st.columns([1, 8, 2])
        with cols[0]:
            checked = st.checkbox(
                f"Include {cd.label}",
                value=False,
                key=f"run_custom_check_{cd.id}",
                disabled=disabled,
                label_visibility="collapsed",
            )
        with cols[1]:
            st.markdown(f"**{cd.label}**")
            org_label = cd.organism or "let AI infer"
            st.caption(f"`{cd.fastq_dir}` · {org_label}")
            if disabled and cd.validation_message:
                st.caption(f"⚠ {cd.validation_message} — fix in Identify > My Datasets")
        with cols[2]:
            badge = {
                "valid": "🟢 valid",
                "unvalidated": "🟡 unvalidated",
                "missing": "🔴 missing",
                "invalid": "🔴 invalid",
            }.get(cd.validation_status, cd.validation_status)
            st.markdown(badge)

        if checked and not disabled:
            ticked.append(cd.id)

    if st.button("Re-validate all", key="run_revalidate_all"):
        _revalidate_all_customs(project)

    st.divider()
    return ticked


def _revalidate_all_customs(project: Project) -> None:
    """Re-validate every registered custom dataset in one SSH session.

    Important: must NOT call the component's `_revalidate_one`, which calls
    `st.rerun()` — that would interrupt the loop. Inline the validation here
    using a single SSH connection for efficiency.
    """
    from datetime import datetime as _dt
    from uorca.gui.hpc.custom_dataset_validator import (
        remote_file_exists,
        validate_fastq_dir,
    )
    from uorca.gui.hpc.ssh_manager import close as ssh_close, connect as ssh_connect
    from uorca.gui.project.manager import ProjectManager

    if not project.hpc.host_alias:
        st.error("HPC host_alias is not set on this project.")
        return

    password = st.session_state.get("run_hpc_password", "") or st.session_state.get(
        "identify_hpc_password", ""
    )
    if not password:
        st.error(
            "Re-validation needs an HPC password. Enter it in the password field "
            "above (a Test Connection click also caches it for the session)."
        )
        return

    try:
        client = ssh_connect(project.hpc.host_alias, password=password)
    except Exception as e:
        st.error(f"SSH connection failed: {e}")
        return

    mgr = ProjectManager()
    project_slug = mgr.get_project_slug(project.name)

    try:
        now = _dt.now().isoformat()
        for cd in project.custom_datasets:
            remote_csv = (
                f"{project.hpc.remote_working_dir}/scratch/custom_metadata/"
                f"{project_slug}/{cd.metadata_filename}"
            )
            status, message = validate_fastq_dir(client, cd.fastq_dir)
            csv_present = remote_file_exists(client, remote_csv)
            if status == "valid" and csv_present:
                cd.validation_status = "valid"
                cd.validation_message = None
            elif status == "valid" and not csv_present:
                cd.validation_status = "invalid"
                cd.validation_message = (
                    f"Staged metadata CSV is missing at {remote_csv}"
                )
            else:
                cd.validation_status = status
                cd.validation_message = message
            cd.last_validated = now
    finally:
        try:
            ssh_close(client)
        except Exception:
            pass

    project.updated = _dt.now().isoformat()
    mgr.save_project(project_slug, project)
    st.session_state["active_project"] = project
    st.success("Re-validated all custom datasets.")
    st.rerun()


# ---------------------------------------------------------------------------
# Right panel — Monitor
# ---------------------------------------------------------------------------


def _render_right_panel(project: Project) -> None:
    st.subheader("Pipeline Monitor")

    hpc = project.hpc
    hpc_ok = _hpc_configured(project)
    pipeline_runs = project.get_runs_by_type("pipeline")

    if not pipeline_runs:
        st.info("No pipeline runs yet. Select datasets and submit jobs.")
        return

    # Legacy data may contain RunEntries that share an id (this happened
    # before run_ids included a time suffix). Streamlit raises
    # StreamlitDuplicateElementKey on per-run buttons / expanders if the
    # same id appears twice. Dedupe in place by suffixing later
    # occurrences with a counter; persist so subsequent loads stay clean.
    seen_ids: dict[str, int] = {}
    deduped = False
    for r in pipeline_runs:
        if r.id in seen_ids:
            seen_ids[r.id] += 1
            r.id = f"{r.id}_dup{seen_ids[r.id]}"
            deduped = True
        else:
            seen_ids[r.id] = 0
    if deduped:
        _save_project(project)

    # ------------------------------------------------------------------
    # Action buttons
    # ------------------------------------------------------------------
    if hpc_ok:
        col_refresh, col_sync, col_explore = st.columns([1, 1, 1])
        with col_refresh:
            refresh_clicked = st.button("Refresh Status", key="run_refresh_btn")
        with col_sync:
            sync_clicked = st.button("Sync Results Locally", key="run_sync_btn")
        with col_explore:
            if st.button("Explore Results →", key="run_explore_btn"):
                st.switch_page(st.session_state["_pages"]["explore"])
    else:
        refresh_clicked = False
        sync_clicked = False
        if st.button("Explore Results →", key="run_explore_btn_nohpc"):
            st.switch_page(st.session_state["_pages"]["explore"])

    # ------------------------------------------------------------------
    # Refresh: fetch per-dataset status from HPC
    # ------------------------------------------------------------------
    if refresh_clicked and hpc_ok:
        password = st.session_state.get("run_hpc_password", "")
        username = _get_ssh_username(hpc.host_alias)

        with st.spinner("Fetching job status from HPC…"):
            try:
                from uorca.gui.hpc.slurm import squeue_poll
                from uorca.gui.hpc.ssh_manager import close, connect, run_command

                client = connect(hpc.host_alias, password=password or None)
                active_jobs = squeue_poll(client, username)

                # For each running pipeline run, fetch per-dataset status
                for run in pipeline_runs:
                    if run.status not in ("running", "pending"):
                        continue
                    if not run.output_path:
                        continue

                    # Check if screen session is still running
                    screen_name = run.params.get("screen_name", "")
                    if screen_name:
                        _, screen_check, _ = run_command(
                            client, f"screen -ls | grep {screen_name} || true", timeout=10
                        )
                        screen_alive = screen_name in screen_check
                        if screen_alive:
                            st.caption(f"Submission process active (screen: `{screen_name}`)")
                        else:
                            st.caption("Submission process finished.")

                    # Read job_status JSON files from remote
                    _, status_json, _ = run_command(
                        client,
                        f'shopt -s nullglob; for f in {run.output_path}/job_status/*_status.json; do '
                        f'acc=$(basename "$f" _status.json); '
                        f'state=$(python3 -c "import json; d=json.load(open(\'$f\')); print(d.get(\'state\',\'unknown\'))" 2>/dev/null || echo unknown); '
                        f'size=$(python3 -c "import json; d=json.load(open(\'$f\')); print(d.get(\'storage_gb\',0))" 2>/dev/null || echo 0); '
                        f'echo "$acc|$state|$size"; done',
                        timeout=30,
                    )

                    dataset_statuses = {}
                    for line in status_json.strip().splitlines():
                        parts = line.strip().split("|")
                        if len(parts) >= 3:
                            dataset_statuses[parts[0]] = {
                                "state": parts[1],
                                "size_gb": parts[2],
                            }

                    # Cross-reference with active SLURM jobs
                    active_datasets = set()
                    for job in active_jobs:
                        name = job.get("name", "")
                        for acc in (run.datasets or []):
                            if acc in name:
                                active_datasets.add(acc)
                                if acc not in dataset_statuses:
                                    dataset_statuses[acc] = {"state": "running", "size_gb": "0"}
                                elif dataset_statuses[acc]["state"] == "submitted":
                                    dataset_statuses[acc]["state"] = "running"

                    # Check if all datasets are done
                    all_done = (
                        len(dataset_statuses) > 0
                        and not active_datasets
                        and all(ds["state"] in ("completed", "failed") for ds in dataset_statuses.values())
                    )

                    if all_done:
                        n_completed = sum(1 for ds in dataset_statuses.values() if ds["state"] == "completed")
                        # Check for DEG results
                        _, deg_out, _ = run_command(
                            client,
                            f"ls -d {run.output_path}/GSE*/RNAseqAnalysis/ 2>/dev/null | wc -l",
                            timeout=15,
                        )
                        n_with_degs = int(deg_out.strip()) if deg_out.strip().isdigit() else 0

                        run.status = "completed"
                        run.summary = {
                            "submitted": len(dataset_statuses),
                            "completed": n_completed,
                            "succeeded": n_with_degs,
                            "failed": len(dataset_statuses) - n_with_degs,
                        }

                    # Store per-dataset status in session state for display
                    st.session_state[f"dataset_status_{run.id}"] = dataset_statuses

                _save_project(project)
                close(client)

            except ImportError as exc:
                st.error(f"HPC module unavailable: {exc}")
            except Exception as exc:
                st.error(f"Could not fetch status: {exc}")

    # ------------------------------------------------------------------
    # Display run history with per-dataset detail
    # ------------------------------------------------------------------
    custom_lookup = {cd.id: cd for cd in (project.custom_datasets or [])}

    for run in reversed(pipeline_runs):
        status_emoji = {"completed": "✅", "running": "🔄", "failed": "❌", "pending": "⏳", "cancelled": "🚫"}.get(run.status, "❓")
        n_geo = len(run.datasets or [])
        n_custom = len(run.custom_dataset_ids or [])
        n_total = n_geo + n_custom
        run_date = (run.timestamp or "")[:10]

        if n_custom and n_geo:
            count_label = f"{n_total} datasets ({n_geo} GEO + {n_custom} custom)"
        elif n_custom:
            count_label = f"{n_custom} custom dataset(s)"
        elif n_geo:
            count_label = f"{n_geo} datasets"
        else:
            count_label = f"{run.params.get('n_datasets', '?')} datasets"

        header = f"{status_emoji} **{run.id}** — {count_label} ({run_date})"
        if run.summary:
            s = run.summary
            header += f" | {s.get('succeeded', '?')} succeeded, {s.get('failed', '?')} failed"

        with st.expander(header, expanded=(run.status == "running")):
            # Load dataset sizes from identification results if available
            size_lookup: dict[str, float] = {}
            input_run_id = run.input_from
            if input_run_id:
                for id_run in project.get_runs_by_type("identification"):
                    if id_run.id == input_run_id and id_run.output_path:
                        csv_path = Path(id_run.output_path) / "Dataset_identification_result.csv"
                        if csv_path.exists():
                            try:
                                df_id = pd.read_csv(csv_path)
                                acc_col = "GEO_Accession" if "GEO_Accession" in df_id.columns else "Accession"
                                if acc_col in df_id.columns and "DatasetSizeGB" in df_id.columns:
                                    for _, row in df_id.iterrows():
                                        size_lookup[str(row[acc_col])] = float(row["DatasetSizeGB"])
                            except Exception:
                                pass
                        break

            # Per-dataset status table
            dataset_statuses = st.session_state.get(f"dataset_status_{run.id}", {})
            geo_datasets = run.datasets or [
                k for k in dataset_statuses.keys() if k not in (run.custom_dataset_ids or [])
            ]
            custom_ids = run.custom_dataset_ids or []

            state_map = {
                "completed": "✅ Completed",
                "submitted": "⏳ Queued",
                "queued": "⏳ Queued",
                "running": "🔄 Running",
                "failed": "❌ Failed",
                "unknown": "⏳ Queued",
                "cancelled": "🚫 Cancelled",
            }

            if geo_datasets or custom_ids:
                rows = []

                # GEO accession rows (per-dataset SLURM status looked up by accession)
                for acc in geo_datasets:
                    ds = dataset_statuses.get(acc, {})
                    state = ds.get("state", "queued")
                    size_from_status = ds.get("size_gb")
                    if size_from_status and str(size_from_status) not in ("?", "0", "0.0"):
                        size = size_from_status
                    else:
                        size = f"{size_lookup.get(acc, 0):.1f}" if acc in size_lookup else "—"
                    rows.append({
                        "Dataset": acc,
                        "Status": state_map.get(state, f"⏳ {state}"),
                        "Size (GB)": size,
                    })

                # Custom dataset rows: label, fall back to overall run status when
                # we have no per-dataset SLURM info (custom job names differ from
                # GEO _UORCA.sbatch convention).
                for cid in custom_ids:
                    cd = custom_lookup.get(cid)
                    label = f"📁 {cd.label}" if cd else f"📁 {cid}"
                    ds = dataset_statuses.get(cid, {})
                    state = ds.get("state") or run.status or "queued"
                    rows.append({
                        "Dataset": label,
                        "Status": state_map.get(state, f"⏳ {state}"),
                        "Size (GB)": "—",
                    })

                df_status = pd.DataFrame(rows)
                st.dataframe(df_status, use_container_width=True, hide_index=True)

                if not dataset_statuses and geo_datasets:
                    st.caption("Click **Refresh Status** to fetch live progress.")
            else:
                st.caption("No dataset details available.")

            # Show screen session status and cancel button for running runs
            if run.status == "running" and hpc_ok:
                screen_name = run.params.get("screen_name", "")
                if screen_name:
                    st.caption(f"Screen session: `{screen_name}`")
                cancel_key = f"cancel_{run.id}"
                if st.button("Cancel Submission", key=cancel_key, type="secondary"):
                    password = st.session_state.get("run_hpc_password", "")
                    if password and screen_name:
                        try:
                            from uorca.gui.hpc.ssh_manager import close, connect, run_command
                            client = connect(hpc.host_alias, password=password or None)
                            # Kill screen session
                            run_command(client, f"screen -S {screen_name} -X quit 2>/dev/null || true", timeout=10)
                            # Cancel any SLURM jobs for this run's datasets
                            for acc in (run.datasets or []):
                                run_command(client, f"scancel --name={acc}_UORCA.sbatch 2>/dev/null || true", timeout=5)
                            close(client)
                            run.status = "cancelled"
                            _save_project(project)
                            st.warning("Submission cancelled.")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Cancel failed: {exc}")
                    else:
                        st.warning("Enter HPC password to cancel.")

            st.caption(f"Output: `{run.output_path or '—'}`")

    # ------------------------------------------------------------------
    # Sync results locally via rsync
    # ------------------------------------------------------------------
    if hpc_ok and sync_clicked:
        # Find completed runs whose output_path is still remote (absolute).
        # After a successful sync we rewrite output_path to a local relative
        # path, so this filter naturally excludes already-synced runs.
        remote_runs = sorted(
            [r for r in pipeline_runs
             if r.status == "completed"
             and r.output_path
             and Path(r.output_path).is_absolute()],
            key=lambda r: r.timestamp or "",
            reverse=True,
        )
        if not remote_runs:
            st.warning("No completed pipeline runs with remote paths to sync.")
        else:
            run = remote_runs[0]  # most recent
            remote_path = run.output_path
            run_name = Path(remote_path).name
            local_path = Path("results") / run_name
            local_path.mkdir(parents=True, exist_ok=True)

            with st.spinner(f"Syncing {run_name} from {hpc.host_alias}..."):
                try:
                    import subprocess
                    result = subprocess.run(
                        ["rsync", "-avz", "--progress",
                         f"{hpc.host_alias}:{remote_path}/",
                         f"{local_path}/"],
                        capture_output=True, text=True, timeout=600,
                    )
                    if result.returncode == 0:
                        # Update project YAML to point to local path
                        run.output_path = str(local_path)
                        _save_project(project)
                        st.success(f"Synced **{run_name}** to `{local_path}`")
                    else:
                        st.error(f"rsync failed: {result.stderr[-300:]}")
                except FileNotFoundError:
                    st.error("rsync not found. Please install rsync.")
                except subprocess.TimeoutExpired:
                    st.error("rsync timed out after 10 minutes.")
                except Exception as exc:
                    st.error(f"Sync error: {exc}")

    # ------------------------------------------------------------------
    # Previous pipeline runs
    # ------------------------------------------------------------------
    if pipeline_runs:
        with st.expander(f"Previous Pipeline Runs ({len(pipeline_runs)})", expanded=False):
            for run in sorted(pipeline_runs, key=lambda r: r.timestamp or "", reverse=True):
                ts = (run.timestamp or "—")[:19]
                n_geo_p = len(run.datasets or [])
                n_custom_p = len(run.custom_dataset_ids or [])
                n_ds = (
                    n_geo_p + n_custom_p
                    if (n_geo_p or n_custom_p)
                    else run.params.get("n_datasets", "?")
                )
                breakdown = (
                    f" ({n_geo_p} GEO + {n_custom_p} custom)"
                    if (n_geo_p and n_custom_p)
                    else ""
                )
                st.markdown(
                    f"- **{run.id}** &nbsp; `{run.status}` &nbsp; {ts} &nbsp; "
                    f"— {n_ds} dataset(s){breakdown}  \n"
                    f"  &nbsp;&nbsp; Output: `{run.output_path or '—'}`"
                )
    else:
        st.caption("No pipeline runs recorded yet.")


# ---------------------------------------------------------------------------
# Page entry point
# ---------------------------------------------------------------------------


def page() -> None:
    st.title("Run Pipeline")

    project = _get_project()
    if not project:
        st.warning("No active project. Please select or create a project first.")
        if st.button("Go to Project Setup"):
            st.switch_page(st.session_state["_pages"]["project-setup"])
        return

    col_submit, col_monitor = st.columns([1, 1])

    with col_submit:
        _render_left_panel(project)

    with col_monitor:
        _render_right_panel(project)
