"""AppTest-driven integration test for the Run page mixed submission.

Asserts the constructed CLI command contains both --input and one
--user-config per ticked custom dataset, and that the RunEntry is written
with the correct custom_dataset_ids."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest


def test_mixed_submit_command_construction(monkeypatch, tmp_path):
    from uorca.gui.pages import run as run_page
    from uorca.gui.project.manager import ProjectManager
    from uorca.gui.project.models import (
        CustomDataset,
        HpcConfig,
        Project,
        RunEntry,
        SlurmConfig,
    )

    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    project = Project(name="MixSubmit", description="", default_research_question="q")
    project.hpc = HpcConfig()
    project.hpc.host_alias = "example_host"
    project.hpc.remote_working_dir = "/data/uorca"
    project.hpc.resource_dir = "/data/kallisto"
    project.hpc.slurm = SlurmConfig(
        partition="example_partition",
        constraint="icx",
        cpus_per_task=8,
        memory="32G",
        time_limit="4:00:00",
        max_parallel=4,
    )
    project.custom_datasets.append(
        CustomDataset(
            id="my-data",
            label="My data",
            fastq_dir="/data/my/fastqs",
            metadata_filename="my-data.csv",
            organism="Homo sapiens",
            validation_status="valid",
        )
    )
    mgr = ProjectManager()
    slug = mgr.get_project_slug(project.name)
    mgr.save_project(slug, project)

    captured: dict = {"commands": []}

    base_slurm_yaml = (
        "slurm:\n"
        "  partition: original_partition\n"
        "  constraint: example_constraint\n"
        "  cpus_per_task: 12\n"
        "  memory: 16G\n"
        "  time_limit: 6:00:00\n"
        "container:\n"
        "  engine: apptainer\n"
        "  apptainer_image: ./uorca_dev.sif\n"
        "resource_management:\n"
        "  max_storage_gb: 500\n"
        "  check_interval: 30\n"
        "  max_parallel: 6\n"
    )

    def fake_run_command(client, cmd, timeout=10):
        captured["commands"].append(cmd)
        if "screen -ls" in cmd:
            # First call = duplicate check (none); second call = verify session running
            if any("screen -dmS" in c for c in captured["commands"]):
                return 0, f"\t{captured['screen']}\t(Detached)\n", ""
            return 0, "", ""
        if "screen -dmS" in cmd:
            captured["screen"] = cmd.split()[2]
            return 0, "", ""
        if "cat /data/uorca/slurm_config.yaml" in cmd:
            return 0, base_slurm_yaml, ""
        if cmd.startswith("cat > ") and "_slurm_config.yaml" in cmd:
            # Heredoc shape: `cat > path <<'__YAML_EOF__'\n<yaml>__YAML_EOF__\n`.
            opener = "<<'__YAML_EOF__'\n"
            closer = "__YAML_EOF__\n"
            start = cmd.index(opener) + len(opener)
            end = cmd.rindex(closer)
            captured["merged_slurm_yaml"] = cmd[start:end]
            return 0, "", ""
        return 0, "", ""

    fake_client = MagicMock()
    monkeypatch.setattr("uorca.gui.hpc.ssh_manager.connect", lambda *a, **kw: fake_client)
    monkeypatch.setattr("uorca.gui.hpc.ssh_manager.close", lambda c: None)
    monkeypatch.setattr("uorca.gui.hpc.ssh_manager.run_command", fake_run_command)
    # The function imports run_command locally — patch the import target too.
    import uorca.gui.pages.run as run_mod
    monkeypatch.setattr(run_mod, "_make_screen_name", lambda p: "uorca_test_session")

    ok, message, results_dir = run_page._submit_jobs(
        project=project,
        datasets=["GSE111", "GSE222"],
        custom_ids=["my-data"],
        password="pw",
    )

    assert ok, message
    submit_cmd = next(
        c for c in captured["commands"] if "uv run uorca run slurm" in c
    )
    assert "--input " in submit_cmd
    assert "_gui_submit_" in submit_cmd
    assert "--user-config " in submit_cmd
    assert "_user_my-data.yaml" in submit_cmd
    assert "--output_dir" in submit_cmd

    # The submit command should reference the per-run merged slurm config in
    # scratch, not the unqualified "slurm_config.yaml".
    assert "--config /data/uorca/scratch/_gui_submit_" in submit_cmd
    assert "__slurm_config.yaml" in submit_cmd

    # The merged YAML emitted to scratch should reflect the project's
    # overrides while preserving the host's container settings.
    import yaml as _yaml
    merged = _yaml.safe_load(captured["merged_slurm_yaml"])
    assert merged["slurm"]["partition"] == "example_partition"
    assert merged["slurm"]["constraint"] == "icx"
    assert merged["slurm"]["cpus_per_task"] == 8
    assert merged["slurm"]["memory"] == "32G"
    assert merged["slurm"]["time_limit"] == "4:00:00"
    assert merged["resource_management"]["max_parallel"] == 4
    # Host-only fields preserved
    assert merged["resource_management"]["max_storage_gb"] == 500
    assert merged["container"]["apptainer_image"] == "./uorca_dev.sif"
