"""Tests for the slurm subparser accepting --input and repeated --user-config."""

import pytest


def _build_parser():
    """Build the same argparse tree the CLI uses, isolated for testing."""
    from uorca.cli import _build_parser  # exposed by Task 3 refactor
    return _build_parser()


def test_user_config_repeats():
    parser = _build_parser()
    args = parser.parse_args([
        "run", "slurm",
        "--user-config", "a.yaml",
        "--user-config", "b.yaml",
        "--output_dir", "/tmp/out",
    ])
    assert args.user_config == ["a.yaml", "b.yaml"]
    assert args.input is None


def test_input_and_user_config_both_allowed():
    parser = _build_parser()
    args = parser.parse_args([
        "run", "slurm",
        "--input", "geo.csv",
        "--user-config", "u.yaml",
        "--output_dir", "/tmp/out",
    ])
    assert args.input == "geo.csv"
    assert args.user_config == ["u.yaml"]


def test_run_batch_slurm_calls_both_submission_paths(monkeypatch, tmp_path):
    """run_batch_slurm should call submit_array_job once and submit_user_data_job
    once per --user-config."""
    import argparse

    from uorca import cli

    geo_csv = tmp_path / "geo.csv"
    geo_csv.write_text("Accession\nGSE111\n")
    user_yaml_a = tmp_path / "a.yaml"
    user_yaml_b = tmp_path / "b.yaml"
    user_yaml_a.write_text("mode: user\n")
    user_yaml_b.write_text("mode: user\n")

    array_calls: list[str] = []
    user_calls: list[str] = []

    class FakeProcessor:
        def __init__(self, *a, **kw): pass
        def submit_datasets(self, input_path, output_dir, **kw):
            array_calls.append(input_path); return 1
        def submit_user_data_job(self, yaml_path): user_calls.append(yaml_path); return "5678"

    # Stub load_config to bypass WorkflowConfig validation for these stubs.
    class _FakeUserCfg:
        resource_dir = tmp_path / "res"

    import uorca.graph.config as graph_config
    monkeypatch.setattr(graph_config, "load_config", lambda *a, **kw: _FakeUserCfg())

    monkeypatch.setattr(cli, "SlurmBatchProcessor", FakeProcessor)
    monkeypatch.setattr(cli, "_load_environment_variables", lambda: None)
    monkeypatch.setattr(cli, "_check_environment_requirements", lambda **kw: None)
    monkeypatch.setattr(cli, "_check_kallisto_indices", lambda *a, **kw: None)

    args = argparse.Namespace(
        input=str(geo_csv),
        user_config=[str(user_yaml_a), str(user_yaml_b)],
        output_dir=str(tmp_path / "out"),
        resource_dir=str(tmp_path / "res"),
        config=None,
        max_parallel=2,
        max_storage_gb=10,
        no_cleanup=False,
    )

    cli.run_batch_slurm(args)

    assert array_calls == [str(geo_csv)]
    assert user_calls == [str(user_yaml_a), str(user_yaml_b)]


def test_run_batch_slurm_errors_when_neither_provided(monkeypatch, capsys):
    import argparse

    from uorca import cli

    monkeypatch.setattr(cli, "_load_environment_variables", lambda: None)
    monkeypatch.setattr(cli, "_check_environment_requirements", lambda **kw: None)

    args = argparse.Namespace(
        input=None,
        user_config=[],
        output_dir="/tmp",
        resource_dir="/tmp",
        config=None,
        max_parallel=2,
        max_storage_gb=10,
        no_cleanup=False,
    )

    with pytest.raises(SystemExit):
        cli.run_batch_slurm(args)
    captured = capsys.readouterr()
    err = captured.err + captured.out
    assert "--input" in err or "--user-config" in err
