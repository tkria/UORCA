"""Tests for SSH-side custom dataset validation."""

from io import BytesIO
from unittest.mock import MagicMock


def _stub_run_command(stdout: str, stderr: str = "", exit_code: int = 0):
    """Return a stub for ssh_manager.run_command."""

    def _stub(client, cmd, timeout=10):
        return exit_code, stdout, stderr

    return _stub


def test_validate_fastq_dir_ok(monkeypatch):
    from uorca.gui.hpc import custom_dataset_validator as v

    monkeypatch.setattr(v, "run_command", _stub_run_command("OK\n"))
    client = MagicMock()
    status, message = v.validate_fastq_dir(client, "/data/fq")
    assert status == "valid"
    assert message is None


def test_validate_fastq_dir_notdir(monkeypatch):
    from uorca.gui.hpc import custom_dataset_validator as v

    monkeypatch.setattr(
        v, "run_command", _stub_run_command("NOTDIR\n", "", 1)
    )
    client = MagicMock()
    status, message = v.validate_fastq_dir(client, "/missing")
    assert status == "missing"
    assert "does not exist" in message.lower()


def test_validate_fastq_dir_empty(monkeypatch):
    from uorca.gui.hpc import custom_dataset_validator as v

    monkeypatch.setattr(
        v, "run_command", _stub_run_command("EMPTY\n", "", 2)
    )
    client = MagicMock()
    status, message = v.validate_fastq_dir(client, "/data/empty")
    assert status == "invalid"
    assert "fastq.gz" in message


def test_validate_fastq_dir_unrecognised(monkeypatch):
    from uorca.gui.hpc import custom_dataset_validator as v

    monkeypatch.setattr(
        v, "run_command", _stub_run_command("", "permission denied\n", 5)
    )
    client = MagicMock()
    status, message = v.validate_fastq_dir(client, "/data/x")
    assert status == "invalid"
    assert "permission denied" in message.lower()
