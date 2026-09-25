"""SSH-side validation and SFTP staging helpers for custom datasets.

Used by the Identify > My Datasets registration flow and by the Run page
re-validation action. Errors are surfaced to the caller — never swallowed —
in keeping with the project's loud-errors policy.
"""

from __future__ import annotations

import logging
from typing import IO, Tuple

import paramiko

from uorca.gui.hpc.ssh_manager import run_command

logger = logging.getLogger(__name__)
# Stream INFO-level events to stderr so external monitors (Streamlit log,
# tail -f) can observe what the validator actually saw on the wire.
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(asctime)s %(levelname)s validator] %(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)
    logger.propagate = False


# Composite remote check — exits 0 on success, prints OK/NOTDIR/EMPTY tokens.
_VALIDATE_SCRIPT = r"""
sh -c '
  test -d "$0" || { echo NOTDIR; exit 1; }
  count=$(find "$0" -maxdepth 1 -name "*.fastq.gz" -print 2>/dev/null | head -1 | wc -l)
  test "$count" -ge 1 || { echo EMPTY; exit 2; }
  echo OK
'
"""


def validate_fastq_dir(
    client: paramiko.SSHClient, fastq_dir: str
) -> Tuple[str, str | None]:
    """Run a composite remote check for an HPC FASTQ directory.

    Returns ``(status, message)`` where ``status`` is one of
    ``"valid"`` / ``"missing"`` / ``"invalid"`` and ``message`` is ``None``
    on success or a human-readable error message otherwise.
    """
    quoted = fastq_dir.replace('"', '\\"')
    # Strip leading/trailing newlines from the heredoc so the path argument
    # is on the same shell line as `sh -c '...'` (otherwise the path becomes
    # a separate command and `$0` defaults to "sh", silently failing as NOTDIR).
    cmd = f'{_VALIDATE_SCRIPT.strip()} "{quoted}"'
    logger.info("validate_fastq_dir: sending command for path=%r", fastq_dir)
    logger.debug("validate_fastq_dir: cmd=%r", cmd)
    exit_code, stdout, stderr = run_command(client, cmd, timeout=15)
    logger.info(
        "validate_fastq_dir: exit_code=%s stdout=%r stderr=%r",
        exit_code, stdout, stderr,
    )

    token = (stdout.strip().splitlines() or [""])[-1]
    if token == "OK" and exit_code == 0:
        return "valid", None
    if token == "NOTDIR":
        return "missing", f"Path does not exist or is not a directory: {fastq_dir}"
    if token == "EMPTY":
        return "invalid", f"Directory contains no *.fastq.gz files: {fastq_dir}"

    detail = stderr.strip() or stdout.strip() or f"exit code {exit_code}"
    return "invalid", f"Validation failed: {detail}"


def stage_metadata_csv(
    client: paramiko.SSHClient,
    csv_bytes: IO[bytes],
    remote_path: str,
) -> None:
    """Upload an in-memory CSV to *remote_path* via SFTP.

    Parent directories are created if missing. Raises on failure — callers
    must catch and surface the error to the user via ``st.error``.
    """
    parent = remote_path.rsplit("/", 1)[0]
    # Ensure parent exists.
    run_command(client, f'mkdir -p "{parent}"', timeout=10)

    sftp = client.open_sftp()
    try:
        csv_bytes.seek(0)
        sftp.putfo(csv_bytes, remote_path)
    finally:
        sftp.close()


def remote_file_exists(client: paramiko.SSHClient, remote_path: str) -> bool:
    """Return True if *remote_path* exists on the host."""
    quoted = remote_path.replace('"', '\\"')
    exit_code, stdout, _ = run_command(
        client, f'test -f "{quoted}" && echo OK', timeout=10
    )
    return exit_code == 0 and stdout.strip() == "OK"
