"""SLURM job management utilities for remote HPC clusters."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import paramiko


# ---------------------------------------------------------------------------
# Output parsers
# ---------------------------------------------------------------------------

def parse_squeue_output(raw: str) -> list[dict[str, str]]:
    """Parse pipe-delimited ``squeue`` output into a list of job dicts.

    Expected column order (no header):
        job_id | name | state | time | time_limit | partition | reason

    Empty or whitespace-only input returns an empty list.
    """
    jobs: list[dict[str, str]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) < 7:
            continue
        jobs.append(
            {
                "job_id": parts[0],
                "name": parts[1],
                "state": parts[2],
                "time": parts[3],
                "time_limit": parts[4],
                "partition": parts[5],
                "reason": parts[6],
            }
        )
    return jobs


def parse_sacct_output(raw: str) -> list[dict[str, str]]:
    """Parse pipe-delimited ``sacct`` output into a list of job dicts.

    Expected column order (no header):
        job_id | name | state | exit_code | start | end | elapsed | max_rss

    Lines whose ``job_id`` field contains a ``.`` (sub-step entries such as
    ``12345.batch``) are skipped.

    Empty or whitespace-only input returns an empty list.
    """
    jobs: list[dict[str, str]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) < 8:
            continue
        job_id = parts[0]
        if "." in job_id:
            continue
        jobs.append(
            {
                "job_id": job_id,
                "name": parts[1],
                "state": parts[2],
                "exit_code": parts[3],
                "start": parts[4],
                "end": parts[5],
                "elapsed": parts[6],
                "max_rss": parts[7],
            }
        )
    return jobs


# ---------------------------------------------------------------------------
# Remote SLURM operations
# ---------------------------------------------------------------------------

_SQUEUE_FORMAT = "%i|%j|%T|%M|%l|%P|%r"
_SACCT_FORMAT = "jobid,jobname,state,exitcode,start,end,elapsed,maxrss"


def squeue_poll(
    client: "paramiko.SSHClient",
    username: str,
) -> list[dict[str, str]]:
    """Query the SLURM queue for *username* and return parsed job dicts."""
    from .ssh_manager import run_command  # noqa: PLC0415 — lazy to avoid early paramiko import

    cmd = f'squeue -u "{username}" -h --format="{_SQUEUE_FORMAT}"'
    _exit_code, stdout, _stderr = run_command(client, cmd)
    return parse_squeue_output(stdout)


def sacct_query(
    client: "paramiko.SSHClient",
    job_ids: list[str],
) -> list[dict[str, str]]:
    """Query ``sacct`` for the given *job_ids* and return parsed job dicts."""
    from .ssh_manager import run_command  # noqa: PLC0415

    ids_str = ",".join(job_ids)
    cmd = (
        f"sacct -j {ids_str} --noheader --parsable2 "
        f"--format={_SACCT_FORMAT}"
    )
    _exit_code, stdout, _stderr = run_command(client, cmd)
    return parse_sacct_output(stdout)


def sbatch_submit(
    client: "paramiko.SSHClient",
    script_path: str,
    cwd: str,
) -> str:
    """Submit *script_path* via ``sbatch --parsable`` from *cwd*.

    Returns the submitted job ID as a string.
    """
    from .ssh_manager import run_command  # noqa: PLC0415

    cmd = f'cd "{cwd}" && sbatch --parsable "{script_path}"'
    exit_code, stdout, stderr = run_command(client, cmd)
    if exit_code != 0:
        raise RuntimeError(
            f"sbatch failed (exit {exit_code}): {stderr.strip()}"
        )
    # --parsable output is "<jobid>" or "<jobid>;<cluster>" — take first token
    job_id = stdout.strip().split(";")[0].strip()
    return job_id
