"""Environment for the pipeline containers: AI provider settings and keys.

The container must resolve the same AI model as the host, and it needs the selected
provider's key. Keys are passed **by variable name only**, never by value:

* Docker: ``-e NAME`` without ``=value``; Docker copies the value from the environment
  of the ``docker`` process. A value is never on the command line, so ``ps`` cannot show
  it, and it is never written to a status file.
* Apptainer under SLURM: the sbatch script contains ``export APPTAINERENV_NAME="${NAME}"``.
  The script file holds the variable name; the shell expands it at run time from the
  job's environment, which sbatch copies from the submitting shell. ``APPTAINERENV_``
  variables pass ``--cleanenv``, and an ``export`` is a shell builtin, so the value is on
  no command line either.

Non-secret settings (provider, model, region, endpoint) are passed with their values.
"""

from __future__ import annotations

import shlex
from pathlib import Path

from uorca.ai_provider import ContainerEnv

# NCBI settings every pipeline job needs. ENTREZ_API_KEY is a key, so both go by name.
ENTREZ_VARS = ("ENTREZ_EMAIL", "ENTREZ_API_KEY")

# Docker runs the image as root, so its home is /root.
DOCKER_AWS_DIR = "/root/.aws"


def docker_env_args(cenv: ContainerEnv, home: Path | None = None) -> list[str]:
    """``docker run`` arguments for the NCBI and AI provider environment.

    Args:
        cenv: From :func:`uorca.ai_provider.container_env`.
        home: Home directory holding ``.aws`` (defaults to ``Path.home()``).
    """
    args: list[str] = []
    for name in ENTREZ_VARS:
        args += ["-e", name]
    for key, value in cenv.settings.items():
        args += ["-e", f"{key}={value}"]
    for name in cenv.secret_names:
        args += ["-e", name]
    if cenv.needs_aws_config:
        aws_dir = (home or Path.home()) / ".aws"
        if aws_dir.is_dir():
            args += ["-v", f"{aws_dir}:{DOCKER_AWS_DIR}:ro"]
    return args


def docker_env_shell(cenv: ContainerEnv) -> str:
    """Bash lines that build a ``DOCKER_AI_ARGS`` array, for a script run by sbatch.

    Use it as ``docker run ... "${DOCKER_AI_ARGS[@]}" ...``.
    """
    parts = [f"-e {name}" for name in ENTREZ_VARS]
    parts += [f"-e {shlex.quote(f'{key}={value}')}" for key, value in cenv.settings.items()]
    parts += [f"-e {name}" for name in cenv.secret_names]
    lines = [
        "# AI provider settings by value; keys by name only (Docker reads them from this",
        "# job's environment, so no key value is written here or shown by ps).",
        "DOCKER_AI_ARGS=(" + " ".join(parts) + ")",
    ]
    if cenv.needs_aws_config:
        lines += [
            "# Bedrock: AWS profiles and SSO tokens, read-only.",
            'if [ -d "$HOME/.aws" ]; then DOCKER_AI_ARGS+=(-v "$HOME/.aws:'
            + DOCKER_AWS_DIR + ':ro"); fi',
        ]
    return "\n".join(lines)


def apptainer_env_shell(cenv: ContainerEnv) -> str:
    """Bash lines that export ``APPTAINERENV_*`` for the NCBI and AI provider settings.

    Keys are copied from the job's environment by name, and only when set.
    """
    lines = [
        "# AI provider settings by value; keys by name only. APPTAINERENV_* variables",
        "# pass --cleanenv; the key values come from this job's environment at run time.",
    ]
    for key, value in cenv.settings.items():
        lines.append(f"export APPTAINERENV_{key}={shlex.quote(value)}")
    for name in (*ENTREZ_VARS, *cenv.secret_names):
        lines.append(
            f'if [ -n "${{{name}+x}}" ]; then export APPTAINERENV_{name}="${{{name}}}"; fi'
        )
    return "\n".join(lines)


def apptainer_aws_bind_shell(cenv: ContainerEnv) -> str:
    """Bash line that adds a read-only ``~/.aws`` bind to ``$BIND`` for Bedrock."""
    if not cenv.needs_aws_config:
        return ""
    return (
        "# Bedrock: AWS profiles and SSO tokens, read-only.\n"
        'if [ -d "$HOME/.aws" ]; then BIND="$BIND -B $HOME/.aws:$HOME/.aws:ro"; fi'
    )
