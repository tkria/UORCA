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

The repository is mounted into every container, and the pipeline calls
``load_dotenv()``, so the repository's ``.env`` would hand **every** key in it to every
job. Each backend therefore mounts an empty read-only file over it.

Every variable name is checked against :data:`ENV_NAME_RE` before it is written into a
command or script, so a name cannot inject shell code.
"""

from __future__ import annotations

import shlex
from pathlib import Path

from uorca.ai_provider import ENV_NAME_RE, ContainerEnv

# NCBI settings every pipeline job needs. ENTREZ_API_KEY is a key, so both go by name.
ENTREZ_VARS = ("ENTREZ_EMAIL", "ENTREZ_API_KEY")

# Docker runs the image as root, so its home is /root.
DOCKER_AWS_DIR = "/root/.aws"

# Name of the empty file mounted over the repository's .env inside a container.
EMPTY_ENV_NAME = "uorca-empty.env"


def _checked(name: str) -> str:
    if not ENV_NAME_RE.match(name):
        raise ValueError(
            f"{name!r} is not a valid environment variable name, so it cannot be passed "
            "into a pipeline container."
        )
    return name


def _names(cenv: ContainerEnv) -> tuple[list[str], list[str]]:
    """``(setting names, by-name variables)``, all validated."""
    settings = [_checked(k) for k in cenv.settings]
    by_name = [_checked(n) for n in (*ENTREZ_VARS, *cenv.secret_names)]
    return settings, by_name


# ---------------------------------------------------------------------------
# Docker (uorca run local)
# ---------------------------------------------------------------------------

def docker_env_args(cenv: ContainerEnv, home: Path | None = None) -> list[str]:
    """``docker run`` arguments for the NCBI and AI provider environment.

    Args:
        cenv: From :func:`uorca.ai_provider.container_env`.
        home: Home directory holding ``.aws`` (defaults to ``Path.home()``).
    """
    _, by_name = _names(cenv)
    args: list[str] = []
    for key, value in cenv.settings.items():
        args += ["-e", f"{key}={value}"]
    for name in by_name:
        args += ["-e", name]
    if cenv.needs_aws_config:
        aws_dir = (home or Path.home()) / ".aws"
        if aws_dir.is_dir():
            args += ["-v", f"{aws_dir}:{DOCKER_AWS_DIR}:ro"]
    return args


def docker_dotenv_mask_args(project_root: Path, mount_point: str, scratch_dir: Path) -> list[str]:
    """Mount an empty read-only file over ``<mount_point>/.env`` when the repo has one.

    A bind-mounted file works on Docker Desktop (macOS, Windows) and on Linux. Nothing
    is mounted when the repository has no ``.env``, so Docker never creates one.

    Args:
        project_root: The repository on the host, mounted at *mount_point*.
        mount_point: Where the repository appears inside the container.
        scratch_dir: A host directory for the empty file.
    """
    if not (project_root / ".env").exists():
        return []
    empty = scratch_dir / EMPTY_ENV_NAME
    empty.write_text("")
    empty.chmod(0o444)
    return ["-v", f"{empty}:{mount_point}/.env:ro"]


def docker_env_shell(cenv: ContainerEnv, project_root: Path, mount_point: str) -> str:
    """Bash lines that build a ``DOCKER_AI_ARGS`` array, for a script run by sbatch.

    Use it as ``docker run ... "${DOCKER_AI_ARGS[@]}" ...``. Needs ``$TEMP_DIR``.
    """
    _, by_name = _names(cenv)
    parts = [f"-e {shlex.quote(f'{key}={value}')}" for key, value in cenv.settings.items()]
    parts += [f"-e {name}" for name in by_name]
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
    lines += _dotenv_mask_shell(
        project_root,
        'DOCKER_AI_ARGS+=(-v "${TEMP_DIR}/' + EMPTY_ENV_NAME + ':' + mount_point + '/.env:ro")',
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Apptainer (uorca run slurm)
# ---------------------------------------------------------------------------

def apptainer_env_shell(cenv: ContainerEnv) -> str:
    """Bash lines that export ``APPTAINERENV_*`` for the NCBI and AI provider settings.

    Keys are copied from the job's environment by name, and only when set.
    """
    settings, by_name = _names(cenv)
    lines = [
        "# AI provider settings by value; keys by name only. APPTAINERENV_* variables",
        "# pass --cleanenv; the key values come from this job's environment at run time.",
    ]
    for key in settings:
        lines.append(f"export APPTAINERENV_{key}={shlex.quote(cenv.settings[key])}")
    for name in by_name:
        lines.append(
            f'if [ -n "${{{name}+x}}" ]; then export APPTAINERENV_{name}="${{{name}}}"; fi'
        )
    return "\n".join(lines)


def _add_apptainer_bind(spec: str) -> str:
    """Append *spec* (already quoted for bash) to APPTAINER_BIND (comma-separated)."""
    return f'export APPTAINER_BIND="${{APPTAINER_BIND:+$APPTAINER_BIND,}}"{spec}'


def apptainer_binds_shell(cenv: ContainerEnv, project_root: Path, mount_point: str) -> str:
    """Bash lines adding read-only binds through ``APPTAINER_BIND``.

    * Bedrock: ``~/.aws`` read-only (profiles and SSO tokens).
    * The repository's ``.env``, masked by an empty file. Needs ``$TEMP_DIR``.

    ``APPTAINER_BIND`` is used instead of the word-split ``$BIND`` string so that a
    path with spaces, such as ``$HOME``, stays one argument.
    """
    lines: list[str] = []
    if cenv.needs_aws_config:
        lines += [
            "# Bedrock: AWS profiles and SSO tokens, read-only.",
            'if [ -d "$HOME/.aws" ]; then '
            + _add_apptainer_bind('"$HOME/.aws:$HOME/.aws:ro"') + "; fi",
        ]
    lines += _dotenv_mask_shell(
        project_root,
        _add_apptainer_bind('"${TEMP_DIR}/' + EMPTY_ENV_NAME + ':' + mount_point + '/.env:ro"'),
    )
    return "\n".join(lines)


def _dotenv_mask_shell(project_root: Path, add_mount: str) -> list[str]:
    env_file = shlex.quote(str(project_root / ".env"))
    return [
        "# Hide the repository's .env (it holds every key) behind an empty read-only file.",
        f"if [ -f {env_file} ]; then",
        f'  : > "${{TEMP_DIR}}/{EMPTY_ENV_NAME}"',
        f"  {add_mount}",
        "fi",
    ]
