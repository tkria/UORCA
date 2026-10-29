"""Pipeline containers resolve the host's AI model and get its key by name only.

For each provider, a host configuration (``~/.uorca/config.yaml`` and/or environment)
is turned into the real Docker argv (``run_single_dataset_local``) and the real sbatch
scripts (single dataset with Apptainer and with Docker, and user data). Each test then:

* rebuilds the environment the container would see: the settings passed by value, plus
  each variable passed by name with the value the runtime would copy from the host;
* resolves the model spec in that environment and requires it to equal the host's spec.
  Docker does not mount the home directory, so it resolves with **no** config file.
  Apptainer mounts ``$HOME`` by default, so ``~/.uorca/config.yaml`` is visible there,
  and it resolves with the host's config file present (the job runs under ``bash -lc``;
  the user's login files are not modelled);
* scans the argv, the scripts and every file the job wrote for each fake key value, and
  fails if one appears.

No Docker, SLURM or Apptainer is run; ``subprocess.run`` is replaced.
"""

from __future__ import annotations

import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import yaml

import uorca.ai_provider as ai_provider

pytestmark = pytest.mark.unit

# Every variable that can influence resolution or carry a key. All start unset.
_ALL_VARS = (
    "UORCA_AI_PROVIDER", "UORCA_OPENAI_MODEL", "UORCA_BEDROCK_MODEL", "UORCA_BEDROCK_REGION",
    "UORCA_BEDROCK_PROFILE", "UORCA_BEDROCK_MODEL_PREFIX", "UORCA_AZURE_DEPLOYMENT",
    "UORCA_OPENAI_COMPATIBLE_BASE_URL", "UORCA_OPENAI_COMPATIBLE_MODEL",
    "UORCA_OPENAI_COMPATIBLE_API_KEY_ENV", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY", "MISTRAL_API_KEY", "CO_API_KEY",
    "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "OPENAI_API_VERSION",
    "GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION", "GOOGLE_APPLICATION_CREDENTIALS",
    "AWS_PROFILE", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
    "AWS_REGION", "AWS_DEFAULT_REGION", "AWS_CONFIG_FILE", "AWS_SHARED_CREDENTIALS_FILE",
    "LOCAL_LLM_KEY", "ENTREZ_EMAIL", "ENTREZ_API_KEY",
)

ENTREZ = {"ENTREZ_EMAIL": "contract@example.org", "ENTREZ_API_KEY": "SECRET-ENTREZ-0001"}


@dataclass
class Case:
    name: str
    config: dict | None = None             # ai_provider section of ~/.uorca/config.yaml
    env: dict[str, str] = field(default_factory=dict)      # non-secret host settings
    secrets: dict[str, str] = field(default_factory=dict)  # key variables (fake values)


CASES = [
    Case("openai", config={"provider": "openai", "openai": {"model": "gpt-5"}},
         secrets={"OPENAI_API_KEY": "SECRET-OPENAI-0001"}),
    Case("anthropic", env={"UORCA_AI_PROVIDER": "anthropic:claude-sonnet-4-5"},
         secrets={"ANTHROPIC_API_KEY": "SECRET-ANTHROPIC-0001"}),
    Case("google-gla", config={"provider": "google-gla", "google-gla": {"model": "gemini-2.5-flash"}},
         secrets={"GOOGLE_API_KEY": "SECRET-GOOGLE-0001", "GEMINI_API_KEY": "SECRET-GEMINI-0001"}),
    Case("groq", env={"UORCA_AI_PROVIDER": "groq:llama-3.3-70b-versatile"},
         secrets={"GROQ_API_KEY": "SECRET-GROQ-0001"}),
    Case("mistral", env={"UORCA_AI_PROVIDER": "mistral:mistral-small-latest"},
         secrets={"MISTRAL_API_KEY": "SECRET-MISTRAL-0001"}),
    Case("cohere", env={"UORCA_AI_PROVIDER": "cohere:command-r-plus"},
         secrets={"CO_API_KEY": "SECRET-COHERE-0001"}),
    Case("bedrock",
         config={"provider": "bedrock", "bedrock": {
             "model": "anthropic.claude-sonnet-4-5-20250929-v1:0", "model_prefix": "au.",
             "region": "ap-southeast-2", "profile": "uorca-sso"}},
         secrets={"AWS_ACCESS_KEY_ID": "SECRET-AWSID-0001",
                  "AWS_SECRET_ACCESS_KEY": "SECRET-AWSKEY-0001",
                  "AWS_SESSION_TOKEN": "SECRET-AWSTOKEN-0001"}),
    Case("azure",
         config={"provider": "azure", "azure": {
             "endpoint": "https://contract.openai.azure.com", "deployment": "uorca-gpt",
             "api_version": "2024-10-21"}},
         secrets={"AZURE_OPENAI_API_KEY": "SECRET-AZURE-0001"}),
    Case("openai_compatible",
         config={"provider": "openai_compatible", "openai_compatible": {
             "base_url": "http://host.docker.internal:11434/v1", "model": "qwen3:8b",
             "api_key_env": "LOCAL_LLM_KEY"}},
         secrets={"LOCAL_LLM_KEY": "SECRET-LOCAL-0001",
                  # Set but not selected: must not be passed at all.
                  "OPENAI_API_KEY": "SECRET-OPENAI-UNUSED-0001"}),
]


@pytest.fixture
def host(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Return a function that installs a case as the host configuration."""
    home = tmp_path / "home"
    (home / ".aws").mkdir(parents=True)

    def install(case: Case) -> ai_provider.ModelSpec:
        for name in _ALL_VARS:
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("HOME", str(home))
        # Path.home() reads USERPROFILE on Windows, not HOME.
        monkeypatch.setenv("USERPROFILE", str(home))
        config_path = home / ".uorca" / "config.yaml"
        if case.config is not None:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(yaml.dump({"ai_provider": case.config}))
        monkeypatch.setattr(ai_provider, "CONFIG_PATH", config_path)
        for name, value in {**ENTREZ, **case.env, **case.secrets}.items():
            monkeypatch.setenv(name, value)
        ai_provider._load_ai_config.cache_clear()
        return ai_provider.resolve_model_spec()

    yield install, home
    ai_provider._load_ai_config.cache_clear()


def _resolve_in_container(monkeypatch, tmp_path, settings: dict[str, str],
                          names: list[str], host_env: dict[str, str],
                          config_path: Path | None = None) -> ai_provider.ModelSpec:
    """Resolve the spec with only what the container receives.

    *config_path* is the config file the container can see (Apptainer mounts $HOME);
    ``None`` means none (Docker).
    """
    for name in _ALL_VARS:
        monkeypatch.delenv(name, raising=False)
    for key, value in settings.items():
        monkeypatch.setenv(key, value)
    for name in names:  # the runtime copies by-name variables from the host
        if name in host_env:
            monkeypatch.setenv(name, host_env[name])
    monkeypatch.setattr(
        ai_provider, "CONFIG_PATH", config_path or tmp_path / "container-has-no-config.yaml"
    )
    ai_provider._load_ai_config.cache_clear()
    return ai_provider.resolve_model_spec()


def _assert_no_secret(case: Case, *texts: str) -> None:
    for value in [*case.secrets.values(), ENTREZ["ENTREZ_API_KEY"]]:
        for text in texts:
            assert value not in text, f"key value {value!r} leaked"


def _files_text(root: Path) -> str:
    return "\n".join(
        p.read_text(errors="replace") for p in root.rglob("*") if p.is_file()
    )


# ---------------------------------------------------------------------------
# Docker: uorca run local
# ---------------------------------------------------------------------------

def _docker_argv(monkeypatch, tmp_path) -> tuple[list[str], Path]:
    from uorca.batch import local

    captured: list[list[str]] = []

    class Done:
        returncode = 0
        stdout = "ok"
        stderr = ""

    def fake_run(cmd, **_kwargs):
        captured.append(list(cmd))
        return Done()

    monkeypatch.setattr(subprocess, "run", fake_run)
    out = tmp_path / "out"
    status = out / "job_status"
    status.mkdir(parents=True)
    (status / "GSE1_status.json").write_text("{}")
    result = local.run_single_dataset_local(
        "GSE1", str(out), str(tmp_path / "res"), cleanup=False, status_dir=str(status),
    )
    assert result["success"], result
    docker = [c for c in captured if c[:2] == ["docker", "run"]]
    assert len(docker) == 1
    return docker[0], out


def _parse_docker_env(argv: list[str]) -> tuple[dict[str, str], list[str], list[str]]:
    settings: dict[str, str] = {}
    names: list[str] = []
    mounts: list[str] = []
    for flag, value in zip(argv, argv[1:]):
        if flag == "-e":
            if "=" in value:
                key, val = value.split("=", 1)
                settings[key] = val
            else:
                names.append(value)
        elif flag == "-v":
            mounts.append(value)
    return settings, names, mounts


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_docker_container_resolves_host_model(case, host, monkeypatch, tmp_path):
    install, home = host
    host_spec = install(case)
    host_env = {**ENTREZ, **case.env, **case.secrets}

    argv, out = _docker_argv(monkeypatch, tmp_path)
    settings, names, mounts = _parse_docker_env(argv)

    _assert_no_secret(case, " ".join(argv), _files_text(out))
    for name in case.secrets:
        if name == "OPENAI_API_KEY" and host_spec.provider != "openai":
            assert name not in names, "an unselected provider's key must not be passed"
        else:
            assert name in names, f"{name} not passed by name"
    assert {"ENTREZ_EMAIL", "ENTREZ_API_KEY"} <= set(names)
    aws_mount = f"{home / '.aws'}:/root/.aws:ro"
    assert (aws_mount in mounts) == (host_spec.provider == "bedrock")

    container_spec = _resolve_in_container(monkeypatch, tmp_path, settings, names, host_env)
    assert container_spec == host_spec


# ---------------------------------------------------------------------------
# SLURM: uorca run slurm (single dataset, Apptainer and Docker; user data)
# ---------------------------------------------------------------------------

_EXPORT = re.compile(r"^export APPTAINERENV_(\w+)=(.*)$")
_BY_NAME = re.compile(r'^if \[ -n "\$\{(\w+)\+x\}" \]; then export APPTAINERENV_\1="\$\{\1\}"; fi$')


def _parse_apptainer_env(script: str) -> tuple[dict[str, str], list[str]]:
    settings: dict[str, str] = {}
    names: list[str] = []
    for line in script.splitlines():
        line = line.strip()
        if m := _BY_NAME.match(line):
            names.append(m.group(1))
        elif m := _EXPORT.match(line):
            settings[m.group(1)] = shlex.split(m.group(2))[0]
    # The by-name lines are the only place a key may be referenced.
    assert "--env OPENAI_API_KEY" not in script and "-e OPENAI_API_KEY=" not in script
    return settings, names


def _parse_docker_array(script: str) -> tuple[dict[str, str], list[str]]:
    line = next(l for l in script.splitlines() if l.startswith("DOCKER_AI_ARGS=("))
    return _parse_docker_env(shlex.split(line[len("DOCKER_AI_ARGS=("):-1]))[:2]


def _processor(monkeypatch):
    from uorca.batch.slurm import SlurmBatchProcessor

    return SlurmBatchProcessor.__new__(SlurmBatchProcessor)


def _user_config(tmp_path: Path) -> Path:
    fastq = tmp_path / "fq"
    fastq.mkdir()
    meta = tmp_path / "meta.csv"
    meta.write_text("sample_id\nS1\n")
    cfg = tmp_path / "user_config.yaml"
    cfg.write_text(yaml.safe_dump({
        "mode": "user",
        "user_data": {"fastq_dir": str(fastq), "metadata_path": str(meta)},
        "output_dir": str(tmp_path / "out"),
        "resource_dir": str(tmp_path / "res"),
    }))
    return cfg


@pytest.mark.parametrize("kind", ["apptainer", "docker", "user_data"])
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_sbatch_container_resolves_host_model(case, kind, host, monkeypatch, tmp_path):
    install, _ = host
    host_spec = install(case)
    host_env = {**ENTREZ, **case.env, **case.secrets}
    processor = _processor(monkeypatch)

    if kind == "user_data":
        script = processor._generate_user_data_slurm_script(
            str(_user_config(tmp_path)), tmp_path / "out", tmp_path / "logs",
        )
    else:
        script = processor._generate_slurm_script(
            "GSE1", str(tmp_path / "out"), tmp_path / "logs", container_engine=kind,
            apptainer_image=str(tmp_path / "uorca.sif"),
        )

    _assert_no_secret(case, script)
    if kind == "docker":
        settings, names = _parse_docker_array(script)
        assert '"${DOCKER_AI_ARGS[@]}"' in script
        aws_line = 'DOCKER_AI_ARGS+=(-v "$HOME/.aws:/root/.aws:ro")'
        env_mask = ':/workspace/src/.env:ro")'
        visible_config = None
    else:
        settings, names = _parse_apptainer_env(script)
        aws_line = '"$HOME/.aws:$HOME/.aws:ro"'
        env_mask = "/.env:ro\""
        assert "bash -lc" in script
        visible_config = ai_provider.CONFIG_PATH  # Apptainer mounts $HOME
    assert env_mask in script, "the repository .env must be masked"
    assert (aws_line in script) == (host_spec.provider == "bedrock")
    for name in case.secrets:
        if name == "OPENAI_API_KEY" and host_spec.provider != "openai":
            assert name not in names
        else:
            assert name in names, f"{name} not passed by name"

    container_spec = _resolve_in_container(
        monkeypatch, tmp_path, settings, names, host_env, visible_config
    )
    assert container_spec == host_spec


def test_sbatch_copies_the_submitting_environment(host, monkeypatch, tmp_path):
    """Keys reach the job through sbatch's environment copy, by name only."""
    install, _ = host
    install(CASES[1])
    from uorca.batch.slurm import SlurmBatchProcessor

    processor = SlurmBatchProcessor.__new__(SlurmBatchProcessor)
    calls: list[list[str]] = []

    class R:
        returncode = 0
        stdout = "Submitted batch job 1"
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: (calls.append(list(cmd)), R())[1])
    out = tmp_path / "out"
    out.mkdir()
    processor._submit_single_dataset(
        "GSE1", str(out), tmp_path / "logs", tmp_path, 1.0,
        container_engine="apptainer", apptainer_image=str(tmp_path / "uorca.sif"),
    )
    sbatch = [c for c in calls if c and c[0] == "sbatch"]
    assert sbatch and "--export=ALL" in sbatch[0]
    _assert_no_secret(CASES[1], " ".join(sbatch[0]))


def test_base_url_with_credentials_is_refused(host, monkeypatch):
    install, _ = host
    install(Case("openai_compatible", config={"provider": "openai_compatible", "openai_compatible": {
        "base_url": "https://bob:s3cret@llm.example.org/v1", "model": "m"}}))
    with pytest.raises(ai_provider.ProviderConfigError) as excinfo:
        ai_provider.container_env()
    assert "s3cret" not in str(excinfo.value)


# ---------------------------------------------------------------------------
# Security review: secrets in URLs, injected names, the repository .env,
# and setups that pass on the host but cannot work in a container.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "base_url",
    [
        "https://llm.example.org/v1?api-key=SECRET-QUERY-0001",
        "https://llm.example.org/v1?key=SECRET-QUERY-0001&x=1",
        "https://llm.example.org/v1#SECRET-QUERY-0001",
        "https://bob:SECRET-QUERY-0001@llm.example.org/v1",
    ],
)
def test_openai_compatible_url_with_a_secret_is_refused(host, base_url):
    install, _ = host
    install(Case("openai_compatible", config={"provider": "openai_compatible",
                 "openai_compatible": {"base_url": base_url, "model": "m"}}))
    with pytest.raises(ai_provider.ProviderConfigError, match="query string") as excinfo:
        ai_provider.container_env()
    assert "SECRET-QUERY-0001" not in str(excinfo.value)
    assert "https://llm.example.org/v1" in str(excinfo.value)


def test_azure_endpoint_with_a_query_is_refused(host):
    install, _ = host
    install(Case("azure", config={"provider": "azure", "azure": {
        "endpoint": "https://r.openai.azure.com/?api-key=SECRET-QUERY-0001",
        "deployment": "d", "api_version": "2024-10-21"}},
        secrets={"AZURE_OPENAI_API_KEY": "fake"}))
    with pytest.raises(ai_provider.ProviderConfigError) as excinfo:
        ai_provider.container_env()
    assert "SECRET-QUERY-0001" not in str(excinfo.value)


INJECTION = 'X}"; touch /tmp/pwn2; #'


def test_injected_key_variable_name_is_refused(host):
    install, _ = host
    install(Case("openai_compatible", config={"provider": "openai_compatible",
                 "openai_compatible": {"base_url": "http://llm.internal:8000/v1",
                                       "model": "m", "api_key_env": INJECTION}}))
    with pytest.raises(ai_provider.ProviderConfigError, match="must name an environment variable"):
        ai_provider.container_env()


@pytest.mark.parametrize("where", ["secret", "setting"])
def test_every_builder_refuses_an_invalid_name(tmp_path, where):
    from uorca.batch import container_env as ce

    if where == "secret":
        cenv = ai_provider.ContainerEnv(settings={}, secret_names=(INJECTION,))
    else:
        cenv = ai_provider.ContainerEnv(settings={INJECTION: "v"}, secret_names=())
    builders = [
        lambda: ce.docker_env_args(cenv, home=tmp_path),
        lambda: ce.docker_env_shell(cenv, tmp_path, "/workspace/src"),
        lambda: ce.apptainer_env_shell(cenv),
    ]
    for build in builders:
        with pytest.raises(ValueError, match="not a valid environment variable name"):
            build()


@pytest.mark.skipif(sys.platform == "win32", reason="Apptainer binds are a Linux shell snippet")
def test_aws_bind_quotes_home_with_spaces(tmp_path):
    """$HOME stays one argument: APPTAINER_BIND, not the word-split $BIND string."""
    from uorca.batch import container_env as ce

    cenv = ai_provider.ContainerEnv(settings={}, secret_names=(), needs_aws_config=True)
    project = tmp_path / "repo"
    project.mkdir()
    (project / ".env").write_text("OPENAI_API_KEY=SECRET-DOTENV-0001\n")
    shell = ce.apptainer_binds_shell(cenv, project, "/workspace")
    home = tmp_path / "my home"
    (home / ".aws").mkdir(parents=True)
    temp = tmp_path / "tmp dir"
    temp.mkdir()
    out = subprocess.run(
        ["bash", "-c", shell + '\nprintf "%s" "$APPTAINER_BIND"'],
        env={"HOME": str(home), "TEMP_DIR": str(temp), "PATH": "/usr/bin:/bin"},
        capture_output=True, text=True, check=True,
    ).stdout
    assert out == (
        f"{home}/.aws:{home}/.aws:ro,{temp}/{ce.EMPTY_ENV_NAME}:/workspace/.env:ro"
    )
    assert (temp / ce.EMPTY_ENV_NAME).read_text() == ""


def test_docker_masks_the_repository_dotenv(tmp_path):
    from uorca.batch import container_env as ce

    project = tmp_path / "repo"
    project.mkdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    assert ce.docker_dotenv_mask_args(project, "/workspace/src", scratch) == []  # no .env

    (project / ".env").write_text("OPENAI_API_KEY=SECRET-DOTENV-0001\n")
    args = ce.docker_dotenv_mask_args(project, "/workspace/src", scratch)
    empty = scratch / ce.EMPTY_ENV_NAME
    assert args == ["-v", f"{empty}:/workspace/src/.env:ro"]
    assert empty.read_text() == ""


def test_local_run_masks_the_dotenv_when_the_repository_has_one(host, monkeypatch, tmp_path):
    """The real Docker argv: masked iff the repository mounted at /workspace/src has .env."""
    install, _ = host
    install(CASES[0])
    from uorca.batch import local

    project_root = Path(local.__file__).resolve().parent.parent.parent
    argv, _ = _docker_argv(monkeypatch, tmp_path)
    masked = any(a.endswith(":/workspace/src/.env:ro") for a in argv)
    assert masked == (project_root / ".env").exists()


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        (Case("google-vertex", config={"provider": "google-vertex", "google-vertex": {
            "model": "gemini-2.5-pro", "project": "p"}}), "Vertex AI is not supported"),
        (Case("openai_compatible", config={"provider": "openai_compatible",
              "openai_compatible": {"base_url": "http://localhost:11434/v1", "model": "m"}}),
         "host.docker.internal"),
        (Case("openai_compatible", config={"provider": "openai_compatible",
              "openai_compatible": {"base_url": "http://127.0.0.1:8000/v1", "model": "m"}}),
         "host.docker.internal"),
    ],
    ids=["vertex", "localhost", "loopback"],
)
def test_setups_that_cannot_work_in_a_container_are_refused(host, case, expected):
    install, _ = host
    install(case)
    with pytest.raises(ai_provider.ProviderConfigError, match=expected):
        ai_provider.container_env()


def test_bedrock_with_aws_config_file_is_refused(host, monkeypatch, tmp_path):
    install, _ = host
    install(CASES[6])
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "elsewhere"))
    with pytest.raises(ai_provider.ProviderConfigError, match="AWS_CONFIG_FILE is set"):
        ai_provider.container_env()


def test_bedrock_with_credential_process_is_refused(host):
    install, home = host
    install(CASES[6])  # profile uorca-sso
    (home / ".aws" / "config").write_text(
        "[profile uorca-sso]\ncredential_process = /usr/local/bin/get-creds\n"
    )
    with pytest.raises(ai_provider.ProviderConfigError, match="credential_process"):
        ai_provider.container_env()


def test_bedrock_aws_config_file_at_the_default_path_is_accepted(host, monkeypatch):
    install, home = host
    install(CASES[6])
    monkeypatch.setenv("AWS_CONFIG_FILE", str(home / ".aws" / "config"))
    assert ai_provider.container_env().needs_aws_config
