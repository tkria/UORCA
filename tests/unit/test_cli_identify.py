"""Regression tests for running identification from a fresh install.

Each test pins a defect found by installing the public branch in a clean checkout:

- ``uorca identify`` forwarded ``--model`` to a parser that does not define it, so every
  run died with "unrecognized arguments: --model gpt-5-mini".
- The CLI defined the scoring flags but did not forward them, so user values were ignored.
- ``-n/--num-assess`` was accepted but never used.
- ``uorca/identify.py`` changed into the package directory, so a relative ``-o`` landed
  inside the package.
- A missing ``ENTREZ_EMAIL`` returned exit code 0.
- A rate-limited (429) Entrez key check was reported as "API key invalid" and the key
  was dropped for the whole run.
- A small-corpus theme map logged an ERROR traceback on every small test run.
"""

from __future__ import annotations

import logging
import os
import sys
import urllib.error
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def _cli_args(argv: list[str]):
    from uorca.cli import _build_parser

    return _build_parser().parse_args(["identify", *argv])


# ---------------------------------------------------------------------------
# CLI -> identification module forwarding
# ---------------------------------------------------------------------------


def test_default_identify_argv_is_accepted_by_the_identification_parser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from uorca.cli import build_identify_argv
    from uorca.identification.dataset_identification import build_arg_parser

    monkeypatch.chdir(tmp_path)
    argv = build_identify_argv(_cli_args(["-q", "IL-6 hepatocytes"]))

    assert "--model" not in argv
    inner = build_arg_parser().parse_args(argv)
    assert inner.query == "IL-6 hepatocytes"
    assert inner.num_assess is None


def test_every_identify_option_reaches_the_identification_parser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from uorca.cli import build_identify_argv
    from uorca.identification.dataset_identification import build_arg_parser

    monkeypatch.chdir(tmp_path)
    args = _cli_args([
        "-q", "q", "-o", "out", "-t", "6.5", "-m", "7", "-r", "1", "-b", "5",
        "--biology-weight", "0.5", "--design-min", "2",
        "--stage1-biology-threshold", "4", "--library-source", "sc",
        "--context", "only human", "--model", "gpt-5-mini", "-v",
    ])
    inner = build_arg_parser().parse_args(build_identify_argv(args))

    assert inner.output == str((tmp_path / "out").resolve())
    assert inner.threshold == 6.5
    assert inner.max_per_term == 7
    assert inner.rounds == 1
    assert inner.batch_size == 5
    assert inner.biology_weight == 0.5
    assert inner.design_min == 2
    assert inner.stage1_biology_threshold == 4
    assert inner.library_source == "sc"
    assert inner.context == "only human"
    assert inner.verbose is True


def test_run_identify_applies_model_through_the_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import uorca.cli as cli
    import uorca.identify as identify_wrapper

    monkeypatch.chdir(tmp_path)
    # setenv first so monkeypatch restores the original state after run_identify
    # writes the variable; a bare delenv of an absent variable records nothing.
    monkeypatch.setenv("UORCA_OPENAI_MODEL", "placeholder")
    monkeypatch.delenv("UORCA_OPENAI_MODEL")
    monkeypatch.setattr(cli, "_load_environment_variables", lambda: None)
    monkeypatch.setattr(cli, "_check_environment_requirements", lambda **_: None)
    seen: dict = {}

    def fake_main() -> None:
        seen["argv"] = list(sys.argv)
        seen["model"] = os.environ.get("UORCA_OPENAI_MODEL")

    monkeypatch.setattr(identify_wrapper, "main", fake_main)
    monkeypatch.setattr(sys, "argv", ["uorca"])

    cli.run_identify(_cli_args(["-q", "q", "--model", "gpt-5-mini"]))

    assert seen["model"] == "gpt-5-mini"
    assert "--model" not in seen["argv"]


def test_num_assess_is_rejected_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    from uorca.identification import dataset_identification

    monkeypatch.setattr(sys, "argv", ["identify", "-q", "q", "-n", "5"])
    with pytest.raises(SystemExit) as excinfo:
        dataset_identification.main()
    assert excinfo.value.code == 2


def test_missing_entrez_email_exits_non_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from uorca.identification import dataset_identification

    monkeypatch.delenv("ENTREZ_EMAIL", raising=False)
    monkeypatch.setattr(
        sys, "argv", ["identify", "-q", "q", "-o", str(tmp_path / "out")]
    )
    with pytest.raises(SystemExit) as excinfo:
        dataset_identification.main()
    assert excinfo.value.code == 1


def test_identify_wrapper_does_not_change_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import uorca.identify as identify_wrapper
    from uorca.identification import dataset_identification

    monkeypatch.chdir(tmp_path)
    seen: dict = {}
    monkeypatch.setattr(
        dataset_identification, "main", lambda: seen.setdefault("cwd", os.getcwd())
    )

    identify_wrapper.main()

    assert Path(seen["cwd"]).resolve() == tmp_path.resolve()


# ---------------------------------------------------------------------------
# Entrez API key check
# ---------------------------------------------------------------------------


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://x", code, "msg", None, None)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("code", "expected"),
    [(400, False), (429, None), (503, None)],
)
def test_only_http_400_marks_the_entrez_key_invalid(
    monkeypatch: pytest.MonkeyPatch, code: int, expected
) -> None:
    import urllib.request

    from uorca.identification import dataset_identification

    def fake_urlopen(*args, **kwargs):
        raise _http_error(code)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(dataset_identification.time, "sleep", lambda s: None)

    valid, detail = dataset_identification._validate_entrez_api_key("k", "e@x.org")

    assert valid is expected
    assert str(code) in detail


def test_rate_limited_key_check_keeps_the_key(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from uorca.identification import dataset_identification

    monkeypatch.setenv("ENTREZ_API_KEY", "real-key")
    monkeypatch.setenv("ENTREZ_EMAIL", "e@x.org")
    monkeypatch.setattr(
        dataset_identification,
        "_validate_entrez_api_key",
        lambda key, email: (None, "HTTP 429"),
    )
    with caplog.at_level(logging.WARNING):
        dataset_identification._configure_entrez()

    assert os.environ.get("ENTREZ_API_KEY") == "real-key"
    assert dataset_identification.Entrez.api_key == "real-key"
    assert "HTTP 429" in caplog.text
    assert "invalid" not in caplog.text.lower()


def test_rejected_key_is_dropped_with_the_real_status(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from uorca.identification import dataset_identification

    monkeypatch.setenv("ENTREZ_API_KEY", "bad-key")
    monkeypatch.setenv("ENTREZ_EMAIL", "e@x.org")
    monkeypatch.setattr(
        dataset_identification,
        "_validate_entrez_api_key",
        lambda key, email: (False, "HTTP 400 (NCBI rejected the API key)"),
    )
    with caplog.at_level(logging.WARNING):
        dataset_identification._configure_entrez()

    assert "ENTREZ_API_KEY" not in os.environ
    assert dataset_identification.Entrez.api_key is None
    assert "HTTP 400" in caplog.text


# ---------------------------------------------------------------------------
# Theme map on a small run
# ---------------------------------------------------------------------------


def test_small_corpus_theme_map_is_a_warning_not_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from uorca.identification import theme_map
    from uorca.identification.dataset_identification import (
        build_theme_map_for_finished_run,
    )
    from uorca.identification.theme_map.cluster import CorpusTooSmallError

    def too_small(run_dir, **kwargs):
        raise CorpusTooSmallError("27 datasets, minimum 50")

    monkeypatch.setattr(theme_map, "build_theme_map", too_small)

    with caplog.at_level(logging.WARNING):
        built = build_theme_map_for_finished_run(tmp_path)

    assert built is False
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert "Traceback" not in caplog.text
    assert "corpus too small" in caplog.text
    recorded = theme_map.read_failure(tmp_path)
    assert recorded is not None and "CorpusTooSmallError" in recorded
