"""Browser-test fixtures: a Streamlit app per test with an isolated HOME, driven by Playwright.

Run with:  uv run --group e2e pytest -m e2e -p no:cacheprovider --no-cov -q
The tests skip when not selected with ``-m e2e``, when Playwright is not installed, or when
its Chromium build is missing.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Callable, Iterator, Optional

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
APP = REPO_ROOT / "uorca" / "gui" / "app.py"


def pytest_collection_modifyitems(config, items):
    markexpr = config.getoption("-m") or ""
    if "e2e" in markexpr and "not e2e" not in markexpr:
        return
    skip = pytest.mark.skip(reason="browser test: run with -m e2e (and uv run --group e2e)")
    for item in items:
        if "e2e" in item.keywords:
            item.add_marker(skip)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class App:
    def __init__(self, proc: subprocess.Popen, url: str, home: Path, log: Path):
        self.proc = proc
        self.url = url
        self.home = home
        self.log = log

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)


def _start_app(home: Path, extra_env: Optional[dict[str, str]] = None) -> App:
    port = _free_port()
    env = {k: v for k, v in os.environ.items() if not k.startswith("UORCA_")}
    env.update({
        "HOME": str(home),
        "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
        "PYTHONUNBUFFERED": "1",
    })
    env.update(extra_env or {})
    log = home / f"streamlit_{port}.log"
    log_fh = log.open("w")
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(APP),
         "--server.headless", "true", "--server.address", "127.0.0.1",
         "--server.port", str(port), "--browser.gatherUsageStats", "false",
         "--server.fileWatcherType", "none"],
        cwd=REPO_ROOT, env=env, stdout=log_fh, stderr=subprocess.STDOUT,
    )
    url = f"http://127.0.0.1:{port}"
    app = App(proc, url, home, log)
    deadline = time.time() + 90
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Streamlit exited early:\n{log.read_text()}")
        try:
            with urllib.request.urlopen(f"{url}/_stcore/health", timeout=2) as r:
                if r.status == 200:
                    return app
        except OSError:
            time.sleep(0.5)
    app.stop()
    raise RuntimeError(f"Streamlit did not become healthy:\n{log.read_text()}")


@pytest.fixture
def home(tmp_path) -> Path:
    h = tmp_path / "home"
    h.mkdir()
    return h


@pytest.fixture
def start_app(home) -> Iterator[Callable[..., App]]:
    apps: list[App] = []

    def _start(extra_env: Optional[dict[str, str]] = None) -> App:
        app = _start_app(home, extra_env)
        apps.append(app)
        return app

    yield _start
    for app in apps:
        app.stop()


@pytest.fixture(scope="session")
def browser():
    sync_api = pytest.importorskip("playwright.sync_api", reason="playwright not installed (uv run --group e2e)")
    with sync_api.sync_playwright() as p:
        try:
            b = p.chromium.launch(headless=True)
        except Exception as exc:  # missing browser build
            pytest.skip(f"Playwright Chromium unavailable ({exc}); run `uv run --group e2e playwright install chromium`")
        yield b
        b.close()


@pytest.fixture
def page(browser):
    context = browser.new_context(viewport={"width": 1400, "height": 1000})
    pg = context.new_page()
    pg.set_default_timeout(30_000)
    yield pg
    context.close()
