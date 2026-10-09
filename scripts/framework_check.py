#!/usr/bin/env python3
"""Run the central TI framework validator on this project (local and offline use).

Managed by the TI research software framework; do not edit. The validator itself
is not copied into projects. GitHub Actions runs it centrally through
.github/workflows/framework-check.yml. Locally, this shim runs it from, in order:

1. TI_FRAMEWORK_ROOT (a checkout of tkria/ti-research-software-framework);
2. the framework repository itself, when this file lives inside it;
3. the newest installed TI skill in ~/.claude/skills (installed by
   scripts/install-skill.sh, which works offline once installed).

    python scripts/framework_check.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
SKILLS = ("new-research-project", "adopt-research-project")


def _version_key(text: str) -> tuple:
    parts = []
    for piece in str(text).split("."):
        parts.append(int(piece) if piece.isdigit() else -1)
    return tuple(parts)


def candidates():
    root = os.environ.get("TI_FRAMEWORK_ROOT", "").strip()
    if root:
        yield Path(root).expanduser() / "shared" / "framework_check.py", "TI_FRAMEWORK_ROOT"
    yield PROJECT / "shared" / "framework_check.py", "framework repository checkout"
    installed = []
    home = Path(os.environ.get("HOME", "~")).expanduser()
    for skill in SKILLS:
        skill_dir = home / ".claude" / "skills" / skill
        script = skill_dir / "scripts" / "_shared" / "framework_check.py"
        if script.is_file():
            try:
                version = json.loads((skill_dir / "TOOLKIT.json").read_text(encoding="utf-8")).get("version", "0")
            except (OSError, ValueError):
                version = "0"
            installed.append((_version_key(version), script, f"installed skill {skill}"))
    for _, script, label in sorted(installed, key=lambda item: item[0], reverse=True):
        yield script, label


def main() -> int:
    for script, label in candidates():
        if script.is_file():
            print(f"Using the central validator from {label}.", flush=True)
            return subprocess.call([sys.executable, str(script), "--root", str(PROJECT), *sys.argv[1:]])
    print("The TI framework validator was not found. Install the framework skills once:\n"
          "  gh repo clone tkria/ti-research-software-framework\n"
          "  ./ti-research-software-framework/scripts/install-skill.sh\n"
          "or point TI_FRAMEWORK_ROOT at a checkout of tkria/ti-research-software-framework.\n"
          "In GitHub Actions the framework-check workflow runs the central validator automatically.",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
