"""Folder pickers for path text boxes.

Streamlit has no folder picker. Two are offered here:

- an in-app browser that lists folders on the server, so it works for a local app and for one
  served from a remote host alike;
- on macOS, the Finder "Choose Folder" dialog. A native dialog opens on the machine running
  the server, so it is offered only when the viewer's browser is on that same machine.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional

import streamlit as st

# osascript exits 1 with this AppleScript error number when the user clicks Cancel.
_USER_CANCELLED = "(-128)"

# The path is passed as an argument, never interpolated into the script.
_CHOOSE_FOLDER_SCRIPT = [
    "on run argv",
    "activate",
    'POSIX path of (choose folder with prompt "Choose a UORCA results folder" '
    "default location (POSIX file (item 1 of argv)))",
    "end run",
]


class FolderDialogError(RuntimeError):
    """The native folder dialog failed for a reason other than the user cancelling it."""


def finder_dialog_available(platform: str, viewer_ip: Optional[str]) -> bool:
    """True when the Finder dialog would open on the viewer's own screen.

    That needs a macOS server and a browser on the same machine. Streamlit reports no IP
    (``None``) for a viewer on localhost.
    """
    return platform == "darwin" and viewer_ip in (None, "127.0.0.1", "::1")


def choose_folder_with_finder(start: Path) -> Optional[Path]:
    """Open the macOS "Choose Folder" dialog at ``start``; return the chosen folder.

    Returns ``None`` when the user cancels. Raises ``FolderDialogError`` on any other failure.
    Blocks until the dialog closes.
    """
    args = ["osascript"]
    for line in _CHOOSE_FOLDER_SCRIPT:
        args += ["-e", line]
    args.append(str(start))
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise FolderDialogError(f"Could not open the Finder dialog: {exc}") from exc
    if result.returncode != 0:
        if _USER_CANCELLED in result.stderr:
            return None
        raise FolderDialogError(
            f"The Finder dialog failed (osascript exit {result.returncode}): {result.stderr.strip()}"
        )
    chosen = result.stdout.strip()
    if not chosen:
        raise FolderDialogError("The Finder dialog returned no folder.")
    return Path(chosen)


def list_subfolders(folder: Path) -> list[Path]:
    """Return the visible subfolders of ``folder``, sorted case-insensitively.

    Hidden folders (leading ``.``) are left out. Raises ``OSError`` (including
    ``PermissionError``) if ``folder`` cannot be listed, so the caller can surface it.
    """
    return sorted(
        (p for p in folder.iterdir() if p.is_dir() and not p.name.startswith(".")),
        key=lambda p: p.name.lower(),
    )


def start_folder(typed: str) -> Path:
    """The folder the browser opens at: the typed path if it is an existing folder, else HOME."""
    if typed.strip():
        candidate = Path(typed.strip()).expanduser()
        if candidate.is_dir():
            return candidate.resolve()
    return Path.home()


def render_folder_browser(target_key: str, *, label: str = "Browse for a folder") -> None:
    """Render a folder browser that writes the chosen folder into the text box ``target_key``.

    Must be called after the text box with key ``target_key`` is created; the chosen path is
    written from a button callback, which runs before that widget is instantiated again.
    """
    dir_key = f"{target_key}_browse_dir"
    pick_key = f"{target_key}_browse_pick"

    def _go(folder: Path) -> None:
        st.session_state[dir_key] = str(folder)

    def _open_picked() -> None:
        picked: Optional[str] = st.session_state.get(pick_key)
        if picked:
            st.session_state[dir_key] = picked
        st.session_state[pick_key] = None

    def _use(folder: Path) -> None:
        st.session_state[target_key] = str(folder)

    finder_error_key = f"{target_key}_finder_error"

    def _choose_in_finder() -> None:
        st.session_state.pop(finder_error_key, None)
        try:
            chosen = choose_folder_with_finder(start_folder(st.session_state.get(target_key, "")))
        except FolderDialogError as exc:
            st.session_state[finder_error_key] = str(exc)
            return
        if chosen is not None:
            _use(chosen)
            st.session_state[dir_key] = str(chosen)

    if finder_dialog_available(sys.platform, st.context.ip_address):
        st.button("Choose in Finder…", key=f"{target_key}_finder", icon="📂",
                  on_click=_choose_in_finder,
                  help="Opens the macOS folder dialog. It can open behind the browser window.")
        if finder_error_key in st.session_state:
            st.error(st.session_state[finder_error_key])

    # A toggle, not an expander: the import panel itself sits in an expander on the Explore page,
    # and Streamlit advises against nesting expanders.
    if not st.toggle(label, key=f"{target_key}_browse_open"):
        return
    with st.container(border=True):
        if dir_key not in st.session_state:
            st.session_state[dir_key] = str(start_folder(st.session_state.get(target_key, "")))
        current = Path(st.session_state[dir_key])

        st.markdown(f"**Current folder:** `{current}`")
        c_up, c_home, c_use = st.columns([1, 1, 2])
        c_up.button("⬆ Up", key=f"{target_key}_browse_up", on_click=_go, args=(current.parent,),
                    disabled=current.parent == current, use_container_width=True)
        c_home.button("🏠 Home", key=f"{target_key}_browse_home", on_click=_go, args=(Path.home(),),
                      use_container_width=True)
        c_use.button("Use this folder", key=f"{target_key}_browse_use", type="primary",
                     on_click=_use, args=(current,), use_container_width=True)

        try:
            subfolders = list_subfolders(current)
        except OSError as exc:
            st.error(f"Cannot list `{current}`: {exc}")
            return

        if not subfolders:
            st.caption("No subfolders here.")
            return
        st.selectbox(
            f"Open a subfolder ({len(subfolders)})",
            [str(p) for p in subfolders],
            index=None,
            format_func=lambda s: f"📁 {Path(s).name}",
            placeholder="Choose a subfolder to open",
            key=pick_key,
            on_change=_open_picked,
        )
