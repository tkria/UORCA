"""An in-app folder browser for path text boxes.

Streamlit has no folder picker, and a native OS dialog would open on the machine running the
server, not the viewer's. This browser lists folders on the server, so it works for a local app
and for one served from a remote host alike.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import streamlit as st


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
