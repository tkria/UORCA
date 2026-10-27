"""Render one Explore tab (or one app page) so that its failure cannot stop the others.

Each tab imports its own module inside its own ``with tab:`` block. If the import or the
render raises, the guard shows the error in that tab (``st.error`` with the exception type and
message, and the full traceback in a collapsed expander) and logs it with
``logger.exception``. The error is never swallowed: it stays visible where it happened, and
the remaining tabs still render.

Streamlit's control-flow signals (``st.rerun``, ``st.stop``, ``st.switch_page``) derive from
``BaseException``, not ``Exception``, so they pass through the guard unchanged.

Test-only switch
----------------
``UORCA_TEST_BREAK_TAB`` makes the guard raise ``ImportError`` before it imports the named
tab or page, as if that module were broken. Its value is a comma-separated list of keys; a
key is the module's last name without a ``_tab`` suffix (``ai_assistant`` for
``uorca.gui.components.ai_assistant_tab``, ``identify`` for ``uorca.gui.pages.identify``).
The browser tests use it to prove that one broken tab leaves the others working. It is read
only here, and it has no effect when unset.
"""

from __future__ import annotations

import importlib
import logging
import os
import traceback
from typing import Any

import streamlit as st

logger = logging.getLogger(__name__)

BREAK_TAB_ENV = "UORCA_TEST_BREAK_TAB"


def tab_key(module_path: str) -> str:
    """Return the ``UORCA_TEST_BREAK_TAB`` key for a module path."""
    return module_path.rsplit(".", 1)[-1].removesuffix("_tab")


def _raise_if_broken_on_purpose(module_path: str) -> None:
    keys = {k.strip() for k in os.environ.get(BREAK_TAB_ENV, "").split(",") if k.strip()}
    if tab_key(module_path) in keys:
        raise ImportError(
            f"{module_path} was broken on purpose by {BREAK_TAB_ENV}={os.environ[BREAK_TAB_ENV]!r} "
            "(test-only switch)"
        )


def render_tab_safely(
    name: str, module_path: str, func_name: str, kind: str = "tab", /, **kwargs: Any
) -> bool:
    """Import ``module_path``, call ``func_name(**kwargs)``, and contain any failure.

    Args:
        name: the label the user sees, used in the error message ("The {name} tab failed").
        module_path: dotted module path, imported lazily here.
        func_name: the render function in that module.
        kind: "tab" or "page", for the error message.
        **kwargs: passed to the render function.

    Returns:
        True when the tab rendered, False when it failed and the error was shown.
    """
    try:
        _raise_if_broken_on_purpose(module_path)
        module = importlib.import_module(module_path)
        render = getattr(module, func_name)
        render(**kwargs)
    except Exception as exc:
        logger.exception("The %s %s failed to load (%s.%s)", name, kind, module_path, func_name)
        st.error(f"The {name} {kind} failed to load: {type(exc).__name__}: {exc}")
        with st.expander("Error details (traceback)", expanded=False):
            st.code(traceback.format_exc(), language="text")
        return False
    return True
