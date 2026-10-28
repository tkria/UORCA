"""Sidebar dataset selection keeps earlier choices across searches."""

import pytest

from uorca.gui.components.sidebar_controls import merge_dataset_selection


@pytest.mark.unit
def test_selection_persists_across_two_searches():
    # Search "GSE295568", tick it, Apply
    sel = merge_dataset_selection(set(), visible={"GSE295568"}, checked={"GSE295568"})
    # Search "GSE265830", tick it, Apply: GSE295568 is hidden and must stay selected
    sel = merge_dataset_selection(sel, visible={"GSE265830"}, checked={"GSE265830"})
    assert sel == {"GSE295568", "GSE265830"}


@pytest.mark.unit
def test_unticking_a_visible_dataset_removes_only_it():
    sel = {"GSE1", "GSE2", "GSE3"}
    sel = merge_dataset_selection(sel, visible={"GSE2", "GSE3"}, checked={"GSE3"})
    assert sel == {"GSE1", "GSE3"}


@pytest.mark.unit
def test_empty_search_result_keeps_selection():
    assert merge_dataset_selection({"GSE1"}, visible=set(), checked=set()) == {"GSE1"}
