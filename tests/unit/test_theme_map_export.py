"""Unit tests for the standalone-HTML export of the theme map (spec 5.8, 9.4-9.6).

Nothing here opens a browser or renders a page — no agent in this workflow can see one.
Everything is asserted by string containment, by JSON round-trip of the payload the page
embeds, and by parsing the document with :mod:`html.parser`.

No Streamlit session is started and no API is called.
"""

from __future__ import annotations

import html
import json
import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from uorca.gui.components.theme_map_export import (
    DIMMED_RGBA,
    PLOT_DIV_ID,
    ThemeMapExportError,
    build_standalone_html,
    export_filename,
    highlight_colors,
    point_scores,
    relevance_point_colors,
    theme_point_colors,
    threshold_steps,
    with_alpha,
)
from uorca.gui.components.theme_map_panel import (
    DEFAULT_HIGHLIGHT_THRESHOLD,
    HIGHLIGHT_THRESHOLD_MAX,
    HIGHLIGHT_THRESHOLD_MIN,
    HIGHLIGHT_THRESHOLD_STEP,
    is_highlighted,
)
from uorca.identification.theme_map.types import (
    DatasetPoint,
    Provenance,
    Theme,
    ThemeMapConfig,
    ThemeMapResult,
)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _point(
    accession: str,
    *,
    x: float = 0.0,
    y: float = 0.0,
    theme_id: int = 0,
    theme_prob: float = 1.0,
    relevance: float | None = 5.0,
    query_sim: float = 0.5,
    query_sim_rank: int = 1,
) -> DatasetPoint:
    return DatasetPoint(
        geo_accession=accession,
        x=x,
        y=y,
        theme_id=theme_id,
        theme_prob=theme_prob,
        relevance_score=relevance,
        query_sim=query_sim,
        query_sim_rank=query_sim_rank,
    )


def _theme(theme_id: int, label: str, **kwargs) -> Theme:
    defaults = dict(
        description=f"Description {theme_id}",
        n=2,
        keywords=["alpha", "beta"],
        representatives=["A study"],
        mean_relevance=3.0,
        max_relevance=6.0,
    )
    defaults.update(kwargs)
    return Theme(id=theme_id, label=label, **defaults)  # type: ignore[arg-type]


PROVENANCE = Provenance(
    embedding_model="text-embedding-3-small",
    seed=42,
    omp_num_threads="8",
    sklearn_version="1.7.0",
    numpy_version="2.3.1",
    hdbscan_version="0.8.40",
    n_datasets=7,
    n_themes=2,
    noise_fraction=2.0 / 7.0,
    estimated_cost_usd=0.0038,
    timestamp="2026-09-16T12:00:00+00:00",
    config=ThemeMapConfig(),
)


POINTS = [
    _point("GSE1", x=0.0, y=0.0, theme_id=0, relevance=9.0),
    _point("GSE2", x=1.0, y=0.5, theme_id=0, relevance=7.0),
    _point("GSE3", x=5.0, y=5.0, theme_id=1, relevance=2.5),
    _point("GSE4", x=5.5, y=4.5, theme_id=1, relevance=0.0),
    _point("GSE5", x=6.0, y=5.5, theme_id=1, relevance=None),
    _point("GSE6", x=-4.0, y=3.0, theme_id=-1, theme_prob=0.0, relevance=8.5),
    _point("GSE7", x=-4.5, y=2.0, theme_id=-1, theme_prob=0.0, relevance=None),
]

THEMES = [
    _theme(0, "Hepatic PPAR&gamma; signalling", n=2),
    _theme(1, "R&D pipeline <models>", n=3, keywords=["r&d", "xeno<graft>"]),
]

TITLES = {
    "GSE1": "Liver PPARg agonist timecourse",
    "GSE2": "<script>alert(1)</script>",
    "GSE3": "R&D screen of 40 compounds",
    "GSE4": "Fasting vs fed, 5 & 10 days",
    "GSE5": "Unscored control series",
    "GSE6": "Unthemed but highly relevant",
    "GSE7": "Unthemed and unscored",
}


def _result(points=POINTS, themes=THEMES, provenance=PROVENANCE) -> ThemeMapResult:
    embeddings = np.zeros((len(points), 4), dtype=np.float32)
    return ThemeMapResult(
        points=list(points),
        themes=list(themes),
        embeddings=embeddings,
        provenance=provenance,
    )


@pytest.fixture(scope="module")
def document() -> str:
    return build_standalone_html(_result(), TITLES, "run_2026_09_16")


def _payload(document: str) -> dict:
    match = re.search(
        r'<script id="theme-map-data" type="application/json">(.*?)</script>',
        document,
        re.S,
    )
    assert match is not None, "the page embeds no theme-map-data payload"
    return json.loads(match.group(1))


def _vendor_free(document: str) -> str:
    """The document with the vendored plotly.js bundle elided.

    Everything this module authors lives outside that bundle, so this is the text that
    must be free of external references.
    """
    return re.sub(
        r"<script[^>]*>.{100000,}?</script>", "<script>PLOTLY_BUNDLE</script>", document, flags=re.S
    )


# ---------------------------------------------------------------------------
# Offline: no CDN, nothing fetched
# ---------------------------------------------------------------------------


def test_no_external_script_or_stylesheet(document: str) -> None:
    assert re.search(r"<script[^>]*\ssrc\s*=", document) is None
    assert re.search(r"<link[^>]*\shref\s*=", document) is None


def test_nothing_we_author_references_a_cdn(document: str) -> None:
    """No external resource is ever LOADED. Outbound hyperlinks are a different thing.

    Narrowed 2026-09-17. This test previously banned the substring ``https://`` anywhere
    outside the plotly bundle, which was true while the page had no links. The dataset
    table gives every accession a GEO hyperlink, which the reader clicks deliberately and
    which the page never fetches, so the blanket ban would now reject a correct page.

    The property that matters is unchanged and is asserted more precisely here: the page
    must not fetch anything. Every remaining absolute URL must sit in an ``<a href>``, in
    the JSON payload's ``u`` field that such a link is built from, or in the prose that
    names ncbi.nlm.nih.gov. Nothing may appear in a ``src``, a stylesheet ``@import`` or
    a CSS ``url()``.
    """
    outside_bundle = _vendor_free(document)
    assert "PLOTLY_BUNDLE" in outside_bundle, "the plotly bundle was not located"
    assert "cdn.plot.ly" not in outside_bundle

    # Nothing is fetched.
    assert re.search(r"\ssrc\s*=\s*[\"']https?:", outside_bundle) is None
    assert re.search(r"@import\s+(url\()?[\"']?https?:", outside_bundle) is None
    assert re.search(r"url\(\s*[\"']?https?:", outside_bundle) is None

    # Every absolute URL that remains is a hyperlink target, not a resource.
    without_links = re.sub(r'<a\s[^>]*href="https?://[^"]*"', "<a>", outside_bundle)
    without_links = re.sub(r'"u":\s*"https?://[^"]*"', '"u":""', without_links)
    without_links = without_links.replace("ncbi.nlm.nih.gov", "")
    without_links = re.sub(
        r'link\.href\s*=\s*row\.u;|"https://www\.ncbi[^"]*"', "", without_links
    )
    leftover = [
        match.group(0)
        for match in re.finditer(r"https?://[^\s\"'<>)]{0,60}", without_links)
    ]
    assert leftover == [], f"unexpected absolute URLs outside a hyperlink: {leftover}"


def test_the_only_cdn_string_is_plotlys_inert_topojson_default(document: str) -> None:
    """plotly.js ships one `https://cdn.plot.ly/` literal: the `topojsonURL` default.

    A scatter plot never dereferences it, and it cannot be stripped from a minified
    4.6 MB bundle without corrupting the vendored library. This test pins it at exactly
    one occurrence, in that context, so a future plotly that adds a real CDN fetch fails
    here instead of shipping a page that needs the network.
    """
    assert document.count("cdn.plot.ly") == 1
    index = document.index("cdn.plot.ly")
    assert "topojsonURL" in document[index - 200 : index]


def test_plotly_library_and_div_are_inlined(document: str) -> None:
    assert "plotly.js v" in document
    assert "Plotly.newPlot" in document
    assert 'id="theme-map-plot"' in document
    # The bundle really is present, not a stub.
    assert len(document) > 1_000_000


# ---------------------------------------------------------------------------
# Content
# ---------------------------------------------------------------------------


def test_header_carries_run_id_counts_and_noise(document: str) -> None:
    assert "run_2026_09_16" in document
    assert '<span class="stat-value">7</span><span class="stat-label">datasets</span>' in document
    assert '<span class="stat-value">2</span><span class="stat-label">themes</span>' in document
    assert (
        '<span class="stat-value">28.6%</span><span class="stat-label">unthemed</span>'
        in document
    )


def test_every_theme_label_appears(document: str) -> None:
    for theme in THEMES:
        assert html.escape(theme.label) in document


def test_unthemed_row_appears(document: str) -> None:
    assert "Unthemed (2)" in document


def test_provenance_block_names_model_seed_and_library_versions(document: str) -> None:
    for expected in (
        "text-embedding-3-small",
        "42",
        "1.7.0",
        "2.3.1",
        "0.8.40",
        "OMP_NUM_THREADS",
    ):
        assert expected in document


def test_provenance_block_carries_the_full_parameter_set(document: str) -> None:
    for expected in ("hdbscan_min_cluster_size", "tsne_perplexity", "covariance_eigh", "leaf"):
        assert expected in document


def test_provenance_does_not_promise_regeneration_from_the_seed(document: str) -> None:
    """Spec 5.8 forbids any claim that the map regenerates from the seed."""
    assert "not regenerable" in document or "not promised" in document


# ---------------------------------------------------------------------------
# Escaping
# ---------------------------------------------------------------------------


def test_data_text_is_escaped(document: str) -> None:
    assert "<script>alert(1)</script>" not in document
    # plotly additionally writes "/" as \u002f inside its own JSON, so the closing
    # half of the escaped tag is "&lt;\\u002fscript&gt;" there rather than "&lt;/script&gt;".
    assert "&lt;script&gt;alert(1)" in document
    assert "alert(1)&lt;" in document
    assert "R&amp;D screen of 40 compounds" in document
    assert "R&D screen of 40 compounds" not in document


def test_escaped_text_survives_into_the_embedded_payload(document: str) -> None:
    payload = _payload(document)
    labels = [annotation["text"] for annotation in payload["annotations"]]
    assert "R&amp;D pipeline &lt;models&gt;" in labels


# ---------------------------------------------------------------------------
# The embedded arrays drive the controls
# ---------------------------------------------------------------------------


def test_score_array_has_one_entry_per_point(document: str) -> None:
    payload = _payload(document)
    assert len(payload["scores"]) == len(POINTS)
    assert payload["scores"] == [9.0, 7.0, 2.5, 0.0, None, 8.5, None]


def test_colour_arrays_have_one_entry_per_point(document: str) -> None:
    payload = _payload(document)
    assert len(payload["relevanceColors"]) == len(POINTS)
    assert len(payload["themeColors"]) == len(POINTS)
    # An unscored point is never coloured by either view; it keeps its own grey trace.
    assert payload["relevanceColors"][4] is None
    assert payload["themeColors"][4] is None


def test_the_colour_array_the_page_builds_fits_the_scored_trace(document: str) -> None:
    """The page pushes one colour per scored point, in ``points`` order.

    ``locate_traces`` already refuses a figure whose scored trace is not exactly that, so
    this pins the other half: the array the browser assembles is the same length.
    """
    payload = _payload(document)
    built_by_the_page = [score for score in payload["scores"] if score is not None]
    assert len(built_by_the_page) == sum(1 for p in POINTS if p.relevance_score is not None)


def test_the_embedded_rule_reproduces_is_highlighted_at_every_step(document: str) -> None:
    """The page's `score !== null && score >= threshold` must be `is_highlighted`."""
    payload = _payload(document)
    scores = payload["scores"]
    for threshold in payload["thresholds"]:
        in_page = [score is not None and score >= threshold for score in scores]
        in_panel = [is_highlighted(point, threshold) for point in POINTS]
        assert in_page == in_panel, f"threshold {threshold}"


def test_one_caption_per_threshold_step(document: str) -> None:
    payload = _payload(document)
    assert len(payload["captions"]) == len(payload["thresholds"])
    assert payload["thresholds"][0] == HIGHLIGHT_THRESHOLD_MIN
    assert payload["thresholds"][-1] == HIGHLIGHT_THRESHOLD_MAX
    assert "3 of 7 datasets at or above 7.0" in payload["captions"][14]


def test_payload_opens_at_the_requested_threshold_and_view(document: str) -> None:
    payload = _payload(document)
    assert payload["thresholds"][payload["thresholdIndex"]] == DEFAULT_HIGHLIGHT_THRESHOLD
    assert payload["colorBy"] == "Relevance"


def test_trace_indices_are_resolved_not_assumed(document: str) -> None:
    payload = _payload(document)
    traces = payload["traces"]
    assert traces["scored"] is not None
    assert traces["unscored"] is not None
    assert traces["colorbar"] is not None
    assert len({traces["scored"], traces["unscored"], traces["colorbar"]}) == 3


def test_annotations_are_present_and_suppressible() -> None:
    with_names = _payload(build_standalone_html(_result(), TITLES, "r"))
    assert len(with_names["annotations"]) == len(THEMES)
    without = _payload(build_standalone_html(_result(), TITLES, "r", show_theme_names=False))
    assert without["annotations"] == []


def test_theme_view_can_be_the_opening_view() -> None:
    payload = _payload(build_standalone_html(_result(), TITLES, "r", color_by="Theme"))
    assert payload["colorBy"] == "Theme"


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_threshold_steps_span_the_slider_range() -> None:
    steps = threshold_steps()
    assert steps[0] == HIGHLIGHT_THRESHOLD_MIN
    assert steps[-1] == HIGHLIGHT_THRESHOLD_MAX
    assert all(
        round(b - a, 6) == HIGHLIGHT_THRESHOLD_STEP for a, b in zip(steps, steps[1:])
    )


def test_with_alpha_handles_hex_and_rgb() -> None:
    assert with_alpha("#9aa0a6", 0.45) == "rgba(154,160,166,0.45)"
    assert with_alpha("rgb(68, 1, 84)", 0.85) == "rgba(68,1,84,0.85)"


def test_with_alpha_refuses_a_colour_it_cannot_parse() -> None:
    with pytest.raises(ThemeMapExportError):
        with_alpha("chartreuse", 0.5)


def test_point_scores_keeps_unscored_as_none() -> None:
    assert point_scores(POINTS) == [9.0, 7.0, 2.5, 0.0, None, 8.5, None]


def test_relevance_colors_are_monotone_on_the_ramp() -> None:
    colors = relevance_point_colors(POINTS, 10.0)
    assert colors[4] is None and colors[6] is None
    assert all(color.startswith("rgba(") for color in colors if color is not None)
    # Distinct scores get distinct colours.
    assert len({c for c in colors if c is not None}) == 5


def test_theme_colors_grey_the_unthemed() -> None:
    colors = theme_point_colors(POINTS)
    assert colors[0] == colors[1]
    assert colors[0] != colors[2]
    assert colors[5] == with_alpha("#9aa0a6", 0.85)


def test_export_filename_is_safe() -> None:
    assert export_filename("run 1/2") == "theme_map_run_1_2.html"


# ---------------------------------------------------------------------------
# The scored trace the page restyles
# ---------------------------------------------------------------------------


def _figure_traces(document: str) -> list[dict[str, Any]]:
    """The trace list plotly would hand the browser, parsed out of ``Plotly.newPlot``.

    Read from the shipped document rather than from a ``go.Figure`` in the test, so the
    assertions are against what a reader's browser actually receives. (It also keeps
    plotly's untyped ``Figure`` attributes out of this file; pyright's inference of them
    is budget-sensitive and perturbs error counts in unrelated modules.)
    """
    # The bundle itself mentions ``Plotly.newPlot(`` inside an error message, so the call
    # is located by its div id rather than by the function name alone.
    call = re.search(r'Plotly\.newPlot\(\s*"' + PLOT_DIV_ID + r'"\s*,', document)
    assert call is not None, "the page never calls Plotly.newPlot for the map"
    start = document.index("[", call.end())
    depth, in_string, escaped = 0, False, False
    for index in range(start, len(document)):
        char = document[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "[{":
            depth += 1
        elif char in "]}":
            depth -= 1
            if depth == 0:
                return json.loads(document[start : index + 1])
    raise AssertionError("the Plotly.newPlot trace list is not bracket-balanced")


def test_the_scored_trace_holds_every_scored_point(document: str) -> None:
    traces = _figure_traces(document)
    payload = _payload(document)
    scored = [p for p in POINTS if p.relevance_score is not None]
    assert len(traces[payload["traces"]["scored"]]["x"]) == len(scored)
    assert len(traces[payload["traces"]["unscored"]]["x"]) == len(POINTS) - len(scored)


def test_the_scored_trace_carries_explicit_colours_and_no_numeric_ramp(document: str) -> None:
    """plotly.js treats a trace with numeric cmin/cmax as colour-scaled.

    The page pushes rgba strings onto this trace, so the ramp must not be on it. The
    colourbar placeholder keeps it.
    """
    traces = _figure_traces(document)
    payload = _payload(document)
    marker = traces[payload["traces"]["scored"]]["marker"]
    assert "colorscale" not in marker
    assert "cmin" not in marker
    assert "cmax" not in marker
    assert marker["opacity"] == 1.0
    assert all(str(color).startswith("rgba(") for color in marker["color"])

    colorbar = traces[payload["traces"]["colorbar"]]["marker"]
    assert colorbar["showscale"] is True
    assert colorbar["cmax"] is not None


def test_the_unscored_trace_is_hollow_grey_and_untouched_by_the_controls(document: str) -> None:
    """Spec 9.5: an unscored dataset is never drawn like a low-scoring one."""
    traces = _figure_traces(document)
    payload = _payload(document)
    marker = traces[payload["traces"]["unscored"]]["marker"]
    assert marker["symbol"] == "circle-open"
    assert marker["color"] == "#9aa0a6"


def test_the_document_opens_with_everything_below_the_threshold_greyed(document: str) -> None:
    traces = _figure_traces(document)
    payload = _payload(document)
    colors = traces[payload["traces"]["scored"]]["marker"]["color"]
    scored = [p.relevance_score for p in POINTS if p.relevance_score is not None]
    for color, score in zip(colors, scored):
        assert color != DIMMED_RGBA if score >= 7.0 else color == DIMMED_RGBA


def test_highlight_colors_follows_the_view() -> None:
    relevance = relevance_point_colors(POINTS, 10.0)
    themes = theme_point_colors(POINTS)
    at_zero_relevance = highlight_colors(POINTS, 0.0, "Relevance", relevance, themes)
    at_zero_theme = highlight_colors(POINTS, 0.0, "Theme", relevance, themes)
    assert at_zero_relevance == [c for c in relevance if c is not None]
    assert at_zero_theme == [c for c in themes if c is not None]
    # At the top of the range nothing clears the bar, so everything greys.
    assert highlight_colors(POINTS, 10.0, "Theme", relevance, themes) == [DIMMED_RGBA] * 5


def test_the_document_ships_the_dimmed_grey_it_opens_with(document: str) -> None:
    assert DIMMED_RGBA in document


# ---------------------------------------------------------------------------
# Loud failure
# ---------------------------------------------------------------------------


def test_a_missing_title_raises_rather_than_exporting_a_blank_label() -> None:
    titles = dict(TITLES)
    del titles["GSE3"]
    with pytest.raises(Exception):
        build_standalone_html(_result(), titles, "r")


def test_an_unknown_colour_mode_raises() -> None:
    with pytest.raises(Exception):
        build_standalone_html(_result(), TITLES, "r", color_by="Rainbow")


# ---------------------------------------------------------------------------
# The page's own script, executed
# ---------------------------------------------------------------------------

_HARNESS = r"""
const fs = require('fs');
const payloadText = fs.readFileSync(process.argv[2], 'utf8');
const payload = JSON.parse(payloadText);
const restyles = [], relayouts = [];
global.Plotly = {
  restyle: (gd, upd, idx) => restyles.push([JSON.parse(JSON.stringify(upd)), idx]),
  relayout: (gd, upd) => relayouts.push(upd),
};
const slider = { value: String(payload.thresholdIndex), listeners: {},
  addEventListener(k, f) { this.listeners[k] = f; } };
const readout = { textContent: '' }, caption = { textContent: '' };
const radios = [
  { value: 'Relevance', checked: payload.colorBy === 'Relevance', addEventListener() {} },
  { value: 'Theme', checked: payload.colorBy === 'Theme', addEventListener() {} },
];
const byId = {
  'theme-map-data': { textContent: payloadText },
  'theme-map-plot': {}, 'threshold-slider': slider,
  'threshold-value': readout, 'highlight-caption': caption,
  'hide-below': { checked: payload.hideBelow, addEventListener() {} },
  'hidden-caption': { textContent: '' },
};
global.document = { getElementById: (id) => byId[id], querySelectorAll: () => radios };
eval(fs.readFileSync(process.argv[3], 'utf8'));

// Look restyles up by the trace they target, not by call order. The page issues a
// varying number of them -- the unscored trace is only restyled when there is one --
// so a positional index silently reads the wrong call as soon as that changes.
function forTrace(index) {
  for (var i = restyles.length - 1; i >= 0; i--) {
    if (restyles[i][1][0] === index) { return restyles[i][0]; }
  }
  return null;
}
function snapshot() {
  var scored = forTrace(payload.traces.scored);
  var colorbar = forTrace(payload.traces.colorbar);
  return {
    colors: scored['marker.color'][0],
    colorbarVisible: colorbar.visible[0],
    annotations: relayouts[relayouts.length - 1].annotations.length,
    readout: readout.textContent,
    caption: caption.textContent,
  };
}
const opening = snapshot();
radios[0].checked = false; radios[1].checked = true; slider.value = '0';
restyles.length = 0; relayouts.length = 0;
slider.listeners['input']();
console.log(JSON.stringify({ opening: opening, themeViewAtZero: snapshot() }));
"""


def _run_page_script(document: str, tmp_path: Path) -> dict:
    embedded = re.search(
        r'<script id="theme-map-data" type="application/json">(.*?)</script>', document, re.S
    )
    assert embedded is not None
    payload_text = embedded.group(1)
    script = re.findall(r"<script>(.*?)</script>", document, re.S)[-1]
    (tmp_path / "payload.json").write_text(payload_text, encoding="utf-8")
    (tmp_path / "apply.js").write_text(script, encoding="utf-8")
    (tmp_path / "harness.js").write_text(_HARNESS, encoding="utf-8")
    completed = subprocess.run(
        [
            "node",
            str(tmp_path / "harness.js"),
            str(tmp_path / "payload.json"),
            str(tmp_path / "apply.js"),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_page_script_colours_the_scored_trace_as_the_panel_would(
    document: str, tmp_path: Path
) -> None:
    """Run the page's own script against a stubbed DOM and a stubbed Plotly.

    Nobody in this workflow can see a rendered page, so the next best evidence that the
    controls work is to execute the script that drives them and check what it would have
    handed Plotly. Skipped, loudly, where node is absent.
    """
    payload = _payload(document)
    result = _run_page_script(document, tmp_path)

    opening = result["opening"]
    expected_open = [
        payload["relevanceColors"][i] if score >= DEFAULT_HIGHLIGHT_THRESHOLD else payload["dimmedColor"]
        for i, score in enumerate(payload["scores"])
        if score is not None
    ]
    assert opening["colors"] == expected_open
    assert opening["colorbarVisible"] is True
    assert opening["annotations"] == 0  # relevance view carries no theme labels
    assert opening["readout"] == "7.0"
    assert opening["caption"] == payload["captions"][payload["thresholdIndex"]]

    theme_view = result["themeViewAtZero"]
    assert theme_view["colors"] == [c for c in payload["themeColors"] if c is not None]
    assert theme_view["colorbarVisible"] is False
    assert theme_view["annotations"] == len(THEMES)
    assert theme_view["readout"] == "0.0"


# ---------------------------------------------------------------------------
# Well-formedness
# ---------------------------------------------------------------------------


class _BalanceChecker(HTMLParser):
    """Track element nesting outside of raw-text elements."""

    VOID = {
        "area", "base", "br", "col", "embed", "hr", "img", "input",
        "link", "meta", "param", "source", "track", "wbr",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.problems: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag not in self.VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag in self.VOID:
            return
        if not self.stack:
            self.problems.append(f"</{tag}> with nothing open")
            return
        if self.stack[-1] != tag:
            self.problems.append(f"</{tag}> closes <{self.stack[-1]}>")
            return
        self.stack.pop()


def test_document_parses_and_every_element_is_balanced(document: str) -> None:
    # The vendored bundle is elided: it is 4.6 MB of minified JavaScript that html.parser
    # would walk character by character, and it is not what this module authors.
    parser = _BalanceChecker()
    parser.feed(_vendor_free(document))
    parser.close()
    assert parser.problems == []
    assert parser.stack == []


def test_document_is_a_complete_html_page(document: str) -> None:
    assert document.startswith("<!DOCTYPE html>")
    assert '<meta charset="utf-8">' in document
    assert "prefers-color-scheme" in document
    assert document.rstrip().endswith("</html>")


# ---------------------------------------------------------------------------
# The per-dataset table (2026-09-17, modelled on theme_map_explorer.html)
# ---------------------------------------------------------------------------

DETAILS_DF = pd.DataFrame(
    {
        "GEO_Accession": ["GSE1", "GSE2", "GSE3", "GSE4", "GSE5", "GSE6", "GSE7"],
        "Species": ["Mus musculus", "Homo sapiens", "R&D organism", "", "Danio rerio", "", "Mus musculus"],
        "Samples": [6, 12, 4, 8, 3, 20, 5],
        "Justification": [
            "Directly assays PPARg in <hepatocytes> & macrophages",
            "Off topic",
            "R&D screen, weak design",
            "",
            "Never scored",
            "Unthemed but on point",
            "",
        ],
        "GEO_URL": ["https://example.test/GSE1"] + [""] * 6,
    }
)


@pytest.fixture(scope="module")
def detailed_document() -> str:
    from uorca.gui.components.theme_map_export import dataset_details

    return build_standalone_html(
        _result(), TITLES, "run_2026_09_16", details=dataset_details(DETAILS_DF)
    )


def test_dataset_details_reads_every_column_it_wants() -> None:
    from uorca.gui.components.theme_map_export import dataset_details

    details = dataset_details(DETAILS_DF)

    assert details["GSE1"]["species"] == "Mus musculus"
    assert details["GSE1"]["samples"] == "6"
    assert details["GSE1"]["geo_url"] == "https://example.test/GSE1"
    assert "macrophages" in details["GSE1"]["justification"]


def test_dataset_details_tolerates_a_missing_column() -> None:
    """The archived runs do not all carry every column. A blank cell, never a crash."""
    from uorca.gui.components.theme_map_export import dataset_details

    thin = pd.DataFrame(DETAILS_DF[["GEO_Accession", "Species"]])
    details = dataset_details(thin)

    assert details["GSE1"]["species"] == "Mus musculus"
    assert details["GSE1"]["samples"] == ""
    assert details["GSE1"]["justification"] == ""


def test_dataset_details_raises_when_there_is_no_accession_column() -> None:
    from uorca.gui.components.theme_map_export import (
        ThemeMapExportError,
        dataset_details,
    )

    with pytest.raises(ThemeMapExportError, match="GEO_Accession"):
        dataset_details(pd.DataFrame({"Species": ["Mus musculus"]}))


def test_dataset_rows_carry_one_record_per_point(detailed_document: str) -> None:
    rows = _payload(detailed_document)["datasets"]

    assert len(rows) == len(POINTS)
    assert [row["a"] for row in rows] == [p.geo_accession for p in POINTS]


def test_dataset_rows_carry_raw_text_because_the_browser_uses_textContent(
    detailed_document: str,
) -> None:
    """Escaping here would show a reader ``R&amp;D`` where the study says ``R&D``."""
    rows = {row["a"]: row for row in _payload(detailed_document)["datasets"]}

    assert rows["GSE3"]["t"] == "R&D screen of 40 compounds"
    assert rows["GSE3"]["sp"] == "R&D organism"
    assert rows["GSE2"]["t"] == "<script>alert(1)</script>"


def test_the_raw_script_tag_still_never_escapes_the_payload(detailed_document: str) -> None:
    """Raw in the JSON is safe; raw in the document is not. The payload must be inert."""
    body = _vendor_free(detailed_document)

    assert "<script>alert(1)</script>" not in body
    assert "\\u003cscript\\u003ealert(1)\\u003c/script\\u003e" in body


def test_an_unscored_dataset_carries_a_null_score_not_a_zero(
    detailed_document: str,
) -> None:
    rows = {row["a"]: row for row in _payload(detailed_document)["datasets"]}

    assert rows["GSE5"]["r"] is None
    assert rows["GSE7"]["r"] is None
    assert rows["GSE4"]["r"] == 0.0


def test_a_dataset_without_a_geo_url_gets_one_built_from_its_accession(
    detailed_document: str,
) -> None:
    rows = {row["a"]: row for row in _payload(detailed_document)["datasets"]}

    assert rows["GSE1"]["u"] == "https://example.test/GSE1"
    assert rows["GSE3"]["u"].endswith("acc=GSE3")


def test_theme_names_map_every_theme_and_names_unthemed(detailed_document: str) -> None:
    names = _payload(detailed_document)["themeNames"]

    assert names["-1"] == "Unthemed"
    assert len(names) == len(THEMES) + 1
    assert "Hepatic" in names["0"]


def test_the_show_selector_lists_all_highlighted_and_every_theme(
    detailed_document: str,
) -> None:
    body = _vendor_free(detailed_document)

    assert 'id="show-select"' in body
    assert '<option value="all"' in body
    assert '<option value="highlighted"' in body
    for theme in THEMES:
        assert f'<option value="theme:{theme.id}"' in body
    assert '<option value="theme:-1"' in body


def test_the_show_selector_orders_themes_by_size(detailed_document: str) -> None:
    body = _vendor_free(detailed_document)

    # Theme 1 has three members, theme 0 has two, so theme 1 comes first.
    assert body.index('value="theme:1"') < body.index('value="theme:0"')


def test_the_dataset_table_has_a_body_a_caption_and_eight_headers(
    detailed_document: str,
) -> None:
    body = _vendor_free(detailed_document)

    assert 'id="dataset-body"' in body
    assert 'id="dataset-caption"' in body
    for header in (
        "Accession",
        "Title",
        "Species",
        "Samples",
        "Theme",
        "Relevance",
        "Query cos. (rank)",
        "Justification",
    ):
        assert f">{header}<" in body


def test_the_export_does_not_claim_a_distance_to_nearest_theme(
    detailed_document: str,
) -> None:
    """The explorer showed it. We do not persist it, so we must not imply we have it."""
    body = _vendor_free(detailed_document)

    assert "Dist. to nearest theme" not in body
    assert "deliberately absent" in body


def test_the_table_still_builds_without_any_details(document: str) -> None:
    """``details`` is optional: a caller with only titles must still get a table."""
    rows = _payload(document)["datasets"]

    assert len(rows) == len(POINTS)
    assert all(row["sp"] == "" for row in rows)
    assert all(row["u"].startswith("https://www.ncbi.nlm.nih.gov/geo/") for row in rows)


# ---------------------------------------------------------------------------
# Hiding below the threshold, in the export (decision 16, 2026-09-22)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def hiding_document() -> str:
    return build_standalone_html(
        _result(), TITLES, "run_2026_09_16", threshold=7.0, hide_below=True
    )


def test_the_export_carries_a_hide_checkbox_unticked_by_default(document: str) -> None:
    body = _vendor_free(document)

    assert 'id="hide-below"' in body
    assert 'id="hide-below" checked' not in body
    assert 'id="hidden-caption"' in body


def test_the_hide_checkbox_opens_ticked_when_the_panel_had_it_on(
    hiding_document: str,
) -> None:
    assert 'id="hide-below" checked' in _vendor_free(hiding_document)


def test_the_payload_carries_the_scored_coordinates_to_restore(document: str) -> None:
    payload = _payload(document)
    scored = [p for p in POINTS if p.relevance_score is not None]

    assert len(payload["scoredX"]) == len(scored)
    assert len(payload["scoredY"]) == len(scored)
    assert payload["scoredX"][0] == pytest.approx(scored[0].x)


def test_the_payload_carries_one_annotation_set_per_threshold(document: str) -> None:
    payload = _payload(document)

    assert len(payload["hiddenAnnotations"]) == len(payload["thresholds"])
    assert len(payload["hiddenCaptions"]) == len(payload["thresholds"])


def test_a_theme_with_no_surviving_member_loses_its_label_at_that_threshold(
    document: str,
) -> None:
    """GSE1 (9.0) and GSE2 (7.0) are theme 0; theme 1 tops out at 2.5."""
    payload = _payload(document)
    at = {t: payload["hiddenAnnotations"][i] for i, t in enumerate(payload["thresholds"])}

    labels_at_7 = sorted(a["text"] for a in at[7.0])
    labels_at_0 = sorted(a["text"] for a in at[0.0])

    assert len(labels_at_7) == 1, "only theme 0 survives a 7.0 cut"
    assert len(labels_at_0) == 2
    assert at[10.0] == [], "nothing survives a 10.0 cut, so nothing is labelled"


def test_the_opening_frame_already_hides_rather_than_waiting_for_the_script(
    hiding_document: str,
) -> None:
    """The first paint must match the controls. A flash of the wrong picture is a bug."""
    payload = _payload(hiding_document)

    assert payload["hideBelow"] is True
    # The figure's own scored trace carries nulls where a point is hidden.
    assert "null" in hiding_document[: hiding_document.index('id="theme-map-data"')]


def test_the_hidden_caption_reports_the_unscored_share(document: str) -> None:
    payload = _payload(document)
    index = payload["thresholds"].index(7.0)
    caption = payload["hiddenCaptions"][index]

    # POINTS carries two unscored datasets, GSE5 and GSE7.
    assert "never scored" in caption
    assert "2" in caption


def test_the_hidden_caption_is_empty_of_alarm_when_nothing_is_hidden(
    document: str,
) -> None:
    payload = _payload(document)
    index = payload["thresholds"].index(0.0)

    # Two of the seven points are unscored, so 0.0 still hides those two.
    assert "Hiding 2 of 7" in payload["hiddenCaptions"][index]
