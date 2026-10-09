"""A self-contained, offline HTML export of the theme map (spec 5.8, 9.4-9.6).

Spec 5.8 persists coordinates and labels as artifacts and forbids any promise that the
map regenerates from a seed. A file that someone can email, archive or open in six years
is the natural consequence of that: it carries the plot, the theme table and the full
provenance in one document that needs no server, no Python and no network.

**This module draws nothing new.** Every encoding decision -- the Viridis ramp, the
highlight threshold, the grey for a backgrounded point, the hollow marker for an unscored
one, the theme medoid, the label font scaling, the annotation list -- already lives in
:mod:`uorca.gui.components.theme_map_panel` and is imported from there. What this module
adds is a fixed trace layout the exported page can restyle, the per-point colour arrays
that drive it, and the document around it.

Dependencies flow one way: this module imports the panel, never the reverse at import
time. The panel's download button imports :func:`build_standalone_html` inside the render
function for exactly that reason.

Nothing here degrades quietly. A missing title, an unknown colour mode, a colour string
that cannot be parsed, or a figure whose traces do not line up with the points they were
built from all raise.
"""

from __future__ import annotations

import html
import json
import math
import re
from dataclasses import asdict, replace
from typing import Any, Mapping, Sequence

import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale

from uorca.gui.components.theme_map_panel import (
    COLOR_BY_OPTIONS,
    DEFAULT_HIGHLIGHT_THRESHOLD,
    DIMMED_COLOR,
    DIMMED_OPACITY,
    HIGHLIGHT_THRESHOLD_MAX,
    HIGHLIGHT_THRESHOLD_MIN,
    HIGHLIGHT_THRESHOLD_STEP,
    RELEVANCE_COLORSCALE,
    SCORED_MARKER,
    _theme_color,  # the panel's theme palette lookup; reused, never reimplemented
    build_theme_map_figure,
    build_theme_table,
    hidden_caption,
    highlight_caption,
    is_highlighted,
    theme_label_annotations,
)
from uorca.identification.theme_map.types import DatasetPoint, Provenance, Theme, ThemeMapResult

__all__ = [
    "ThemeMapExportError",
    "PLOT_DIV_ID",
    "PAYLOAD_SCRIPT_ID",
    "build_standalone_html",
    "export_filename",
    "dataset_details",
    "dataset_rows",
    "theme_name_map",
    "escape_titles",
    "escape_themes",
    "point_scores",
    "relevance_point_colors",
    "highlight_colors",
    "prepare_scored_trace",
    "theme_point_colors",
    "threshold_steps",
    "threshold_index",
    "with_alpha",
    "TraceRoles",
    "locate_traces",
]


class ThemeMapExportError(RuntimeError):
    """The export cannot be built from what it was handed, and will not guess."""


PLOT_DIV_ID = "theme-map-plot"
PAYLOAD_SCRIPT_ID = "theme-map-data"

#: Alpha of a highlighted point. The panel carries it as a trace-level opacity, but the
#: exported page recolours a single trace per point, and per-point opacity is not a
#: scattergl channel -- so the alpha is baked into each rgba string instead.
HIGHLIGHT_ALPHA = SCORED_MARKER.opacity


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

_HEX_RE = re.compile(r"^#([0-9a-fA-F]{6})$")
_RGB_RE = re.compile(r"^rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)$")


def with_alpha(color: str, alpha: float) -> str:
    """``#rrggbb`` or ``rgb(r, g, b)`` -> ``rgba(r,g,b,a)``.

    Raises on anything else rather than passing an unparsed string through to the page,
    where it would fail silently as a black dot.
    """
    text = str(color).strip()
    hex_match = _HEX_RE.match(text)
    if hex_match:
        digits = hex_match.group(1)
        red, green, blue = (int(digits[i : i + 2], 16) for i in (0, 2, 4))
    else:
        rgb_match = _RGB_RE.match(text)
        if not rgb_match:
            raise ThemeMapExportError(
                f"Cannot add an alpha channel to colour {color!r}: the export understands "
                "'#rrggbb' and 'rgb(r, g, b)' only."
            )
        red, green, blue = (int(group) for group in rgb_match.groups())
    return f"rgba({red},{green},{blue},{alpha:g})"


DIMMED_RGBA = with_alpha(DIMMED_COLOR, DIMMED_OPACITY)


# ---------------------------------------------------------------------------
# Per-point arrays. One entry per mapped dataset, in ``points`` order.
# ---------------------------------------------------------------------------


def point_scores(points: Sequence[DatasetPoint]) -> list[float | None]:
    """The relevance score of every mapped dataset, ``None`` where it was never scored.

    The distinction must survive into the page: spec 9.5 forbids drawing an unscored
    dataset like a low-scoring one, and the page's highlight rule
    (``score !== null && score >= threshold``) is exactly
    :func:`~uorca.gui.components.theme_map_panel.is_highlighted`.
    """
    return [None if point.relevance_score is None else float(point.relevance_score) for point in points]


def relevance_point_colors(
    points: Sequence[DatasetPoint], ramp_max: float
) -> list[str | None]:
    """The Viridis colour of every scored dataset, ``None`` for the unscored ones.

    Sampled in Python at the same ``cmin``/``cmax`` the figure's colourbar advertises, so
    a point's colour in the page matches the ramp drawn beside it.
    """
    span = float(ramp_max) - HIGHLIGHT_THRESHOLD_MIN
    if span <= 0:
        raise ThemeMapExportError(
            f"The relevance ramp has no range (cmin={HIGHLIGHT_THRESHOLD_MIN}, "
            f"cmax={ramp_max}); the colourbar the figure carries is unusable."
        )
    colors: list[str | None] = []
    for point in points:
        if point.relevance_score is None:
            colors.append(None)
            continue
        fraction = (float(point.relevance_score) - HIGHLIGHT_THRESHOLD_MIN) / span
        fraction = min(max(fraction, 0.0), 1.0)
        sampled = sample_colorscale(RELEVANCE_COLORSCALE, [fraction], colortype="rgb")[0]
        if not isinstance(sampled, str):
            raise ThemeMapExportError(
                f"Sampling {RELEVANCE_COLORSCALE} returned {type(sampled).__name__}, not a "
                "colour string; the export cannot bake an alpha channel into it."
            )
        colors.append(with_alpha(sampled, HIGHLIGHT_ALPHA))
    return colors


def theme_point_colors(points: Sequence[DatasetPoint]) -> list[str | None]:
    """The theme colour of every scored dataset, ``None`` for the unscored ones.

    An unscored point is grey and hollow in both views at every threshold (spec 9.5), so
    it has no theme colour to carry -- it is drawn by its own trace, which this array
    never touches.
    """
    return [
        None if point.relevance_score is None else with_alpha(_theme_color(point.theme_id), HIGHLIGHT_ALPHA)
        for point in points
    ]


def highlight_colors(
    points: Sequence[DatasetPoint],
    threshold: float,
    color_by: str,
    relevance_colors: Sequence[str | None],
    theme_colors: Sequence[str | None],
) -> list[str]:
    """The colour of every **scored** point at ``threshold``, in ``points`` order.

    This is the Python twin of the page's ``apply()`` loop and exists so the file's
    opening state is computed and tested here rather than only in the browser. Unscored
    points are absent: they are drawn by their own trace and never recoloured.
    """
    colors: list[str] = []
    for index, point in enumerate(points):
        if point.relevance_score is None:
            continue
        if not is_highlighted(point, threshold):
            colors.append(DIMMED_RGBA)
            continue
        source = theme_colors if color_by == "Theme" else relevance_colors
        color = source[index]
        if color is None:
            raise ThemeMapExportError(
                f"{point.geo_accession} is scored but has no {color_by.lower()} colour; "
                "the export's colour arrays and its points have drifted apart."
            )
        colors.append(color)
    return colors


def prepare_scored_trace(figure: go.Figure, scored_index: int, colors: Sequence[str]) -> None:
    """Put the scored trace onto explicit rgba colours, in place.

    The panel colours that trace numerically in the relevance view: an array of scores
    plus ``colorscale``/``cmin``/``cmax``. plotly.js decides a trace "has a colorscale"
    when ``cmin`` and ``cmax`` are numeric, and then pushes every entry of ``marker.color``
    through the scale function -- so a page that restyled rgba strings onto that trace
    would feed strings to a numeric scale. The exported figure therefore drops the ramp
    from this trace (it lives on the colourbar placeholder, which is untouched) and holds
    explicit colours instead.

    ``marker.opacity`` goes to 1 at the same time. plotly multiplies it into each colour's
    alpha, and the alpha is now carried per point -- 0.85 for a highlighted point, 0.45 for
    a backgrounded one, exactly the two values the panel uses.
    """
    marker = figure.data[scored_index].marker  # type: ignore[union-attr]
    marker.color = list(colors)
    marker.colorscale = None
    marker.cmin = None
    marker.cmax = None
    marker.showscale = False
    marker.opacity = 1.0


def threshold_steps() -> list[float]:
    """Every position of the page's threshold slider, matching the panel's control."""
    steps: list[float] = []
    value = HIGHLIGHT_THRESHOLD_MIN
    while value <= HIGHLIGHT_THRESHOLD_MAX + 1e-9:
        steps.append(round(value, 6))
        value += HIGHLIGHT_THRESHOLD_STEP
    return steps


def threshold_index(threshold: float) -> int:
    """The slider position for ``threshold``.

    The slider is an integer index rather than a float so the page never has to compare
    accumulated floating-point steps. A threshold that is not on the grid raises: silently
    snapping it would hand the user a file that does not show what they asked for.
    """
    steps = threshold_steps()
    for index, step in enumerate(steps):
        if abs(step - float(threshold)) < 1e-9:
            return index
    raise ThemeMapExportError(
        f"Threshold {threshold} is not one of the slider's positions "
        f"({HIGHLIGHT_THRESHOLD_MIN} to {HIGHLIGHT_THRESHOLD_MAX} in steps of "
        f"{HIGHLIGHT_THRESHOLD_STEP})."
    )


# ---------------------------------------------------------------------------
# Escaping. Every string that came from the data is escaped exactly once.
# ---------------------------------------------------------------------------


def escape_titles(titles: Mapping[str, str]) -> dict[str, str]:
    """HTML-escape every GEO title before it reaches the figure.

    The hover template is HTML, and the figure's JSON is written into a ``<script>``
    block, so an unescaped ``<`` in a title is both an injection and a way to end the
    script early. Plotly renders ``&lt;`` back to ``<`` in the hover box, so the user
    still reads the real title.
    """
    return {str(accession): html.escape(str(title)) for accession, title in titles.items()}


def escape_themes(themes: Sequence[Theme]) -> list[Theme]:
    """The same themes with every user-visible string escaped, for the figure."""
    return [
        replace(
            theme,
            label=html.escape(theme.label),
            description=html.escape(theme.description),
            keywords=[html.escape(keyword) for keyword in theme.keywords],
        )
        for theme in themes
    ]


def _json_for_script(value: Any) -> str:
    """JSON safe to embed in a ``<script>`` element.

    ``</script>`` inside a string literal would end the block, so the three characters
    that can start a tag or an entity are written as unicode escapes. ``json.loads``
    reads them back unchanged.
    """
    return (
        json.dumps(value, allow_nan=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


# ---------------------------------------------------------------------------
# Trace layout
# ---------------------------------------------------------------------------


class TraceRoles:
    """Which trace of the exported figure is which.

    The panel builds up to four traces and their composition depends on the threshold and
    the colour mode, so an exported page must not assume an index. These are resolved by
    reading each trace's ``customdata`` back, where ``row[0]`` is the accession.
    """

    __slots__ = ("colorbar", "scored", "unscored")

    def __init__(self, colorbar: int | None, scored: int | None, unscored: int | None) -> None:
        self.colorbar = colorbar
        self.scored = scored
        self.unscored = unscored

    def as_dict(self) -> dict[str, int | None]:
        return {"colorbar": self.colorbar, "scored": self.scored, "unscored": self.unscored}


def locate_traces(figure: go.Figure, points: Sequence[DatasetPoint]) -> TraceRoles:
    """Resolve the exported figure's traces, and verify they match ``points``.

    The scored trace must hold exactly the scored datasets, in ``points`` order, because
    the page's colour arrays are indexed by that order. A mismatch means the figure the
    panel builds has changed shape underneath this module, and it raises rather than
    shipping a page whose colours are attached to the wrong dots.
    """
    scored = [point.geo_accession for point in points if point.relevance_score is not None]
    unscored = [point.geo_accession for point in points if point.relevance_score is None]

    roles = TraceRoles(colorbar=None, scored=None, unscored=None)
    for index, trace in enumerate(figure.data):
        customdata = getattr(trace, "customdata", None) or []
        accessions = [str(row[0]) for row in customdata]
        if not accessions:
            if roles.colorbar is not None:
                raise ThemeMapExportError(
                    "The exported figure carries more than one trace without point data; "
                    "the colourbar placeholder cannot be identified."
                )
            roles.colorbar = index
        elif accessions == scored:
            roles.scored = index
        elif accessions == unscored:
            roles.unscored = index
        else:
            raise ThemeMapExportError(
                f"Trace {index} of the exported figure holds {len(accessions)} datasets "
                "that are neither exactly the scored set nor exactly the unscored set, in "
                "order. The figure the panel builds no longer matches what this export "
                "recolours."
            )

    if scored and roles.scored is None:
        raise ThemeMapExportError(
            "The exported figure has no trace holding the scored datasets, so the "
            "threshold control would have nothing to recolour."
        )
    if unscored and roles.unscored is None:
        raise ThemeMapExportError(
            "The exported figure has no trace holding the unscored datasets, so they "
            "would be missing from the page."
        )
    if roles.colorbar is None:
        raise ThemeMapExportError(
            "The exported figure has no colourbar trace, so the relevance ramp would have "
            "no legend."
        )
    return roles


def _ramp_max(figure: go.Figure, roles: TraceRoles) -> float:
    """The colourbar's ``cmax``, read off the figure rather than recomputed.

    The panel pins the ramp to the corpus score range. Recomputing it here would be a
    second source of truth that could drift from the ramp the page actually draws.
    """
    if roles.colorbar is None:
        raise ThemeMapExportError("No colourbar trace, so the relevance ramp has no range.")
    marker = figure.data[roles.colorbar].marker  # type: ignore[union-attr]
    cmax = getattr(marker, "cmax", None)
    if cmax is None:
        raise ThemeMapExportError(
            "The colourbar trace carries no cmax, so the export cannot sample the same "
            "ramp the page draws."
        )
    return float(cmax)


# ---------------------------------------------------------------------------
# Document fragments
# ---------------------------------------------------------------------------


def dataset_details(results: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Per-dataset columns for the export table, keyed by accession.

    Pulled from ``Dataset_identification_result.csv``. Every column is optional: the four
    run directories written before the current schema do not all carry ``Samples``, and a
    missing column must produce a blank cell rather than a crash or a fabricated value.

    ``Justification`` is the LLM's stated reason for the relevance score. It is the single
    most useful field for a reader who has the file but not the app, which is why it is
    carried even though it is the bulk of the payload.
    """
    accession_col = _accession_column_name(results)
    wanted = {
        "species": "Species",
        "samples": "Samples",
        "justification": "Justification",
        "geo_url": "GEO_URL",
    }
    details: dict[str, dict[str, Any]] = {}
    for _, row in results.iterrows():
        accession = str(row[accession_col])
        record: dict[str, Any] = {}
        for key, column in wanted.items():
            value = row.get(column) if column in results.columns else None
            record[key] = "" if value is None or pd.isna(value) else str(value).strip()
        details[accession] = record
    return details


def _accession_column_name(results: pd.DataFrame) -> str:
    for candidate in ("GEO_Accession", "Accession"):
        if candidate in results.columns:
            return candidate
    raise ThemeMapExportError(
        "The results table has neither a GEO_Accession nor an Accession column, so the "
        f"export cannot key its dataset rows. Columns present: {list(results.columns)}."
    )


def theme_name_map(themes: Sequence[Theme]) -> dict[str, str]:
    """Theme id (as a string, because JSON object keys are strings) to display label."""
    names = {str(theme.id): f"{theme.id} \u2014 {theme.label}" for theme in themes}
    names["-1"] = "Unthemed"
    return names


def dataset_rows(
    points: Sequence[DatasetPoint],
    titles: Mapping[str, str],
    details: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """One record per dataset for the export's bottom table.

    Strings are carried RAW, not HTML-escaped. The browser writes every cell through
    ``textContent``, which cannot execute markup, so escaping here would show a reader
    ``R&amp;D`` where the study says ``R&D``.

    ``QuerySim`` is carried because the explorer this table is modelled on carried it, and
    it is free once the embeddings exist. It is a weak signal: on the ground-truth set the
    LLM score gave a median rank of 10 of 1,199 and cosine similarity gave 70 (spec 5.7).
    Distance to the nearest theme, which that explorer also showed, is NOT carried -- it is
    not persisted, and a substitute under the same name would be a fabrication.
    """
    detail_map: Mapping[str, Mapping[str, Any]] = details or {}
    rows: list[dict[str, Any]] = []
    for point in points:
        accession = str(point.geo_accession)
        record = detail_map.get(accession, {})
        rows.append(
            {
                "a": accession,
                "t": str(titles.get(accession, "")),
                "sp": str(record.get("species", "")),
                "n": str(record.get("samples", "")),
                "th": int(point.theme_id),
                "r": None if point.relevance_score is None else float(point.relevance_score),
                "qs": float(point.query_sim),
                "qr": int(point.query_sim_rank),
                "j": str(record.get("justification", "")),
                "u": str(record.get("geo_url", ""))
                or f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={accession}",
            }
        )
    return rows


def _show_selector_html(points: Sequence[DatasetPoint], themes: Sequence[Theme]) -> str:
    """The Show selector of the explorer's focus dropdown, in document order by size."""
    counts: dict[int, int] = {}
    for point in points:
        counts[point.theme_id] = counts.get(point.theme_id, 0) + 1

    options = [
        f'<option value="all">All datasets ({len(points):,})</option>',
        '<option value="highlighted">Only at or above the threshold</option>',
    ]
    ordered = sorted(themes, key=lambda theme: counts.get(theme.id, 0), reverse=True)
    for theme in ordered:
        n = counts.get(theme.id, 0)
        label = html.escape(f"{theme.id} \u2014 {theme.label} ({n:,})")
        options.append(f'<option value="theme:{theme.id}">{label}</option>')
    if counts.get(-1):
        options.append(
            f'<option value="theme:-1">Unthemed ({counts[-1]:,})</option>'
        )
    return (
        '<label class="control-label" for="show-select">Show</label>\n'
        '<select id="show-select">' + "".join(options) + "</select>"
    )


def export_filename(run_id: str) -> str:
    """``theme_map_<run id>.html``, with every path-unsafe character flattened."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(run_id)).strip("_") or "run"
    return f"theme_map_{safe}.html"


def _optional_float(value: Any) -> float | None:
    """A relevance cell as a number, or ``None`` when it is genuinely absent.

    ``build_theme_table`` coerces the two relevance columns with ``pd.to_numeric``, so a
    cell is a float or ``NaN``. Anything else is a contract break and ``float()`` raises
    rather than quietly rendering an em dash over real data.
    """
    if value is None:
        return None
    number = float(value)
    return None if math.isnan(number) else number


def _format_number(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "&mdash;"
    return f"{value:.{digits}f}"


def _theme_table_html(table: pd.DataFrame) -> str:
    """The panel's theme table (spec 9.6), rendered as static HTML.

    The rows come from :func:`~uorca.gui.components.theme_map_panel.build_theme_table`, so
    the exported table is the same table -- including the ``Unthemed`` row whose
    ``Max relevance`` cell is the honest warning of spec 9.6.
    """
    header = "".join(f"<th>{html.escape(str(column))}</th>" for column in table.columns)
    body_rows: list[str] = []
    for _, row in table.iterrows():
        cells = [
            f"<td>{int(row['ThemeId'])}</td>",
            f"<td>{html.escape(str(row['Label']))}</td>",
            f"<td>{int(row['n']):,}</td>",
            f"<td>{_format_number(_optional_float(row['Mean relevance']))}</td>",
            f"<td>{_format_number(_optional_float(row['Max relevance']))}</td>",
            f"<td class=\"keywords\">{html.escape(str(row['Keywords']))}</td>",
        ]
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        '<table class="theme-table"><thead><tr>'
        + header
        + "</tr></thead><tbody>"
        + "".join(body_rows)
        + "</tbody></table>"
    )


def _provenance_rows(provenance: Provenance) -> list[tuple[str, str]]:
    """Everything spec section 6 persists, as label/value pairs."""
    threads = provenance.omp_num_threads
    return [
        ("Embedding model", str(provenance.embedding_model)),
        ("Seed", str(provenance.seed)),
        ("OMP_NUM_THREADS", "unset" if threads is None else str(threads)),
        ("scikit-learn", str(provenance.sklearn_version)),
        ("numpy", str(provenance.numpy_version)),
        ("hdbscan", str(provenance.hdbscan_version)),
        ("Datasets (n)", f"{provenance.n_datasets:,}"),
        ("Themes (k)", f"{provenance.n_themes:,}"),
        ("Noise fraction", f"{provenance.noise_fraction * 100:.1f}%"),
        ("Estimated cost", f"${provenance.estimated_cost_usd:.4f}"),
        ("Built", str(provenance.timestamp)),
    ]


def _provenance_html(provenance: Provenance) -> str:
    rows = "".join(
        f"<tr><th>{html.escape(label)}</th><td>{html.escape(value)}</td></tr>"
        for label, value in _provenance_rows(provenance)
    )
    config = "".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in sorted(asdict(provenance.config).items())
    )
    return (
        '<table class="kv">'
        + rows
        + "</table>"
        + '<h3>Pipeline parameters</h3><table class="kv">'
        + config
        + "</table>"
    )


_STYLE = """
:root {
  color-scheme: light dark;
  --bg: #f6f7f9;
  --surface: #ffffff;
  --border: #d9dce1;
  --text: #16191d;
  --muted: #5b626b;
  --accent: #2e6f9e;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #14171a;
    --surface: #1d2126;
    --border: #343a41;
    --text: #e8eaed;
    --muted: #a3a9b1;
    --accent: #7ab7e0;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0;
  padding: 24px 16px 64px;
  background: var(--bg);
  color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  font-size: 15px;
  line-height: 1.5;
}
main { max-width: 1100px; margin: 0 auto; }
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 17px; margin: 0 0 12px; }
h3 { font-size: 14px; margin: 20px 0 8px; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }
.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 18px 20px;
  margin-bottom: 18px;
}
.sub { color: var(--muted); margin: 0 0 14px; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px; }
.stats { display: flex; flex-wrap: wrap; gap: 28px; margin: 12px 0 4px; }
.stat { display: flex; flex-direction: column; }
.stat-value { font-size: 24px; font-weight: 600; line-height: 1.1; }
.stat-label { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; }
.note, .caption { color: var(--muted); font-size: 13px; margin: 10px 0 0; }
.controls { display: flex; flex-wrap: wrap; gap: 28px; align-items: center; margin-bottom: 4px; }
fieldset { border: 0; margin: 0; padding: 0; }
legend, .control-label { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; padding: 0; }
.radio-row { display: flex; gap: 14px; margin-top: 4px; }
.radio-row label { display: flex; align-items: center; gap: 5px; }
.slider-row { display: flex; align-items: center; gap: 10px; margin-top: 4px; }
input[type=range] { width: 260px; accent-color: var(--accent); }
#threshold-value { font-variant-numeric: tabular-nums; font-weight: 600; min-width: 2.5em; }
/* The figure itself is drawn on plotly's light template in both browser themes, so it
   gets an explicit light surface rather than sitting half-lit on a dark page. */
.plot-surface { background: #ffffff; border: 1px solid var(--border); border-radius: 8px; margin-top: 12px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--border); vertical-align: top; }
thead th { color: var(--muted); font-weight: 600; text-transform: uppercase; font-size: 11px; letter-spacing: .04em; }
.theme-table td:first-child, .theme-table td:nth-child(3),
.theme-table td:nth-child(4), .theme-table td:nth-child(5) { font-variant-numeric: tabular-nums; }
.keywords { color: var(--muted); }
.kv th { width: 220px; color: var(--muted); font-weight: 600; }
.kv td { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12.5px; }
"""


_STYLE += """
.tablewrap { max-height: 32rem; overflow: auto; border: 1px solid var(--rule);
  border-radius: 6px; }
table.datatable { border-collapse: collapse; width: 100%; font-size: 0.82rem; }
table.datatable thead th { position: sticky; top: 0; background: var(--card);
  border-bottom: 2px solid var(--rule); text-align: left; padding: 0.4rem 0.55rem;
  font-weight: 600; white-space: nowrap; z-index: 1; }
table.datatable td { padding: 0.35rem 0.55rem; border-bottom: 1px solid var(--rule);
  vertical-align: top; }
table.datatable td.num, table.datatable th.num { text-align: right;
  font-variant-numeric: tabular-nums; white-space: nowrap; }
table.datatable tbody tr:hover { background: var(--hover, rgba(128,128,128,0.10)); }
table.datatable td:nth-child(2) { min-width: 18rem; }
table.datatable td:nth-child(8) { min-width: 20rem; color: var(--muted, inherit); }
label.checkbox-row { display: flex; align-items: center; gap: 0.4rem;
  font-size: 0.85rem; white-space: nowrap; }
select#show-select { padding: 0.3rem 0.5rem; border-radius: 6px;
  border: 1px solid var(--rule); background: var(--card); color: inherit;
  font: inherit; max-width: 28rem; }
"""


_SCRIPT = """
(function () {
  var payload = JSON.parse(document.getElementById("%(payload_id)s").textContent);
  var gd = document.getElementById("%(plot_id)s");
  var slider = document.getElementById("threshold-slider");
  var readout = document.getElementById("threshold-value");
  var caption = document.getElementById("highlight-caption");
  var radios = document.querySelectorAll("input[name=color-by]");
  var hideBox = document.getElementById("hide-below");
  var hiddenCaption = document.getElementById("hidden-caption");

  function selectedMode() {
    for (var i = 0; i < radios.length; i++) {
      if (radios[i].checked) { return radios[i].value; }
    }
    return payload.colorBy;
  }

  function applyPlot() {
    var index = parseInt(slider.value, 10);
    var threshold = payload.thresholds[index];
    var mode = selectedMode();
    readout.textContent = threshold.toFixed(1);
    caption.textContent = payload.captions[index];

    // The page's highlight rule is the panel's is_highlighted: an unscored dataset has a
    // null score and can never clear a threshold, not even 0.0.
    var colors = [];
    for (var i = 0; i < payload.scores.length; i++) {
      var score = payload.scores[i];
      if (score === null) { continue; }
      if (score >= threshold) {
        colors.push(mode === "Theme" ? payload.themeColors[i] : payload.relevanceColors[i]);
      } else {
        colors.push(payload.dimmedColor);
      }
    }
    var hiding = hideBox ? hideBox.checked : false;

    if (payload.traces.scored !== null) {
      var update = {"marker.color": [colors]};
      if (hiding) {
        // A null coordinate draws nothing and answers no hover. An invisible marker that
        // still responded to the pointer would be worse than no hiding at all.
        var xs = [], ys = [], k = 0;
        for (var j = 0; j < payload.scores.length; j++) {
          if (payload.scores[j] === null) { continue; }
          var keep = payload.scores[j] >= threshold;
          xs.push(keep ? payload.scoredX[k] : null);
          ys.push(keep ? payload.scoredY[k] : null);
          k++;
        }
        update.x = [xs];
        update.y = [ys];
      } else {
        update.x = [payload.scoredX];
        update.y = [payload.scoredY];
      }
      Plotly.restyle(gd, update, [payload.traces.scored]);
    }
    if (payload.traces.unscored !== null) {
      // Unscored datasets go with the hidden ones: they can never clear a threshold.
      Plotly.restyle(gd, {visible: [!hiding]}, [payload.traces.unscored]);
    }
    if (payload.traces.colorbar !== null) {
      Plotly.restyle(gd, {visible: [mode !== "Theme"]}, [payload.traces.colorbar]);
    }

    var marks = [];
    if (mode === "Theme" && payload.showThemeNames) {
      marks = hiding ? payload.hiddenAnnotations[index] : payload.annotations;
    }
    Plotly.relayout(gd, {annotations: marks});

    if (hiddenCaption) {
      hiddenCaption.textContent = hiding ? payload.hiddenCaptions[index] : "";
    }
  }

  // --- the dataset table (modelled on theme_map_explorer.html) -------------
  var showSelect = document.getElementById("show-select");
  var tableBody = document.getElementById("dataset-body");
  var tableCaption = document.getElementById("dataset-caption");

  function cell(text, cls) {
    var td = document.createElement("td");
    if (cls) { td.className = cls; }
    td.textContent = text === null || text === undefined ? "" : String(text);
    return td;
  }

  function scoreOf(row) { return row.r === null ? -Infinity : row.r; }

  function visibleRows(threshold) {
    var choice = showSelect ? showSelect.value : "all";
    if (choice === "highlighted") {
      return payload.datasets.filter(function (row) {
        return row.r !== null && row.r >= threshold;
      });
    }
    if (choice.indexOf("theme:") === 0) {
      var wanted = parseInt(choice.slice(6), 10);
      return payload.datasets.filter(function (row) { return row.th === wanted; });
    }
    return payload.datasets.slice();
  }

  function renderTable(threshold) {
    if (!tableBody) { return; }
    var rows = visibleRows(threshold);
    rows.sort(function (a, b) { return scoreOf(b) - scoreOf(a); });

    var frag = document.createDocumentFragment();
    for (var i = 0; i < rows.length; i++) {
      var row = rows[i];
      var tr = document.createElement("tr");

      var tdA = document.createElement("td");
      var link = document.createElement("a");
      link.href = row.u;
      link.target = "_blank";
      link.rel = "noopener";
      link.textContent = row.a;
      tdA.appendChild(link);
      tr.appendChild(tdA);

      tr.appendChild(cell(row.t));
      tr.appendChild(cell(row.sp));
      tr.appendChild(cell(row.n, "num"));
      tr.appendChild(cell(payload.themeNames[String(row.th)] || "Unthemed"));
      tr.appendChild(cell(row.r === null ? "\u2014" : row.r.toFixed(2), "num"));
      tr.appendChild(cell(row.qs.toFixed(4) + " (#" + row.qr + ")", "num"));
      tr.appendChild(cell(row.j));
      frag.appendChild(tr);
    }
    tableBody.replaceChildren(frag);

    if (tableCaption) {
      tableCaption.textContent =
        "Showing " + rows.length.toLocaleString() + " of "
        + payload.datasets.length.toLocaleString()
        + " datasets, highest relevance first. A dash means the run never scored it.";
    }
  }

  function apply() {
    applyPlot();
    renderTable(payload.thresholds[parseInt(slider.value, 10)]);
  }

  slider.addEventListener("input", apply);
  for (var i = 0; i < radios.length; i++) { radios[i].addEventListener("change", apply); }
  if (showSelect) { showSelect.addEventListener("change", apply); }
  if (hideBox) { hideBox.addEventListener("change", apply); }
  apply();
})();
"""


# ---------------------------------------------------------------------------
# The document
# ---------------------------------------------------------------------------


def _hide_below_in_figure(
    figure: go.Figure,
    points: Sequence[DatasetPoint],
    roles: TraceRoles,
    threshold: float,
) -> None:
    """Null out the hidden points so the file's first paint already matches its controls.

    The browser script does the same thing on every control change. Doing it here as well
    only avoids one frame of the wrong picture between load and the script's first run.
    """
    if roles.scored is not None:
        scored = [p for p in points if p.relevance_score is not None]
        keep = [p.relevance_score is not None and p.relevance_score >= threshold for p in scored]
        trace = figure.data[roles.scored]
        trace.x = [p.x if k else None for p, k in zip(scored, keep)]  # type: ignore[union-attr]
        trace.y = [p.y if k else None for p, k in zip(scored, keep)]  # type: ignore[union-attr]
    if roles.unscored is not None:
        figure.data[roles.unscored].visible = False  # type: ignore[union-attr]


def build_standalone_html(
    result: ThemeMapResult,
    titles: Mapping[str, str],
    run_id: str,
    *,
    threshold: float = DEFAULT_HIGHLIGHT_THRESHOLD,
    color_by: str = "Relevance",
    show_theme_names: bool = True,
    hide_below: bool = False,
    details: Mapping[str, Mapping[str, Any]] | None = None,
) -> str:
    """One complete, offline HTML document holding the theme map of ``result``.

    plotly.js is inlined, so the file opens with no network at all. It is roughly 4.7 MB;
    that is the price of a figure that still works in six years, and it is accepted.

    The page carries working controls. The figure is built once at a threshold below every
    score, so its trace layout is fixed -- a colourbar placeholder, the unscored points,
    and one trace holding every scored point -- and the page recolours that last trace
    with ``Plotly.restyle`` as the controls move. :func:`prepare_scored_trace` puts that
    trace onto explicit rgba colours first, so the browser never feeds a string to a
    numeric colour scale. The colour arrays, the captions and the theme annotations are
    all computed here by the panel's own functions and embedded; the browser only indexes
    into them, and the file's opening frame is already the state it was asked for.

    ``threshold``, ``color_by`` and ``show_theme_names`` set the state the file *opens*
    in, so a download matches what the user was looking at.
    """
    if color_by not in COLOR_BY_OPTIONS:
        raise ThemeMapExportError(
            f"Unknown colour mode {color_by!r}; expected one of {list(COLOR_BY_OPTIONS)}."
        )

    points = result.points
    if not points:
        raise ThemeMapExportError("The theme map has no datasets, so there is nothing to export.")

    opening_index = threshold_index(threshold)
    safe_titles = escape_titles(titles)
    safe_themes = escape_themes(result.themes)

    # Build below every score so that no scored point lands in the dimmed group and the
    # trace layout does not depend on the threshold the user happens to be on.
    scores = [float(p.relevance_score) for p in points if p.relevance_score is not None]
    base_threshold = min([HIGHLIGHT_THRESHOLD_MIN] + scores)

    figure = build_theme_map_figure(
        points,
        safe_titles,
        safe_themes,
        color_by="Relevance",
        threshold=base_threshold,
        show_theme_names=False,
    )
    roles = locate_traces(figure, points)
    ramp_max = _ramp_max(figure, roles)

    relevance_colors = relevance_point_colors(points, ramp_max)
    theme_colors = theme_point_colors(points)

    # The file opens already showing the requested state, computed here rather than left
    # to the browser's first frame.
    if roles.scored is not None:
        prepare_scored_trace(
            figure,
            roles.scored,
            highlight_colors(points, threshold, color_by, relevance_colors, theme_colors),
        )
    if color_by == "Theme" and roles.colorbar is not None:
        figure.data[roles.colorbar].visible = False  # type: ignore[union-attr]
    if color_by == "Theme" and show_theme_names:
        visible = (
            [p for p in points if p.relevance_score is not None and p.relevance_score >= threshold]
            if hide_below
            else list(points)
        )
        figure.update_layout(annotations=theme_label_annotations(visible, safe_themes))
    if hide_below:
        _hide_below_in_figure(figure, points, roles, threshold)

    steps = threshold_steps()
    hidden_captions = [hidden_caption(points, step) for step in steps]
    payload = {
        "plotId": PLOT_DIV_ID,
        "colorBy": color_by,
        "showThemeNames": bool(show_theme_names),
        "thresholdIndex": opening_index,
        "thresholds": steps,
        "captions": [highlight_caption(points, step) for step in steps],
        "scores": point_scores(points),
        "relevanceColors": relevance_colors,
        "themeColors": theme_colors,
        "dimmedColor": DIMMED_RGBA,
        "annotations": (
            theme_label_annotations(points, safe_themes) if show_theme_names else []
        ),
        "traces": roles.as_dict(),
        "datasets": dataset_rows(points, titles, details),
        "themeNames": theme_name_map(result.themes),
        "hideBelow": bool(hide_below),
        # The scored trace's own coordinates, so the page can null a point out to hide it
        # and put it back. Plotly draws nothing for a null and gives it no hover, which is
        # what "hidden" has to mean -- an invisible marker that still answers the pointer
        # would be worse than no hiding at all.
        "scoredX": [float(p.x) for p in points if p.relevance_score is not None],
        "scoredY": [float(p.y) for p in points if p.relevance_score is not None],
        # One annotation set per threshold, built from the points still visible at that
        # threshold. A theme whose every member is hidden loses its label rather than
        # floating over an empty patch of map.
        "hiddenCaptions": hidden_captions,
        "hiddenAnnotations": [
            theme_label_annotations(
                [p for p in points if p.relevance_score is not None
                 and p.relevance_score >= step],
                safe_themes,
            )
            for step in steps
        ],
    }

    plot_html = figure.to_html(
        include_plotlyjs=True,
        full_html=False,
        div_id=PLOT_DIV_ID,
        config={
            "displaylogo": False,
            "responsive": True,
            "scrollZoom": True,
            "displayModeBar": True,
        },
    )

    table_html = _theme_table_html(build_theme_table(points, result.themes))
    provenance = result.provenance
    n_points = len(points)
    n_themes = len(result.themes)
    unthemed_percent = 100.0 * sum(1 for p in points if p.theme_id < 0) / n_points

    checked_relevance = "" if color_by == "Theme" else " checked"
    checked_theme = " checked" if color_by == "Theme" else ""
    hide_checked = " checked" if hide_below else ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>UORCA theme map &mdash; {html.escape(str(run_id))}</title>
<style>{_STYLE}</style>
</head>
<body>
<main>
<header class="card">
<h1>Theme map</h1>
<p class="sub">Identification run <code>{html.escape(str(run_id))}</code></p>
<div class="stats">
<div class="stat"><span class="stat-value">{n_points:,}</span><span class="stat-label">datasets</span></div>
<div class="stat"><span class="stat-value">{n_themes:,}</span><span class="stat-label">themes</span></div>
<div class="stat"><span class="stat-value">{unthemed_percent:.1f}%</span><span class="stat-label">unthemed</span></div>
</div>
<p class="note">Every valid dataset of the run is on the map, scored or not. The threshold
below controls colour only &mdash; nothing is ever hidden. An unscored dataset is a small
hollow grey marker at every threshold: it was never scored, so it cannot clear one.</p>
</header>

<section class="card">
<h2>Map</h2>
<div class="controls">
<fieldset>
<legend>Colour by</legend>
<div class="radio-row">
<label><input type="radio" name="color-by" value="Relevance"{checked_relevance}> Relevance</label>
<label><input type="radio" name="color-by" value="Theme"{checked_theme}> Theme</label>
</div>
</fieldset>
<div>
<span class="control-label">Highlight at or above</span>
<div class="slider-row">
<input type="range" id="threshold-slider" min="0" max="{len(steps) - 1}" step="1" value="{opening_index}">
<span id="threshold-value">{threshold:.1f}</span>
</div>
</div>
<div>
<label class="checkbox-row"><input type="checkbox" id="hide-below"{hide_checked}>
Hide datasets below the threshold</label>
</div>
</div>
<p class="caption" id="hidden-caption"></p>
<p class="caption" id="highlight-caption"></p>
<div class="plot-surface">{plot_html}</div>
<p class="note">Theme names are drawn on the theme view only, each on its theme's medoid
dataset. Drag to pan, scroll to zoom, double-click to reset.</p>
</section>

<section class="card">
<h2>Themes</h2>
{table_html}
<p class="note">The <code>Unthemed</code> row is not a theme. Its maximum relevance is the
honest warning: in the benchmark run the single highest-scoring dataset of all was
unthemed.</p>
</section>

<section class="card">
<h2>Datasets</h2>
<div class="controls">{_show_selector_html(points, result.themes)}</div>
<p class="caption" id="dataset-caption"></p>
<div class="tablewrap">
<table class="datatable">
<thead><tr>
<th>Accession</th><th>Title</th><th>Species</th><th class="num">Samples</th>
<th>Theme</th><th class="num">Relevance</th><th class="num">Query cos. (rank)</th>
<th>Justification</th>
</tr></thead>
<tbody id="dataset-body"></tbody>
</table>
</div>
<p class="note">Query cosine is the similarity between the dataset and the run&rsquo;s
question. It is a weak signal kept because it is free: on the benchmark ground truth the
LLM relevance score gave a median rank of 10 of 1,199 while cosine gave 70. Rank by the
relevance score, not by this column. Distance to the nearest theme is deliberately absent
&mdash; it is not persisted, and a substitute under that name would be a fabrication.</p>
</section>

<section class="card">
<h2>Provenance</h2>
<p class="note">Coordinates and labels are persisted artifacts. They are
<strong>not regenerable</strong> from the seed alone: t-SNE output depends on the thread
count at large n, which is why <code>OMP_NUM_THREADS</code> is recorded here.</p>
{_provenance_html(provenance)}
</section>
</main>
<script id="{PAYLOAD_SCRIPT_ID}" type="application/json">{_json_for_script(payload)}</script>
<script>{_SCRIPT % {"payload_id": PAYLOAD_SCRIPT_ID, "plot_id": PLOT_DIV_ID}}</script>
</body>
</html>
"""
