"""Browser test: heatmap x-axis labels are complete and do not overlap in the live app.

Runs the real Streamlit app (isolated HOME, fake results with long contrast
names in two datasets) at 1280, 1440 and 1920 px. From the page's SVG it reads
the rotated rectangle of every x tick label (getBBox + getScreenCTM), the plot
area and the chart's SVG, and asserts:

- every label lies fully inside the chart SVG (labels are clipped at its edge),
- every label lies below the plot area,
- no two labels overlap (separating-axis test on the rotated rectangles),
- each dataset accession appears once above its block, not in the labels,
- the numbered column key and its CSV download are shown.
"""

from __future__ import annotations

import json
import time

import pytest

from tests._results_fixtures import make_dataset

pytestmark = pytest.mark.e2e

LONG_NAMES = (
    "EVT_DMOG_vs_control",
    "EVT_SCH772984_vs_control",
    "EVT_T0070907_day0_vs_control",
    "EVT_T0070907_day1_vs_control",
    "EVT_T0070907_day2_vs_control",
    "EVT_T0070907_day3_vs_control",
    "PPARG_KD1_DLX6_overexpression_vs_control",
    "PPARG_KD1_EVT_vs_control",
    "PPARG_KD1_vs_control_TSC_self_renewal",
    "PPARG_rescue_12p5_vs_15_dox",
    "T0070907_0p3_vs_vehicle_TSC",
    "T0070907_all_doses_vs_vehicle_TSC",
)
SECOND_NAMES = ("T0070907_vs_DMSO", "FTX6746_vs_DMSO", "FX909_vs_DMSO")

MEASURE_JS = """
(root) => {
  const plot = root.querySelector('.js-plotly-plot');
  const svg = plot.querySelector('svg.main-svg').getBoundingClientRect();
  const area = plot.querySelector('.nsewdrag').getBoundingClientRect();
  const labels = [...plot.querySelectorAll('.xtick text')].map(t => {
    const b = t.getBBox(); const m = t.getScreenCTM();
    const pts = [[b.x,b.y],[b.x+b.width,b.y],[b.x+b.width,b.y+b.height],[b.x,b.y+b.height]]
      .map(([x,y]) => { const p = new DOMPoint(x,y).matrixTransform(m); return [p.x, p.y]; });
    return {text: t.textContent, pts};
  });
  const notes = [...plot.querySelectorAll('.annotation-text')].map(a => a.textContent);
  return {svg: {l: svg.left, t: svg.top, r: svg.right, b: svg.bottom},
          area: {l: area.left, t: area.top, r: area.right, b: area.bottom}, labels, notes};
}
"""


def _project(pts, axis):
    vals = [x * axis[0] + y * axis[1] for x, y in pts]
    return min(vals), max(vals)


def rectangles_overlap(a, b, tol: float = 0.5) -> bool:
    """Separating-axis test for two convex quadrilaterals given as 4 (x, y) points."""
    for pts in (a, b):
        for i in range(4):
            (x1, y1), (x2, y2) = pts[i], pts[(i + 1) % 4]
            axis = (y1 - y2, x2 - x1)
            norm = (axis[0] ** 2 + axis[1] ** 2) ** 0.5 or 1.0
            amin, amax = _project(a, axis)
            bmin, bmax = _project(b, axis)
            if amax / norm <= bmin / norm + tol or bmax / norm <= amin / norm + tol:
                return False
    return True


def test_overlap_check_itself():
    def rect(cx, w, h, angle_deg):
        import math
        a = math.radians(angle_deg)
        c, s = math.cos(a), math.sin(a)
        return [(cx + x * c - y * s, x * s + y * c) for x, y in
                [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]]
    assert not rectangles_overlap(rect(0, 150, 14, -45), rect(28, 150, 14, -45))
    assert rectangles_overlap(rect(0, 150, 14, -45), rect(10, 150, 14, -45))


def _results(root):
    root.mkdir(parents=True)
    make_dataset(root, "GSE900101", contrasts=LONG_NAMES)
    make_dataset(root, "GSE900102", contrasts=SECOND_NAMES)
    status = root / "job_status"
    status.mkdir()
    for acc in ("GSE900101", "GSE900102"):
        (status / f"{acc}_status.json").write_text(json.dumps({"state": "completed"}))
    return root


def _wait_idle(page) -> None:
    page.locator('[data-testid="stStatusWidget"]').wait_for(state="hidden", timeout=120_000)
    page.wait_for_timeout(500)


def _click_single(page, locator, timeout_s: float = 60) -> None:
    """Click a locator once it matches exactly one element.

    During a rerun Streamlit can keep the previous copy of a widget (for
    example a tab bar) next to the new one for a moment, and a click then
    fails Playwright's strict mode.
    """
    end = time.monotonic() + timeout_s
    while locator.count() != 1 and time.monotonic() < end:
        page.wait_for_timeout(200)
    locator.click()


def _measure(browser, url: str, width: int) -> dict:
    context = browser.new_context(viewport={"width": width, "height": 1400})
    page = context.new_page()
    page.set_default_timeout(60_000)
    try:
        page.goto(url + "/")
        page.get_by_text("Explore Results").first.wait_for(timeout=90_000)
        sidebar = page.locator('[data-testid="stSidebar"]')
        sidebar.get_by_role("button", name="Select All").click()
        _wait_idle(page)
        _click_single(page, page.get_by_role("tab", name="Explore DEG Heatmap"))
        _wait_idle(page)
        _click_single(page, page.get_by_role("button", name="Generate Heatmap"))
        _wait_idle(page)
        root = page.locator(".st-key-lfc_heatmap_scroll").first
        root.locator(".xtick text").first.wait_for(timeout=120_000)
        root.scroll_into_view_if_needed()
        page.wait_for_timeout(1500)
        measured = root.evaluate(MEASURE_JS)
        measured["key"] = page.get_by_text("Column key").count()
        measured["download"] = page.get_by_role("button", name="Download column key (CSV)").count()
        measured["exceptions"] = page.locator('[data-testid="stException"]').count()
        return measured
    finally:
        context.close()


def test_heatmap_labels_fit_and_do_not_overlap(browser, start_app, tmp_path):
    results = _results(tmp_path / "long_names")
    app = start_app({"UORCA_DEFAULT_RESULTS_DIR": str(results)})
    for width in (1280, 1440, 1920):
        m = _measure(browser, app.url, width)
        assert m["exceptions"] == 0
        labels = m["labels"]
        assert len(labels) == len(LONG_NAMES) + len(SECOND_NAMES), [lab["text"] for lab in labels]
        svg, area = m["svg"], m["area"]
        for lab in labels:
            xs = [p[0] for p in lab["pts"]]
            ys = [p[1] for p in lab["pts"]]
            assert min(xs) >= svg["l"] - 0.5 and max(xs) <= svg["r"] + 0.5, (width, lab["text"], xs, svg)
            assert min(ys) >= svg["t"] - 0.5 and max(ys) <= svg["b"] + 0.5, (width, lab["text"], ys, svg)
            assert min(ys) >= area["b"] - 0.5, (width, lab["text"], "overlaps the plot area")
            assert "GSE" not in lab["text"]
            assert len(lab["text"]) <= 25
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                assert not rectangles_overlap(labels[i]["pts"], labels[j]["pts"]), (
                    width, labels[i]["text"], labels[j]["text"])
        assert [n for n in m["notes"] if n.startswith("GSE")] == ["GSE900101", "GSE900102"]
        assert m["key"] >= 1 and m["download"] == 1
