"""Sample structure section: gene selection, PCA, group display and figures."""

from typing import Any

import numpy as np
import pandas as pd
import pytest

from uorca.gui.sample_structure import (
    MAX_LABELLED_SAMPLES,
    caption,
    create_clustering_heatmap,
    create_pca_figure,
    group_display,
    group_styles,
    run_pca,
    sample_group_series,
    top_variable_genes,
)


def _cpm(n_genes=20, n_samples=6, seed=1):
    rng = np.random.default_rng(seed)
    data: dict[str, list] = {"Gene": [f"G{i:02d}" for i in range(n_genes)]}
    for j in range(n_samples):
        data[f"GSM{j + 1}"] = list(rng.normal(5, 1, n_genes))
    return pd.DataFrame(data)


def test_top_variable_genes_picks_highest_variance():
    cpm = _cpm()
    cpm.loc[3, "GSM1":] = [0, 10, 0, 10, 0, 10]
    sel = top_variable_genes(cpm, n=1)
    assert list(sel.matrix.index) == ["G03"]
    assert sel.n_used == 1 and sel.n_available == 20


def test_ties_break_by_gene_name_and_fewer_genes_uses_all():
    cpm = pd.DataFrame({"Gene": ["B", "A", "C"], "S1": [0, 0, 1], "S2": [2, 2, 1]})
    assert list(top_variable_genes(cpm, n=2).matrix.index) == ["A", "B"]
    sel = top_variable_genes(cpm, n=500)
    assert sel.n_used == 3 and sel.n_available == 3
    assert "all 3 genes" in caption(sel, 2, group_display(["x"]), "PCA of")


def test_one_sample_raises():
    with pytest.raises(ValueError, match="at least 2 samples"):
        top_variable_genes(pd.DataFrame({"Gene": ["A"], "S1": [1.0]}))


def test_pca_is_centred_not_scaled_and_deterministic():
    cpm = _cpm()
    # A gene with a large spread must dominate PC1; scaling would hide this.
    cpm.loc[0, "GSM1":] = [0, 0, 0, 20, 20, 20]
    mat = top_variable_genes(cpm).matrix
    a, b = run_pca(mat), run_pca(mat)
    pd.testing.assert_frame_equal(a.coords, b.coords)
    assert a.variance_fraction[0] > 0.8
    assert sum(a.variance_fraction) <= 1 + 1e-9
    pc1 = a.coords["PC1"]
    assert (pc1[["GSM1", "GSM2", "GSM3"]] * pc1[["GSM4", "GSM5", "GSM6"]].mean() < 0).all()


GSE295568_LIKE = [
    "Placenta_CT27_Human trophoblast stem cells_Cells were treated with 1 μM T0070907 starting on day 6 of EVT differentiation and maintained until day 8.",
    "Placenta_CT27_Human trophoblast stem cells_Cells were treated with 1 μM T0070907 starting on day 4 of EVT differentiation and maintained until day 8.",
    "Placenta_CT27_Human trophoblast stem cells_TSCs were treated with vehicle (DMSO) and maintained for 72 hours.",
]


def test_shared_prefix_is_removed_at_a_separator_and_codes_keep_labels_unique():
    d = group_display(GSE295568_LIKE)
    assert d.prefix == "Placenta_CT27_Human trophoblast stem cells_"
    assert d.uses_codes
    labels = list(d.labels.values())
    assert len(set(labels)) == len(labels)
    assert all(len(lab) <= 40 for lab in labels)
    assert all(lab.startswith(("G1 · ", "G2 · ", "G3 · ")) for lab in labels)


def test_prefix_cut_only_at_separator_and_short_names_unchanged():
    d = group_display(["DMSO", "DLX6"])  # share "D" but not up to a separator
    assert d.prefix == "" and not d.uses_codes
    assert d.labels == {"DLX6": "DLX6", "DMSO": "DMSO"}
    # A name equal to the prefix would become empty, so nothing is removed.
    assert group_display(["KO", "KO_rescue"]).prefix == ""


def test_more_than_24_groups_get_distinct_styles():
    styles = group_styles(30)
    assert len({(s["color"], s["symbol"]) for s in styles}) == 30


def test_pca_figure_legend_below_and_axis_variance():
    cpm = _cpm()
    sel = top_variable_genes(cpm)
    groups = pd.Series({"GSM1": "A", "GSM2": "A", "GSM3": "B", "GSM4": "B", "GSM5": "B"})  # GSM6 unknown
    sg = sample_group_series(list(sel.matrix.columns), groups)
    d = group_display(sg.tolist())
    fig = create_pca_figure(run_pca(sel.matrix), groups, d, x_pc="PC1", y_pc="PC3")
    layout: Any = fig.layout
    assert layout.legend.orientation == "h" and layout.legend.y < 0
    assert "% of variance" in layout.xaxis.title.text and layout.yaxis.title.text.startswith("PC3")
    names = {t.name: [c[0] for c in t.customdata] for t in fig.data}
    assert names == {"A": ["GSM1", "GSM2"], "B": ["GSM3", "GSM4", "GSM5"], "(no group)": ["GSM6"]}


@pytest.mark.parametrize("n_samples,labelled", [(6, True), (MAX_LABELLED_SAMPLES + 1, False)])
def test_heatmap_hides_sample_labels_for_many_samples(n_samples, labelled):
    sel = top_variable_genes(_cpm(n_samples=n_samples))
    groups = pd.Series({s: ("A" if i % 2 else "B") for i, s in enumerate(sel.matrix.columns)})
    d = group_display(groups.tolist())
    fig = create_clustering_heatmap(sel, groups, d)
    layout: Any = fig.layout
    assert layout.xaxis2.showticklabels is labelled
    assert layout.yaxis2.showticklabels is False
    bar, main = fig.data[0], fig.data[1]
    # The group bar follows the clustered sample order.
    assert list(bar.x) == list(main.x)
    assert [d.order[int(v)] for v in bar.z[0]] == [groups[s] for s in main.x]


def test_repeated_gene_names_stay_separate_rows():
    cpm = pd.DataFrame({"Gene": ["A", "A", "B"], "S1": [0.0, 5, 1], "S2": [3.0, 0, 2], "S3": [1.0, 2, 9]})
    sel = top_variable_genes(cpm)
    fig = create_clustering_heatmap(sel, None, group_display(["(no group)"]))
    assert len(set(fig.data[1].y)) == 3


def test_identity_groups_are_unknown():
    from uorca.gui.sample_structure import known_groups

    assert known_groups(pd.Series({"S1": "S1", "S2": "S2"})) is None
    assert known_groups(None) is None
    g = pd.Series({"S1": "A", "S2": "S2"})
    assert known_groups(g) is g


def test_no_group_and_empty_names_do_not_block_prefix_or_vanish():
    d = group_display(["(no group)", "Cells_KO", "Cells_WT", ""])
    assert d.prefix == "Cells_"
    assert d.labels["(no group)"] == "(no group)" and d.labels["(empty)"] == "(empty)"
    assert d.labels["Cells_KO"] == "KO"
    # A prefix that would leave only punctuation is not removed.
    assert group_display(["A__", "A_b"]).prefix == ""


def test_caption_counts_and_missing_genes():
    cpm = _cpm(n_genes=5)
    cpm.loc[0, "GSM2"] = np.nan
    sel = top_variable_genes(cpm, n=4)
    assert sel.n_missing == 1 and "G00" not in sel.matrix.index
    assert "all 4 genes" in caption(sel, 6, group_display(["x"]), "PCA of")
    assert "1 genes with a missing value" in caption(sel, 6, group_display(["x"]), "PCA of")
    sel3 = top_variable_genes(cpm, n=3)
    assert "the 3 genes with the highest" in caption(sel3, 6, group_display(["x"]), "PCA of")


def test_pca_rejects_one_gene_and_constant_genes():
    with pytest.raises(ValueError, match="at least 2 samples and 2 genes"):
        run_pca(pd.DataFrame({"S1": [1.0], "S2": [2.0]}, index=pd.Index(["G"])))
    with pytest.raises(ValueError, match="constant"):
        run_pca(pd.DataFrame({"S1": [1.0, 3.0], "S2": [1.0, 3.0]}, index=pd.Index(["G1", "G2"])))


def test_package_reports_failed_pdfs(tmp_path, monkeypatch):
    import zipfile

    from tests._results_fixtures import make_dataset
    from uorca.gui.components import helpers
    from uorca.gui.results_integration import ResultsIntegrator

    make_dataset(tmp_path, "GSE1")
    ri = ResultsIntegrator(str(tmp_path))
    ri.load_data()
    monkeypatch.setattr(helpers, "plotly_fig_to_pdf_bytes", lambda *a, **k: None)
    data = helpers.create_dataset_download_package(
        ri=ri, results_dir=str(tmp_path), dataset_id="GSE1", dataset_accession="GSE1",
        components={"qc_plots": True, "include_pdfs": True, "deg_analysis": False,
                    "metadata": False, "abundance": False, "r_script": False},
    )
    assert data is not None
    with zipfile.ZipFile(__import__("io").BytesIO(data)) as zf:
        warn = [n for n in zf.namelist() if n.endswith(helpers.PACKAGE_WARNINGS_FILE)]
        assert warn, zf.namelist()
        assert "PCA_plot.pdf was not made" in zf.read(warn[0]).decode()
