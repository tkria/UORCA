

def test_dataset_blocks_sort_by_accession_number():
    from uorca.gui.results_integration import _accession_sort_key

    assert sorted(["GSE900102", "GSE10", "GSE900101", "GSE2"], key=_accession_sort_key) == [
        "GSE2", "GSE10", "GSE900101", "GSE900102"]
