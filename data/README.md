# Data provenance

UORCA does not commit input data to this repository. All data below is downloaded or generated
on the user's machine. `.gitignore` excludes `data/*`, except this file.

## Reference data

| File or directory | Source | Version | How to get it | Used by |
|---|---|---|---|---|
| `data/kallisto_indices/<species>/index.idx`, `t2g.txt` | [pachterlab/kallisto-transcriptome-indices](https://github.com/pachterlab/kallisto-transcriptome-indices) | Release `v1`, `standard` index type (default) | `./download_kallisto_indices.sh [species] [type]` | kallisto quantification (`uorca/graph/nodes/kallisto.py`), R script bundles |
| `data/mammalian_orthologues.csv` | Not recorded. The owner must confirm the source and the Ensembl/NCBI release. | Not recorded | Not distributed with the repository | `uorca/gui/ortholog_mapper.py` |

Supported species for the kallisto indices: human, mouse, dog, monkey, zebrafish.

The kallisto index release `v1` does not state the Ensembl release in this repository. Record the
Ensembl release from the upstream release notes when you cite results.

## Analysis input data

| Data | Source | Access | Restrictions |
|---|---|---|---|
| Dataset metadata (titles, summaries, sample annotations) | NCBI GEO, through the Entrez E-utilities | Public, `ENTREZ_EMAIL` required | None. NCBI usage policies apply. |
| Raw reads (FASTQ) | NCBI SRA / ENA | Public | None |
| PubMed abstracts (identification stage 2) | NCBI PubMed | Public | None |

UORCA writes the metadata of each dataset to `<accession>/metadata/` and timestamped logs to
`<accession>/logs/` (see "Expected Output" in the main README). The log timestamps give the
retrieval date.

## Example data

| Data | Source | Notes |
|---|---|---|
| `sample_inputs/` | Output of `uorca identify`, run 2025-08-19 | Query: "Comparison of transcriptomic changes induced by Type 1 Diabetes and SARS-CoV-2 infection". 260 datasets assessed, 45 above the threshold of 7.0 (`sample_inputs/identification_metadata.json`). |
| Example results for `uorca explore` | [Zenodo record 17403428](https://zenodo.org/records/17403428) | Precomputed results, so that you can use the explorer without a pipeline run. |

## Sensitivity

All input data is public. UORCA sends dataset text (titles, summaries, metadata, abstracts) to the
configured LLM provider. The AI assistant in `uorca explore` also sends differential expression
statistics to the provider through its MCP tools. UORCA does not send raw reads to the provider.
