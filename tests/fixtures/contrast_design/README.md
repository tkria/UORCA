# Contrast-design fixtures

Small sample tables and contrast lists derived from the UORCA run
`results/20261007_PPARg/` (run on 2026-10-07; that directory is not in the
repository). They are used by `tests/unit/test_design_validation.py`,
`tests/unit/test_contrast_checks.py` and `tests/unit/test_review_flags.py`.
GEO metadata is public.

Derived on 2026-10-27. No values were edited.

## `<GSE>_samples.csv`

Rows: every sample in `<GSE>/metadata/<GSE>_metadata.csv`, in file order.
Columns kept: `GSM`, `title`, the `characteristics_ch1.*` columns listed below
(including the constant ones, so single-level dropping is tested), and
`uorca_group`, which is the group column of that run, joined on `GSM` from
`<GSE>/metadata/edger_analysis_samples.csv`.

| Accession | Characteristics columns kept | `uorca_group` source column |
|---|---|---|
| GSE298581 | `0.cell line`, `2.genotype`, `3.treatment`, `4.time`, `5.replicate`, `6.batch` | `treatment_group` |
| GSE295568 | `0.tissue`, `1.cell line`, `2.cell type`, `3.treatment`, `4.batch` | `experimental_group` |
| GSE166308 | `0.strain`, `1.genotype`, `2.tissue`, `3.cell population` | `genotype` |
| GSE324244 | `0.tissue`, `1.genotype`, `2.treatment` | `genotype` |
| GSE132902 | `0.tissue`, `1.subtype`, `2.treatment` | `treatment_group` |
| GSE161060 | `source_name_ch1`, `0.tissue`, `1.cell type`, `2.strain` | `cell_type` |
| GSE210935 | `0.tissue`, `1.cell type`, `2.donor`, `3.treatment` | `treatment_group` |
| GSE288554 | `1.cell line`, `3.genotype`, `4.treatment` | `treatment_group` |

GSE161060, GSE210935 and GSE288554 were added on 2026-10-28 as regression
cases: two cell types that R's `make.names` merges, level names that are
prefixes of each other (the R run failed), and 6 single-sample groups.

All other columns (protocol text, contact and submission fields, SRX/SRR) were
dropped. In the source files they are constant or are identifiers.

## `<GSE>_contrasts.csv`

The `name` and `expression` columns of `<GSE>/metadata/contrasts.csv`, as the
LLM produced them in that run. `description` and `justification` were dropped.
