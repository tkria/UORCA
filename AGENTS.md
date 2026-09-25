# UORCA Repository Navigation Guide for Agents

This document maps the UORCA repository so AI agents can navigate it efficiently.

> Last corrected 2026-08-12. The previous version of this file described a `main_workflow/`
> package (`main_workflow/reporting/`, `streamlit_tabs/`, `agents/`, `run_helpers/`) that **no
> longer exists**. Everything now lives in the `uorca/` package. If you find a reference to
> `main_workflow/` anywhere outside `main_workflow/gene_annotation/`, it is stale.

## Overall notes

The repository uses `uv` for package management and is a proper Python package. Run commands via
`uv run uorca <subcommand>`, or `uorca <subcommand>` if installed. Containerized execution is
supported via Docker and Apptainer.

---

## High-Level Structure

| Path | What it is |
|---|---|
| `uorca/` | **The entire application.** CLI, pipeline engine, GUI, batch backends |
| `tests/` | `unit/` and `integration/` pytest suites (`pytest.ini` sets `testpaths = tests`) |
| `docs/` | Plans, specs, lessons learned. Historical intent, not current state. Gitignored |
| `todos/` | TODO lists. Gitignored |
| `sample_inputs/` | Example input CSVs for testing batch runs |
| `logs/` | Runtime logs and AI tool-call tracking |
| `data/`, `scratch/`, `results/` | Data and outputs — not needed for code editing |
| `main_workflow/gene_annotation/` | **Unrelated, unwired subsystem.** Nothing in `uorca/` imports it |

---

## Package Structure (`uorca/`)

### 1. CLI Interface

- **`uorca/cli.py`** — argparse entry point defining three subcommands:
  - `uorca identify` — GEO dataset identification via AI relevance scoring
  - `uorca run slurm|local` — batch RNA-seq pipeline execution
  - `uorca explore` — launch the Streamlit web application
- **`uorca/identify.py`** — wrapper around dataset identification
- **`uorca/explore.py`** — Streamlit launcher; resolves and runs `uorca/gui/app.py`
- **`uorca/ai_provider.py`** — model/provider selection (OpenAI, Bedrock)

### 2. Pipeline Engine — `uorca/graph/` ★

This is the **live** RNA-seq workflow, built on `pydantic-graph`. Both batch backends invoke it as
`python -m uorca.graph.runner` (`batch/local.py:136`, `batch/slurm.py:449`).

- **`runner.py`** — entry point; accepts `--accession/--output_dir/--resource_dir` or `--config`
- **`graph.py`** — graph assembly and diagram export
- **`state.py`** — shared mutable workflow state
- **`nodes/`** — one file per pipeline stage:
  `geo_extract`, `user_extract`, `metadata`, `contrasts`, `kallisto`, `edger`, `evaluate`, `reflect`
- **`agents.py`, `config.py`, `deps.py`, `models.py`** — agent wiring and typed config

### 3. Bioinformatics Cores — `uorca/analysis/`

Not an independent workflow anymore; the graph nodes import `*_core` functions from here.

- **`agents/extraction.py`** — GEO/SRA extraction, metadata fetch, FASTQ download
- **`agents/metadata.py`** — sample metadata cleaning, grouping column selection, contrast design
- **`agents/analysis.py`** — Kallisto quantification, edgeR prep, edgeR/limma execution
  (`run_kallisto_quantification_core`, `prepare_edgeR_analysis_core`, `run_edger_limma_analysis_core`)
- **`scripts/RNAseq.R`** — the edgeR/limma differential expression workflow
- **`prompts/`** — `analysis.txt`, `extraction.txt`, `metadata.txt`

### 4. Batch Processing — `uorca/batch/`

- **`base.py`** — abstract batch processing interface
- **`slurm.py`** — SLURM submission, resource management, job monitoring
- **`local.py`** — local multiprocessing with resource detection and bind-mounted temp dirs
- **`templates/`** — `run_single_dataset.sbatch.j2`, `run_dataset_array.sbatch.j2`

### 5. Dataset Identification — `uorca/identification/`

- **`dataset_identification.py`** — term extraction, GEO search, batched relevance assessment
- **`scoring.py`** — two-stage scoring: Stage 1 biology-only (recall-tuned) → Stage 2
  biology + design (precision-tuned), with configurable thresholds and library-source hard filtering
- **`prompts/`** — `extract_terms.txt`, `assess_biology_stage1.txt`,
  `assess_relevance_stage2.txt`, `scoring_examples.txt`, `assess_relevance.txt`

### 6. Web Application — `uorca/gui/`

- **`app.py`** ★ — the Streamlit entrypoint (multipage shell)
- **`pages/`** — one module per page: `home`, `project_setup`, `identify`, `run`, `explore`
- **`components/`** — Explorer tab renderers:
  `ai_assistant_tab`, `analysis_plots_tab`, `contrasts_info_tab`, `datasets_info_tab`,
  `expression_plots_tab`, `heatmap_tab`, `uorca_summary_tab`, `sidebar_controls`,
  `custom_datasets`, plus `helpers/` (caching, logging, R-script packaging, zip export)
- **`ai/`** ★ — the live AI layer:
  - `agent_factory.py` — builds pydantic-ai agents wired to the MCP server
  - `config_loader.py` — reads `uorca/config/ai_assistant_config.json`
  - `contrast_relevance.py` — AI scoring/selection of contrasts against the research question
  - `tool_relevance_analyzer.py` — tool usage analysis
  - `gene_schema.py` — Pydantic models for structured gene-analysis output
  - `prompts/` — `ai_agent_analysis.txt`, `assess_and_select_contrasts.txt`
- **`project/`** — `models.py`, `manager.py` (YAML-backed project definitions)
- **`hpc/`** — `ssh_manager.py`, `slurm.py`, `custom_dataset_validator.py`
- **`mcp_server/server_core.py`** — MCP tools exposing UORCA results to the AI assistant
  (`get_most_common_genes`, `get_gene_contrast_stats`, `filter_genes_by_contrast_sets`,
  plus tool-call logging)
- **`results_integration.py`**, **`single_analysis_plots.py`**, **`ortholog_mapper.py`** —
  cross-dataset integration, QC/DE plotting, ortholog mapping

### 7. Shared Utilities

- **`uorca/core/`** — `task_manager.py` (SQLite-backed background tasks), `validation.py`,
  `data_formatters.py`, `gene_selection.py`, `organism_utils.py`, `script_generation.py`
- **`uorca/shared/`** — `entrez_utils.py` (rate-limited Entrez), `workflow_logging.py`
- **`uorca/config/`** — `ai_assistant_config.json`, `dataset_query.json`

---

## Pre-restructure fossils — removed 2026-08-12

The flat pre-package layout left duplicate copies beside their replacements. All were verified
unreferenced and deleted:

| Deleted | Live equivalent |
|---|---|
| `uorca/gui/ai_agent_factory.py` | `uorca/gui/ai/agent_factory.py` |
| `uorca/gui/contrast_relevance.py` | `uorca/gui/ai/contrast_relevance.py` |
| `uorca/gui/tool_relevance_analyzer.py` | `uorca/gui/ai/tool_relevance_analyzer.py` |
| `uorca/gui/config_loader.py` | `uorca/gui/ai/config_loader.py` |
| `uorca/gui/ai_gene_schema.py` | `uorca/gui/ai/gene_schema.py` |
| `uorca/gui/uorca_explorer.py` | `uorca/gui/app.py` |
| `uorca/gui/pages_backup/` | `uorca/gui/pages/` |
| `uorca/analysis/master.py` (+ `prompts/master.txt`) | `uorca/graph/` |

Recoverable via `git show pre-prune-2026-08-12:<path>` if ever needed. A surviving reference to any
of these names is a stale comment — fix the comment, don't restore the file.

---

## Supporting Infrastructure

- **`Dockerfile`** / **`uorca.def`** — Docker and Apptainer container definitions
- **`download_kallisto_indices.sh`** — fetch Kallisto indices per organism
- **`slurm_config_template.yaml`** — template for SLURM settings (`slurm_config.yaml` is gitignored)
- **`pyproject.toml`** — package config, dependencies, `uorca` console script (hatchling backend)
- **`uv.lock`** — dependency lock file. Note `requirements.txt` is a stale July-2025 export
- **`pytest.ini`** — `testpaths = tests`; default coverage is `--cov=uorca/core` only

---

## CLI Usage Summary

```bash
# Dataset identification
uv run uorca identify -q "cancer stem cell differentiation" -o identification_results

# Batch analysis
uv run uorca run slurm --input identification_results/ --output_dir ../UORCA_results
uv run uorca run local --input identification_results/ --output_dir ../UORCA_results --max_workers 4

# Results exploration
uv run uorca explore ../UORCA_results --port 8501

# Single dataset, straight to the pipeline engine
uv run python -m uorca.graph.runner --accession GSE12345 \
  --output_dir ./results --resource_dir ./kallisto_indices
```

---

## Where to work — task to location

| Task | Go to |
|---|---|
| Dataset identification / scoring | `uorca/identification/` (`scoring.py`, `prompts/`) |
| Data extraction from GEO/SRA | `uorca/analysis/agents/extraction.py`, `uorca/graph/nodes/geo_extract.py` |
| Metadata processing, contrast design | `uorca/analysis/agents/metadata.py`, `uorca/graph/nodes/{metadata,contrasts}.py` |
| Quantification / differential expression | `uorca/analysis/agents/analysis.py`, `uorca/graph/nodes/{kallisto,edger}.py`, `uorca/analysis/scripts/RNAseq.R` |
| Adding or reordering a pipeline stage | `uorca/graph/nodes/`, wire into `uorca/graph/graph.py`, extend `uorca/graph/state.py` |
| Batch submission / HPC | `uorca/batch/`, `uorca/gui/hpc/` |
| Explorer UI, plots, tabs | `uorca/gui/components/`, `uorca/gui/pages/explore.py` |
| AI assistant behaviour | `uorca/gui/ai/`, `uorca/gui/mcp_server/server_core.py` |
| Background task handling | `uorca/core/task_manager.py` |
| Container execution | `Dockerfile`, `uorca.def` |

---

## Notes for maintainers

- The Explorer summary tab displays a hardcoded application version (currently `1.0`) at
  `uorca/gui/components/uorca_summary_tab.py:163`. Update it there if you change app versioning.
- `_repo_paths_for_r()` in `uorca/gui/components/helpers/__init__.py` returns `package_root`
  (`uorca/`) and `project_root` (repo root) as distinct keys. Keep them distinct — conflating them
  previously caused `RNAseq.R` and `t2g.txt` to be silently dropped from download bundles
  (fixed 2026-08-12, commit `05442fc`; regression tests in `tests/unit/test_r_script_packaging.py`).
- All unit tests sit flat in `tests/unit/`. The empty
  `tests/unit/{batch,core,gui,identification,pipeline}/` scaffolding directories were removed
  on 2026-08-12.

<!-- Added by the TI research software framework adoption. Review and edit. -->

## Project purpose

> Automate the identification, reanalysis and AI-assisted interpretation of public GEO RNA-seq datasets.

Owner: Kevin Chen. Stage: collaborative. Describe intended users and non-goals here.

## Scientific constraints

- Do not invent assumptions, labels, thresholds, metrics, dataset splits, or expected results.
- Ask for clarification or mark unresolved scientific decisions explicitly.
- Do not silently change filters, exclusions, splits, prompts, models, reference versions, evaluation procedures, or interpretation rules.
- Preserve links between results, code versions, configuration, and data provenance.
- Treat passing tests as necessary evidence, not proof that the scientific design is correct.

## Development workflow

- Branch names: `feature/<task-description>` or `fix/<issue-description>`. Open pull requests against `master`.
- Commit messages: descriptive, imperative mood (for example "Add TaskManager with SQLite persistence").
- Before each commit, run `uv run pytest`. For GUI changes, also test the change in the browser
  (`uv run uorca explore`).
- `uv run pyright` reports existing errors. Do not add new errors; compare against the count before your change.
- For a bug fix, first write a test that reproduces the bug.
- Ask the owner before you change the R analysis script, kallisto parameters, the TaskManager SQLite schema,
  the container definitions, or the SLURM job templates.

## Privacy and security

- Never commit credentials, tokens, private keys, identifying information, or restricted raw data.
- Use `.env.example` for variable names and placeholders only.
- Do not send restricted data to unapproved models, services, tools, or providers.
- Request human approval before expensive, destructive, modifying, or consequential operations.

## Commands

- Install: uv sync
- Framework check: `python scripts/framework_check.py`
- Test: `uv run pytest`
- Run main example: `uv run uorca explore`

## Definition of done

A scoped task is complete only when the requested behaviour is implemented, the relevant tests or validation checks pass, representative outputs have been inspected, documentation and provenance are updated where needed, no secrets or restricted data were introduced, and the complete AI-generated diff has been reviewed by a researcher.
