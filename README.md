# Unified -Omics Reference Corpus of Analyses (UORCA)

A fully containerised, AI-powered workflow for automated RNA-seq analysis of public datasets from the Gene Expression Omnibus (GEO).

## Objective

Automate the identification, reanalysis and AI-assisted interpretation of public GEO RNA-seq datasets.

A researcher gives UORCA a research question. UORCA finds relevant GEO datasets, runs a uniform RNA-seq
analysis on each dataset (kallisto quantification, then edgeR/limma differential expression), and gives an
interactive explorer to compare results across datasets.

## Project Status

- **Status:** active development. Version `0.1.0`.
- **Stage:** collaborative. Other researchers can install and run UORCA. Interfaces and output formats can
  still change between versions.
- **Owner:** Kevin Chen.
- **Publication:** preprint on bioRxiv (see [Citation](#citation)).

## What UORCA Does

UORCA automates the entire RNA-seq analysis workflow:

1. **Dataset Discovery**: AI-powered identification of relevant GEO datasets based on your research question
2. **Data Processing**: Automated download, quality control, and RNA-seq quantification using Kallisto
3. **Statistical Analysis**: Differential expression analysis with automatic experimental design
4. **Interactive Exploration**: Web-based interface for visualising and analysing results across multiple datasets
5. **AI-Powered Insights**: Intelligent analysis assistant for biological interpretation

## Setup

Do the steps below in sequence: prerequisites, installation, then environment variables.

### What each stage needs

| Stage | Needs |
|---|---|
| `uorca identify` (find datasets) | Python 3.13+, `uv`, an OpenAI API key (or Bedrock), `ENTREZ_EMAIL`. No Docker, kallisto or R. |
| `uorca run` (process datasets) | The above, plus kallisto indices and Docker (local) or Apptainer/Singularity with SLURM (HPC). |
| `uorca explore` (the app) | Python 3.13+, `uv`. An OpenAI key enables the AI features and the Identify page. |

### Prerequisites

- **`git`**, to clone the repository. Check with `git --version`. On macOS, `xcode-select --install`
  installs it.
- **Python 3.13+** and the [`uv`](https://docs.astral.sh/uv/) package manager. `uv sync` downloads Python 3.13
  if your system does not have it, so you only need to install `uv`:

  ```bash
  # macOS or Linux
  curl -LsSf https://astral.sh/uv/install.sh | sh

  # Windows (PowerShell)
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

  # Alternatives: brew install uv, or pipx install uv
  ```

  Open a new terminal, then check the install with `uv --version`. See the
  [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) for other options.
- **API keys**: an OpenAI API key and an email address for NCBI (see Environment variables).
- **Only for `uorca run`**:
  - Local: Docker, and enough disk space for the raw reads.
  - HPC: SLURM, and Apptainer or Singularity (runs containers without root, so preferred on shared clusters).

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/tkria/UORCA.git
cd UORCA

# 2. Install the locked dependencies into .venv
uv sync

# 3. Check the install
uv run uorca --help
```

Use `uv sync`, not `uv pip install`: `uv sync` installs the versions pinned in `uv.lock`.

The next steps are only needed for `uorca run`, not for identification:

```bash
# 4. Download Kallisto indices (required for RNA-seq quantification)
./download_kallisto_indices.sh        # Downloads human, mouse, dog, monkey, zebrafish
# OR download specific species:
./download_kallisto_indices.sh human  # Download only human index

# 5. Pull the container
# For Singularity/Apptainer (recommended for HPC/SLURM):
singularity pull uorca_0.1.0.sif docker://kevingchen/uorca:0.1.0
# OR Apptainer (newer name, same command):
apptainer pull uorca_0.1.0.sif docker://kevingchen/uorca:0.1.0

# For Docker (local runs):
docker pull kevingchen/uorca:0.1.0
```

### 🔑 Environment variables

UORCA requires API credentials for accessing biological databases and AI services.

#### Required Variables
- **`ENTREZ_EMAIL`** - Your email address (required by NCBI guidelines)
- **`OPENAI_API_KEY`** - OpenAI API key (required for AI-powered dataset identification)

#### Optional (Recommended)
- **`ENTREZ_API_KEY`** - NCBI API key (raises the NCBI rate limit from 3 to 10 requests/second - free from
  [NCBI](https://www.ncbi.nlm.nih.gov/account/settings/))

#### Setup Instructions
1. Copy the template: `cp .env.example .env`
2. Edit `.env` with your actual values:
```bash
ENTREZ_EMAIL=your.email@institution.edu        # Required: Any valid email
OPENAI_API_KEY=<your-openai-api-key>          # Required: Get from https://platform.openai.com/api-keys
ENTREZ_API_KEY=<your-ncbi-api-key>            # Optional: For faster processing
```

UORCA reads `.env` from the repository root. Never commit `.env` (it is in `.gitignore`).

#### Choosing the AI model

By default UORCA uses OpenAI `gpt-5-mini` (default: `gpt-5-mini`). The model is chosen in this order:

1. Environment variables (in your shell or `.env`): `UORCA_AI_PROVIDER` (`openai` or `bedrock`) and
   `UORCA_OPENAI_MODEL`. `uorca identify --model NAME` sets `UORCA_OPENAI_MODEL` for that run.
2. `~/.uorca/config.yaml`, which the app writes when you click **Save AI Config** on the Project Setup page.
3. The built-in default, OpenAI `gpt-5-mini`.

If identification uses a model you did not expect, check `~/.uorca/config.yaml`: it applies to every run that
does not set the environment variables.

AWS Bedrock: set `UORCA_AI_PROVIDER=bedrock` and, as needed, `UORCA_BEDROCK_MODEL`, `UORCA_BEDROCK_REGION`,
`UORCA_BEDROCK_PROFILE` and `UORCA_BEDROCK_MODEL_PREFIX` (see `.env.example`). Bedrock needs working AWS
credentials. The theme map that ends each identification run uses OpenAI embeddings, so it still needs
`OPENAI_API_KEY`.

#### Troubleshooting
- **Missing variables**: UORCA will show clear error messages with setup instructions
- **Invalid OpenAI key**: Check your key format (should start with `sk-proj-` or `sk-`)
- **Rate limiting**: Add `ENTREZ_API_KEY` for faster NCBI queries (3→10 requests/second)
- **OpenAI HTTP 400 about "function tools with reasoning_effort"**: the configured model does not work with
  UORCA's OpenAI client. Use `gpt-5-mini` (see Choosing the AI model).

## 🚀 The UORCA Workflow: Identify → Run → Explore

UORCA follows a three-step workflow. Each step works from the command line or from the app.

### Step 1: Identify - Find relevant datasets
Discover GEO datasets that match your research question using AI-powered search.

A small first run, to check your installation (about 5 minutes and a small amount of OpenAI usage):

```bash
uv run uorca identify -q "IL-17 stimulation of human keratinocytes" -o identification_test -m 5 -r 1
```

A full run:

```bash
uv run uorca identify -q "cancer stem cell differentiation" -o identification_results
```

Main options (see `uv run uorca identify --help` for all of them):

| Option | Meaning | Default |
|---|---|---|
| `-q`, `--query` | Your research question (required) | |
| `-o`, `--output` | Output directory, relative to where you run the command | `./dataset_identification_results` |
| `-m`, `--max-per-term` | GEO results fetched per search term. The main lever on run time and cost. | 500 |
| `-r`, `--rounds` | Scoring rounds for the Stage 2 candidates | 3 |
| `-t`, `--threshold` | RelevanceScore (0-10) needed to be selected | 7.0 |
| `--library-source` | `bulk`, `sc` (single cell) or `both` | `bulk` |
| `--context` | Extra guidance for the Stage 2 scoring prompt | |
| `--model` | OpenAI model for this run (see Choosing the AI model) | `gpt-5-mini` |

Every valid dataset found is scored once in Stage 1. Datasets with a Stage 1 biology score of 3 or more go
to Stage 2, which adds sample metadata and the PubMed abstract and scores them `-r` times.

**Output**: the output directory contains:
- `Dataset_identification_result.csv`: every dataset found, with validity and scores.
- `selected_datasets.csv`: the datasets at or above the threshold. This is the input to `uorca run`. It can
  be empty if no dataset reaches the threshold.
- `identification_metadata.json`: the query, search terms, times and threshold.
- `theme_map/`: an overview map of the themes in the results. It needs about 50 or more valid datasets; for a
  smaller run the log says "Theme map skipped: corpus too small" and `theme_map/FAILED.txt` records why.
  The identification results are still complete.

The query can be as detailed or broad as you like.

### Step 2: Run - Process datasets through the pipeline
Execute the complete RNA-seq analysis pipeline on identified datasets. This step needs the kallisto indices
and a container runtime (Installation steps 4 and 5).

```bash
# For HPC clusters with SLURM
uv run uorca run slurm --input identification_results/ --output_dir ../UORCA_results

# For local machines
uv run uorca run local --input identification_results/ --output_dir ../UORCA_results --max_workers 4

# Options:
# --input: Directory from identify step OR CSV file
# --output_dir: Where to save results
# --resource_dir: Kallisto indices directory (default: ./data/kallisto_indices/)
# --no-cleanup: Keep FASTQ/SRA intermediate files (they are removed by default)
# --max_workers: Number of parallel jobs (local only)
```
**Note**: Inputting the directory rather than the CSV file is recommended, as this will ensure the research question is considered in the automated analyses. See `sample_inputs` for an example - feel free to test using this directory!

**Output**: Complete analysis results including differential expression and visualisations.

### Step 3: Explore - The UORCA app
Start the app:

```bash
uv run uorca explore                     # opens the app in your browser
uv run uorca explore ../UORCA_results    # opens straight onto the Explore page for these results

# Options:
# --port: Web server port (default: 8501)
# --headless: Do not open a browser (for servers; open the printed URL yourself)

# For remote access via SSH:
# 1. On HPC: uv run uorca explore ../UORCA_results --port 8501 --headless
# 2. On laptop: ssh -L 8000:127.0.0.1:8501 username@hpc_server
# 3. Open browser: http://127.0.0.1:8000
```

The app has these pages:

- **Home** and **Project Setup**: create a project (projects are stored under `~/.uorca/projects/`). Project
  Setup also has the **AI Provider** section, where you can choose and save the model.
- **Identify**: the same identification as `uorca identify`, with a form for the query and the limits
  (for a first test, open the advanced options and set "Max per search term" to 10, its minimum, and "Scoring rounds" to 1). It shows the results table and the theme map.
- **Run**: submit the pipeline for the selected datasets (needs the Step 2 requirements).
- **Explore**: interactive heatmaps, gene expression plots, and cross-dataset integration of pipeline
  results.

The app reads the same `.env` file and model settings as the command line.

**Test Data**: example results for the Explore page are planned on
[Zenodo](https://zenodo.org/records/17403428). The record is not yet open to the public. Until it is, run
the pipeline on `sample_inputs/` (Step 2) to get results to explore.

## Data

All input data is public. UORCA downloads it at run time and does not commit it to this repository:

- Dataset metadata from NCBI GEO, through the Entrez E-utilities.
- Raw reads from NCBI SRA / ENA.
- PubMed abstracts for the second identification stage.
- Kallisto transcriptome indices from
  [pachterlab/kallisto-transcriptome-indices](https://github.com/pachterlab/kallisto-transcriptome-indices),
  release `v1`, through `download_kallisto_indices.sh`.

UORCA sends dataset text to the configured LLM provider. The AI assistant also sends differential expression
statistics to the provider. See [`data/README.md`](data/README.md) for the sources, versions, example data and
sensitivity notes.

## Expected Output

`uorca identify` writes a directory that contains `selected_datasets.csv` and the metadata of the search.
`uorca run` writes one directory for each dataset, as shown below. `uorca explore` reads this results tree.

```
UORCA_results/
├── GSE123456/                           # Individual dataset results
│   ├── metadata/                        # Sample and experimental information
│   │   ├── GSE123456_metadata.csv      # Sample metadata
│   │   ├── analysis_info.json          # Analysis parameters
│   │   ├── contrasts.csv               # Experimental contrasts
│   │   └── edger_analysis_samples.csv  # Samples used in analysis
│   ├── RNAseqAnalysis/                  # Differential expression results
│   │   ├── CPM.csv                     # Counts per million
│   │   ├── DGE_norm.RDS                # Normalised expression object
│   │   ├── MDS.png                     # Multidimensional scaling plot
│   │   ├── filtering_density.png       # Expression filtering diagnostics
│   │   ├── normalization_boxplots.png  # Normalization quality control
│   │   ├── sa_plot.png                 # Sample relationship plot
│   │   ├── voom_mean_variance.png     # Mean-variance trend
│   │   └── Contrast1_vs_Contrast2/    # Per-contrast results
│   │       ├── DEG.csv                # Differentially expressed genes
│   │       ├── volcano_plot.png       # Volcano plot
│   │       ├── ma_plot.png            # MA plot
│   │       └── heatmap_top50.png     # Top 50 DEGs heatmap
│   └── logs/                           # Processing logs
│       ├── analysis_tool_logs_*.json  # Tool execution logs
│       └── *.log                       # Timestamped process logs
├── GSE789012/                          # Another dataset...
├── job_status/                         # SLURM job tracking
│   └── GSE*_status.json               # Individual job status files
└── logs/                               # Batch processing logs
    └── run_GSE*.out/.err              # SLURM output/error logs
```

### Disk Space Requirements

UORCA automatically handles temporary files by writing them directly to your system's disk (not Docker's virtual disk). This means:

- **Local execution**: Requires free disk space on your machine (not Docker's allocation)
- **Storage calculation**: Based on your actual available disk space
- **No Docker configuration needed**: Default Docker Desktop settings (50GB) work fine

For large datasets, ensure you have adequate free space on your system.  UORCA will by default automatically determine how much space can be used - it's designed to be a conservative calculation (and will skip datasets that do not fit), but do still exercise some level of caution.

**Note**: Temporary files are automatically cleaned up after processing. By default, FASTQ and associated intermediate files are removed after processing, though you can set --no-cleanup to retain these files.

## Validation

Run the test suite to validate an installation:

```bash
uv run pytest
```

On 2026-10-09 the result was `443 passed, 11 skipped, 1 xfailed`. The expected failure (`xfailed`) is a known
defect in `TaskManager` (see Current Limitations). Five of the skipped tests are theme-map regression tests.
They need real embedding corpora that the repository does not distribute, so they skip in a clean checkout. CI runs the same command on each pull request
(`.github/workflows/tests.yml`).

To test identification end to end, run the small example in Step 1. To test a full pipeline run, process the
example input in `sample_inputs/` with `uorca run local`, then open the results with `uorca explore`.

The tests check the software behaviour. They do not prove that the automatic dataset selection or the
automatic experimental design is scientifically correct for a given question. Examine the selected datasets
and the contrasts before you use the results.


## Current Limitations

- **LLM steps are not deterministic.** Dataset scoring, metadata interpretation, contrast design and the AI
  assistant use an LLM. Two runs with the same input can give similar but not identical results.
- **An LLM provider is required.** Identification and analysis need an OpenAI API key (or a configured
  Bedrock provider). API calls cost money.
- **Species.** Bundled download support covers kallisto indices for human, mouse, dog, monkey and zebrafish only.
- **Orthologue mapping.** Cross-species mapping in the explorer reads `data/mammalian_orthologues.csv`. The
  repository does not distribute this file.
- **Known defect.** `TaskManager` returns task results as strings (`'10'` instead of `10`), because results
  go through a `TEXT` database column. The test `tests/unit/test_task_manager.py::test_task_submission`
  records this defect.
- **Type checking.** `uv run pyright` reports errors in the code base. Type checking is not part of CI.
- **`requirements.txt` is out of date.** Use `uv sync`, which reads `uv.lock`.

## Feedback Requested

Feedback is most useful on these topics:

1. **Dataset identification.** Does UORCA find the datasets that you expect for your question? Does it miss
   important datasets, or include irrelevant ones?
2. **Experimental design.** Are the automatic sample groups and contrasts correct for the datasets that you
   know?
3. **Explorer.** Which views or comparisons do you need that the explorer does not give?
4. **Installation.** Which step of the setup was difficult on your system (local or HPC)?

Submit feedback via [GitHub Issues](https://github.com/tkria/UORCA/issues).

## Getting Help

- **Command help**: `uv run uorca --help` or `uv run uorca COMMAND --help`
- **Issues**: Submit via [GitHub Issues](https://github.com/tkria/UORCA/issues)

## Citation

If you use UORCA in your research, please cite our preprint:

> **Uncovering biological patterns across studies through automated large-scale reanalyses of public transcriptomic data**
> Chen, K.G. *et al.* (2025)
> bioRxiv: [10.1101/2025.11.04.686647](https://www.biorxiv.org/content/10.1101/2025.11.04.686647v1)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
