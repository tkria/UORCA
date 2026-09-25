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

### Prerequisites

- **Python 3.13+** with `uv` package manager
- **For HPC/SLURM**:
  - SLURM job scheduler
  - Apptainer or Singularity (preferred over Docker for security)
  - Note: Singularity/Apptainer runs containers without root privileges, making it ideal for multi-user HPC environments
- **For local**: Sufficient storage and compute resources
- **API Keys**: OpenAI and NCBI credentials (see Environment variables)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/tkria/UORCA.git
cd UORCA

# 2. Set up the uv environment (installs the locked dependencies)
uv sync

# 3. Download Kallisto indices (required for RNA-seq quantification)
./download_kallisto_indices.sh        # Downloads human, mouse, dog, monkey, zebrafish
# OR download specific species:
./download_kallisto_indices.sh human  # Download only human index

# 4. (Optional) Pull the container for containerised execution
# For Singularity/Apptainer (recommended for HPC/SLURM):
singularity pull uorca_0.1.0.sif docker://kevingchen/uorca:0.1.0
# OR Apptainer (newer name, same command):
apptainer pull uorca_0.1.0.sif docker://kevingchen/uorca:0.1.0

# For Docker (local development only):
docker pull kevingchen/uorca:0.1.0
```

### 🔑 Environment variables

UORCA requires API credentials for accessing biological databases and AI services.

#### Required Variables
- **`ENTREZ_EMAIL`** - Your email address (required by NCBI guidelines)
- **`OPENAI_API_KEY`** - OpenAI API key (required for AI-powered dataset identification)

#### Optional (Recommended)
- **`ENTREZ_API_KEY`** - NCBI API key (enables >3x faster processing - free from [NCBI](https://www.ncbi.nlm.nih.gov/account/settings/))

#### Setup Instructions
1. Copy the template: `cp .env.example .env`
2. Edit `.env` with your actual values:
```bash
ENTREZ_EMAIL=your.email@institution.edu        # Required: Any valid email
OPENAI_API_KEY=<your-openai-api-key>          # Required: Get from https://platform.openai.com/api-keys
ENTREZ_API_KEY=<your-ncbi-api-key>            # Optional: For faster processing
```

#### Troubleshooting
- **Missing variables**: UORCA will show clear error messages with setup instructions
- **Invalid OpenAI key**: Check your key format (should start with `sk-proj-` or `sk-`)
- **Rate limiting**: Add `ENTREZ_API_KEY` for faster NCBI queries (3→10 requests/second)

## 🚀 The UORCA Workflow: Identify → Run → Explore

UORCA follows a streamlined three-step workflow for comprehensive RNA-seq analysis:

### Step 1: Identify - Find relevant datasets
Discover GEO datasets that match your research question using AI-powered search:

```bash
uv run uorca identify -q "cancer stem cell differentiation" -o identification_results

# Options:
# -q: Your research question (required)
# -o: Output directory name
# -m: Max results to fetch for each search term (default: 500)
# -r: Number of ranking iterations (default: 3)
# -t: Relevance threshold 1-10 (default: 7.0)
# --model: GPT model to use (default: gpt-5-mini)
```

**Output**: Directory containing `selected_datasets.csv` with relevant GEO accessions and metadata about your search.

The query can be as detailed or broad as you like.

### Step 2: Run - Process datasets through the pipeline
Execute the complete RNA-seq analysis pipeline on identified datasets:

```bash
# For HPC clusters with SLURM
uv run uorca run slurm --input identification_results/ --output_dir ../UORCA_results

# For local machines
uv run uorca run local --input identification_results/ --output_dir ../UORCA_results --max_workers 4

# Options:
# --input: Directory from identify step OR CSV file
# --output_dir: Where to save results
# --cleanup: Remove intermediate files after processing
# --max_workers: Number of parallel jobs (local only)
```
**Note**: Inputting the directory rather than the CSV file is recommended, as this will ensure the research question is considered in the automated analyses. See `sample_inputs` for an example - feel free to test using this directory!

**Output**: Complete analysis results including differential expression and visualisations.

### Step 3: Explore - Interact with your results
Launch the interactive web interface to explore and analyse results:

```bash
uv run uorca explore ../UORCA_results

# Options:
# --port: Web server port (default: 8501)
# --headless: Run without opening browser

# For remote access via SSH:
# 1. On HPC: uv run uorca explore ../UORCA_results --port 8501
# 2. On laptop: ssh -L 8000:127.0.0.1:8501 username@hpc_server
# 3. Open browser: http://127.0.0.1:8000
```

**Test Data**: To try the interactive functionality without running the full pipeline, download sample results from [Zenodo](https://zenodo.org/records/17403428). After unzipping, run:
```bash
uv run uorca explore path/to/unzipped_folder
```

**Features**: Interactive heatmaps, gene expression plots, and cross-dataset integration.

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

On 2026-09-25 the result was `421 passed, 11 skipped, 1 xfailed`. The expected failure (`xfailed`) is a known
defect in `TaskManager` (see Current Limitations). Five of the skipped tests are theme-map regression tests.
They need real embedding corpora that the repository does not distribute, so they skip in a clean checkout. CI runs the same command on each pull request
(`.github/workflows/tests.yml`).

To test a full pipeline run, process the example input in `sample_inputs/` with `uorca run local`, then open
the results with `uorca explore`. To test only the explorer, use the Zenodo example results.

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
