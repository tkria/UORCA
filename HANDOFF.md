# Session Handoff — 2026-08-12 (session 2: theme-map design + validation)

Branch: `graph` · 110 commits ahead of `origin/master`, nothing pushed · previous session's safety tag
`pre-prune-2026-08-12` still in place.

**No production code was written this session.** It was a design discussion followed by an
independent validation that stopped the design before implementation. The design is **not approved
and must not be implemented as written** — its numeric core was empirically shown to fail.

**Next session's agreed first action: run the OpenAI embedding experiment (§4.1).** It costs ~1 cent
and it decides the shape of the whole feature.

> **UPDATE 2026-08-13 — §4.1 is done ($0.0201, three corpora).** The feature is viable and a
> validated pipeline now exists; see §4.1 and
> `docs/theme_map_validation/real_embeddings/FINDINGS.md`. Decision 8 can be rewritten, but **two
> user decisions are outstanding before implementation**: whether ~33% grey/unthemed datasets is
> acceptable, and whether to name all 40 themes on a large corpus or only the top-N. Still no spec,
> still no production code.

---

## 1. What was accomplished

### A feature was designed: a semantic "theme map" for dataset identification
Goal, in the user's words: *"a semantic embedding of all datasets that are assessed for relevance,
then clustered (visualised, e.g. PCA plot-esque)"* — so users can see the **themes** present among
the datasets a run surfaced. Eight design decisions were settled (§2). Seven survive validation; the
eighth — the algorithm — does not.

### The identification funnel was mapped with real numbers
From `uorca/identification/dataset_identification.py:main()` plus the four run directories on disk:

| Stage | Count (observed) |
|---|---|
| GEO esearch, unique GSEs | ~3,050–3,616 |
| Valid after SRA validation | **793 / 807 / 818 / 2,065** |
| Stage 1 (LLM biology score, Title+Summary, 1 round) | **all valid datasets** |
| Graduate to Stage 2 (`Stage1BiologyScore >= 3`) | ~206–300 |
| Stage 2 (enriched, biology+design, 3 rounds) | same ~206–300 |

Incidental findings worth keeping:
- `hdbscan>=0.8.40` is a **declared dependency imported nowhere** — a fossil.
- `dataset_identification.py:1360` drops a column named `embedding` that nothing ever creates —
  evidence this feature was attempted before and abandoned.
- `umap-learn` is **not** installed. `numpy` 2.3.1, `pandas` 2.3.1, `sklearn` 1.7.0, `plotly` 6.2.0 are.

### Two independent agents validated the method and found it broken
Both were given identical self-contained briefs, were blind to each other, made **no API calls**, and
rebuilt the numeric pipeline on the real GEO `Title + Summary` text using a deterministic TF-IDF+SVD
surrogate for the embeddings. Findings in §3.

---

## 2. Decisions made, and why

Settled by structured Q&A with the user. **Items 1–7 stand. Item 8 is invalidated.**

1. **Population: all valid datasets (~800–2,000).** The literal reading of "assessed for relevance" —
   Stage 1 scores every valid dataset. Large enough for real structure, and every point carries a
   `Stage1BiologyScore` to colour by. *Rejected:* all ~3,600 retrieved (half the map would be
   datasets that can never be analysed); Stage-2 graduates only (~250 — can't show what you missed).
2. **Read-only overview with named clusters.** User was explicit: *"Later the functionality may be
   expanded but this is beyond the scope of this task."* Design must not foreclose a later
   lasso-select → Run wiring, but must not build it. *Rejected:* lasso→Run now; two-way cross-filter
   with the results table; bare unlabelled plot (leaves the user to infer every theme by hovering —
   i.e. does none of the work the feature exists to do).
3. **Embeddings from OpenAI `text-embedding-3-small`.** ~$0.011 per run at n=2,065 (measured), and
   `OPENAI_API_KEY` is already mandatory. **Must fail loudly when the provider is Bedrock** — no
   silent fallback. *Rejected:* local biomedical sentence-transformer (+~2 GB torch into a project
   with neither); TF-IDF+SVD as the primary (lexical, not semantic — it cannot merge "hiPSC" with
   "induced pluripotent stem cell", which is the entire point).
4. **Runs at the end of the identification run, cached to disk.** Provenance lands beside the CSVs and
   no API work happens inside a Streamlit rerun. Call site is the **last** block of `main()`'s `try`,
   after `Dataset_identification_result.csv`, `selected_datasets.csv` and
   `identification_metadata.json` are all written — the map must never be positioned where it can
   cost a completed 60-minute run. *Consequence accepted by the user:* the four existing runs get no
   map. *Rejected:* on-demand GUI compute (2,000 API calls from a Streamlit background thread — the
   `BrokenPipeError` hazard in CLAUDE.md Issue 3); a separate `uorca cluster` CLI subcommand
   (friction against the GUI-first direction).
5. **Layout: a dedicated "🗺️ Theme Map" tab** in `uorca/gui/pages/identify.py`, full-width plot with
   the theme table beneath. Leaves the existing Search GEO tab untouched. *Rejected:* split panel
   above the results table (lengthens an already-busy page); theme-table-first with the plot as a
   footnote (demotes the visualisation the user actually asked for).
6. **Encoding: colour = theme by default, with a radio toggle to colour = relevance.** Themes own
   colour because theme discovery is the stated purpose; the relevance view is where the diagnostic
   value lives (an unexpectedly hot cluster is the finding worth having). Uniform dot size.
   *Rejected:* dot size = sample count — size reads as importance whether intended or not, and
   `CONSTITUTION-analysis.md` §3 already reserves size for set magnitude.
7. **Naming: one structured LLM call per run** returning a short label + one-line description per
   cluster, with c-TF-IDF keywords computed and stored **alongside** as an audit trail so a user can
   see what evidence produced the name. *Rejected:* keywords only (on GEO text they read
   "cells, human, expression" and fail to name the theme); LLM label with no keywords (an
   unauditable assertion about a cluster the user can't check).
8. ~~**Algorithm: L2-normalise → PCA(50) → HDBSCAN → seeded t-SNE for display.**~~
   **INVALIDATED — see §3. This must be redesigned before any implementation.**

### Rejections that validation has since reopened
- **UMAP for display, rejected to avoid a dependency.** One agent argues the rejection is now weak:
  umap-learn is pure Python, and its own documentation turns ARI 0.054 into 0.924 on precisely the
  pipeline that was proposed. Counter-consideration it also raised: umap-learn pulls numba/llvmlite
  and does not declare Python 3.13 support. `openTSNE` (numpy/scipy/sklearn only, no numba,
  peer-reviewed, and supports embedding new points into an existing map) was raised as a third option.
- **Clustering on the 2-D projection, rejected as "methodologically weakest".** **Both agents now
  recommend it as the primary fix.** The reasoning that rejected it was not wrong in the abstract —
  t-SNE distances are not globally meaningful — but it was outweighed by measurement: clustering in
  50-d and displaying in 2-D produces a map whose colours and positions encode nearly disjoint
  partitions (ARI 0.025–0.034), which a user reads as a broken feature.
- **KMeans, rejected for forcing spherical clusters.** Back on the table as a fallback, because it
  cannot return zero clusters — which the chosen method does, on most real corpora.

---

## 3. Validation findings — read this before touching the design

Verdicts: **NOT VIABLE AS SPECIFIED** and **VIABLE WITH CHANGES**. Same evidence, different framing;
both conclude three steps need redesign and the rest is sound.

### Independently replicated (two agents, same numbers, no shared context)

| Measurement | Agent A | Agent B |
|---|---|---|
| PCA-50 → HDBSCAN on the three npc corpora | **0 clusters, 100% noise** | **0 clusters, 100% noise** |
| PCA-50 → HDBSCAN on hcm (n=2,065) | 2 clusters, **68.1%** noise | 2 clusters, **68.1%** noise |
| Variance retained at 50 components | 12.5–19.9% | 16.8–19.9% |
| Sub-series duplicate titles (bracket-stripped) | 9.3–10.5% of rows | 9.3–10.5% of rows |
| Cluster labels vs. structure visible in 2-D | 15-NN purity 0.72 vs 0.95 | ARI **0.025–0.034** |
| Embedding cost per run | — | **$0.0105** at n=2,065 |

**The design's "20–40% noise" budget was never observed in either direction**: 68–100% on real
surrogate data, 0–10% on synthetic data with planted structure.

Both agents independently found that `PCA(n_components=50)` + HDBSCAN is the **worked
counter-example in umap-learn's own clustering documentation** (ARI 0.054, ~17% of points clustered).
The design proposed, in good faith, the exact configuration a major library publishes to demonstrate
how not to do this.

### Agreed defects

| # | Defect | Severity |
|---|---|---|
| 1 | PCA-50 → HDBSCAN is degenerate on this data | CRITICAL |
| 2 | No branch for `k=0` / `k=1` — and it is the *likely* case, not an edge case. As specified the feature renders an all-grey unlabelled scatter, a silent failure of the kind CLAUDE.md forbids | CRITICAL |
| 3 | Clusters computed in 50-d, coordinates in 2-D → near-disjoint partitions; 6–14 of 20 clusters visibly split into 2–4 islands | CRITICAL / MAJOR |
| 4 | `min_cluster_size = max(10, round(n/60))` makes runs mutually incomparable — a 15-dataset theme is named at n=793 and grey noise at n=2,065 | MAJOR |
| 5 | `perplexity = min(30, max(5, n/100))` inverts published guidance; the `min(30,…)` cap means it can only ever go *below* sklearn's default. Costs 0.53–0.60 15-NN overlap against perplexity 30 | MAJOR |
| 6 | What the design calls c-TF-IDF is plain TF-IDF. Under the naive scheme 87.1% of terms pin at maximum idf, so ranking collapses to raw frequency | MAJOR |
| 7 | 8 centroid-nearest representatives: biased (top-8 internal cosine exceeds whole-cluster cosine by 25–60%) and duplicate-contaminated (1.0–1.45 of 8 slots wasted, up to 7/8). One observed cluster's top-5 was the same study three times | MAJOR |
| 8 | ~22% of one corpus clustered on GEO **boilerplate**, not biology — endometriosis grouped with bamboo terpene biosynthesis; yeast with honey bees | MAJOR (surrogate-exaggerated; upper bound) |
| 9 | Excluding `Species` does **not** prevent organism-driven clustering — PC1 vs is-mouse r ≈ +0.31. Organism names live in the free text | MINOR–MAJOR |
| 10 | `PCA(n_components=50)` on 1536-d input **auto-selects the randomized solver**, so "PCA is deterministic" was wrong on its own terms. `hdbscan` also defaults to `approx_min_span_tree=True` | MAJOR |

### Unresolved contradictions between the two agents — do not assume either

| Question | Agent A | Agent B |
|---|---|---|
| **t-SNE across `OMP_NUM_THREADS`** | 8→1 **changed the layout**; k 21→35, noise 12.6%→29.4% | **Bit-identical**, max diff 0.0, 15-NN overlap 1.000 |
| Clusters after clustering in 2-D | **21–35** clusters, 12–29% noise | **3–4** clusters, 1.5–11.6% noise |
| `min_samples` | 5 gave 100% noise → use 3 | keep 5 |
| `min_cluster_size` | fixed 15 | fixed 10 |
| Embedding width | keep 1536, do not shorten | request `dimensions=512` natively from the API |
| `hdbscan` dependency | drop it — `sklearn.cluster.HDBSCAN` exists in 1.7 | keep it — needs `exemplars_` / `probabilities_` |

The first row is a direct factual conflict about a reproducibility claim. The second is a ~10×
disagreement about how many themes the *fixed* method produces — plausibly explained by different
`min_cluster_size` and perplexity, but that is a guess, not a finding.

### Consolidated parameter recommendations (where both agree)

| Parameter | Proposed | Recommended |
|---|---|---|
| Clustering space | PCA-50 | **the 2-D display space**, or 5–10 dims |
| `min_cluster_size` | `max(10, n/60)` | **fixed** (10 or 15 — unresolved) |
| t-SNE perplexity | `min(30, max(5, n/100))` | **fixed 30** |
| PCA solver | default | **`covariance_eigh`** or `full` (exact, not randomized) |
| `approx_min_span_tree` | default `True` | **`False`** |
| Noise expectation | 20–40% | **no fixed band**; hard-assert `k >= 3` and fail loudly otherwise |
| Keywords | sklearn TF-IDF over concatenated docs | **real c-TF-IDF** `tf · log(1 + A/f)` on L1-normalised class counts, **with sqrt frequent-word damping** (cut boilerplate from 2.4/10 to 0.6/10) |
| Naming reps | 8 centroid-nearest titles | **10–15, deduplicated on bracket-stripped title**, via `exemplars_`/`probabilities_`, prompt including c-TF-IDF terms + cluster size |
| Persisted artifacts | embeddings `.npy` + SEED | **+ 2-D coordinates + cluster labels + sklearn/numpy/hdbscan versions** — persist the output, don't promise regeneration |
| Text assembly | `Title + ". " + Summary` | same, **plus** strip `Purpose:/Methods:/Results:/Conclusions:` scaffolding and dedupe sub-series |

### One design claim that survived, and one that did not
- **Survived:** L2-normalising so Euclidean ≡ cosine **does hold through PCA's mean-centring** —
  PCA is a translation plus a rotation with `whiten=False`. Verified to ~5e-15 by both agents. A
  mid-session suspicion that centring broke this was **wrong**; the real loss is one step later, at
  truncation to 50 components.
- **Did not:** the reproducibility claim in general (see defect 10 and the thread contradiction).

---

## 4. Open tasks, priority order

### 4.1 ~~Run the OpenAI embedding experiment~~ ✅ **DONE 2026-08-13 — see `docs/theme_map_validation/real_embeddings/FINDINGS.md`**

Ran on three corpora (hcm_2065, npc_fixed_793, and npc_807 as a held-out generalisation test) for
**$0.0201** total. Embeddings persisted as `.npy`, so all further numeric questions are free.

**Verdict: the feature is viable; the proposed algorithm is confirmed dead; and the fix the
validation agents recommended is not quite right either.**

| Question | Answer |
|---|---|
| Q1 proposed config | **3 clusters / 86.2% noise** (hcm), **2 / 66.1%** (npc_fixed). Defect 1 **confirmed on real data** |
| Q2 variance at PCA-50 | **55.2–60.0%** — not the surrogate's 12.5–19.9%. PCA-50 is defensible; the truncation was never the problem |
| Q3 boilerplate | **Largely dissolved.** 2 of 40 clusters method-driven (~2% of corpus), vs the surrogate's ~22%. Defect 8 mostly dies |
| Q4 Species | AMI **0.074–0.106**. Excluding Species works; organism is not the dominant axis. Defect 9 downgraded |
| Q5 catch-all on focused query | **Does not persist** — largest cluster 7.8%. The 719/793 blob was a TF-IDF artifact |

**Recommended pipeline** (one fixed config, validated on all three corpora):
`Title + ". " + Summary` → `text-embedding-3-small` → `PCA(50, svd_solver="covariance_eigh")` →
`TSNE(perplexity=30, init="pca", random_state=42)` →
`HDBSCAN(min_cluster_size=15, min_samples=3, cluster_selection_method="leaf", approx_min_span_tree=False)`
**in the 2-D display space**. Yields k=20/20/40, noise 30.9/34.9/35.0%, largest cluster 7.8/7.7/4.4%,
silhouette 0.38–0.44.

**Three findings that change the design beyond what validation said:**
1. **`cluster_selection_method` is the decision that matters, and neither agent tested it.** Across an
   84-point grid on three corpora, `leaf` produced a catch-all cluster in **0%** of configurations;
   `eom` in **39–61%**. `eom` is also unstable — on npc_807, `ms` 3→4 at fixed `mcs=12` swings the
   largest cluster from 43.9% to 12.5%. The agents' `mcs` 10-vs-15 and `ms` 3-vs-5 debate was a
   sideshow.
2. **A config fitted on the two corpora §4.1 named broke on a third.** `eom/15/3` passed on
   hcm_2065 + npc_fixed_793, then collapsed to a 43.9% catch-all on held-out npc_807. The extra
   $0.004 corpus is what caught it. Do not fix parameters on two corpora.
3. **The thread-determinism contradiction is resolved: both agents were right, on different corpus
   sizes.** n=793 and n=807 are bit-identical across `OMP_NUM_THREADS` 8 vs 1; n=2,065 is not
   (ARI 0.275, 15-NN 0.526, k 40→38). Same thread count always reproduces exactly. **Coordinates and
   labels must be persisted as artifacts** — regeneration from a seed cannot be promised.

**Open, and needing the user's call before implementation:** noise is 31–35%, so ~a third of datasets
render grey and unthemed. That is a UX decision, not a numeric one. Also undecided: whether to name
all 40 themes on the large corpus or only the top-N by size.

<details><summary>Original task description (superseded)</summary>

Everything downstream hinges on it, and both agents independently named it the single decisive test.
One agent's words: *"Do it before writing any implementation."*

- **Corpus:** `dataset_identification_results/Dataset_identification_result.csv`, rows with
  `Valid == "Yes"` (n = 2,065). Also worth doing `sc_hipsc_npc_fixed/` (n = 793) — it is the
  *focused-query* case, and the surrogate produced a 719/793 catch-all there, which is the biggest
  open threat to the feature's usefulness.
- **Text:** `Title + ". " + Summary`, Species excluded. ~527K tokens.
- **Cost:** ~**$0.0105** for the large corpus, ~$0.0038 for the small one. Requires user authorisation
  to spend, however trivial.
- **Save** the raw embeddings to `.npy` immediately — that artifact makes every subsequent numeric
  question free to re-ask.
- **What it decides:**
  1. Actual cluster count and noise fraction at the proposed parameters → settles defects 1, 2, 4.
     Real bounds are currently 68–100% (surrogate) vs 0–10% (planted structure); reality is between.
  2. `PCA.explained_variance_ratio_.cumsum()` at 5/10/20/50/100 → settles whether 50 components is
     defensible at all. No published curve exists for this model.
  3. Whether semantic embeddings dissolve the boilerplate clusters (bamboo/yeast/honey-bee) →
     settles defect 8.
  4. AMI between cluster labels and `Species` → settles whether excluding Species does anything.
  5. Whether the 719/793 catch-all persists on a focused query → decides whether the feature needs an
     explicit "this corpus has no separable themes" state.
- **Starting scripts** are preserved in `docs/theme_map_validation/` (see §5 for their limitations).
  `exp3_cluster.py` was named as the one to re-run against real embeddings.

</details>

### 4.2 Resolve the two contradictions — **row 1 resolved 2026-08-13, row 2 superseded**
Row 1 (thread determinism) is settled: it is corpus-size-dependent, both agents were correct on the
corpus each tested. See §4.1 finding 3 and `FINDINGS.md`.

Row 2 (21–35 vs 3–4 clusters) is moot — both agents' numbers came from the TF-IDF surrogate. On real
embeddings the answer is k=20–40 under the recommended config, and the spread across their two reports
is explained by `cluster_selection_method` (`leaf` gives more, smaller clusters; `eom` fewer, with a
catch-all), which neither agent recorded.

### 4.3 Revise the design, then write the spec
Three steps change: clustering space, cluster-count guard, keyword extraction. Seven of the eight
user-facing decisions stand, so this is a redesign of the method, not of the feature. The brainstorming
flow was **paused before the spec was written** — no spec exists yet, deliberately.

### 4.4 Carried forward from the previous session — still open, still unblocking nothing
1. **`main_workflow/gene_annotation/` — 246 MB, untracked, protected by nothing.** Zero imports into
   `uorca/`. Decide: track, archive, or delete. Highest real risk of the remaining items.
2. **Five untracked test files are part of the passing suite** (`test_ai_assistant_wiring.py`,
   `test_identification_metadata.py`, `test_metadata_column_selection.py`, `test_row_scoring_nan.py`,
   `tests/integration/replay_edger.py`). A `git clean` deletes tests the 213-pass baseline depends on.
3. **`test_task_submission`** — decide between JSON round-trip, cache-preferred read, or correcting the
   test's expectation.
4. **Two unmerged branches**, each +17 commits: `taskmgr-cancel-fix`, `worktree-hpc-setup-sync-impl`.
   `graph`'s history reads as though this work landed; it has not.
5. **110 commits unpushed** to `origin/master`.
6. **Repo hygiene** — untracked `results/` (373 MB), `dataset_identification_results/` (18 MB),
   stale `requirements.txt`, dead setuptools stanza in `pyproject.toml` (backend is hatchling).
7. **Undocumented backwards-compat code** — `graph/deps.py:23`, four "Old location" fallbacks in
   `results_integration.py`, a version-gated branch in `helpers/__init__.py:957`.

---

## 5. Gotchas and fragile areas

**⚠️ `docs/` is gitignored (`.gitignore:15`), so the validation scripts in
`docs/theme_map_validation/` are NOT protected by git.** So are `CLAUDE.md` (`:18`) and `AGENTS.md`
(`:19`). A disk failure or a `git clean -xfd` loses all of it. Tar them if they matter.

**⚠️ Never run `git clean -xfd` in this repo.** It would delete `.env`, `CLAUDE.md`, `AGENTS.md`,
`docs/`, `todos/`, `main_workflow/` (246 MB) and the five untracked test files. Use `-xfdn` and read
the output first.

**The preserved validation scripts are not a clean record.** Both agents wrote to the same scratchpad
directory with overlapping filenames (`common.py`, `exp6_tsne.py`, `exp7_terms.py`), so
`scratchpad_root/` may contain files from both, and later writes may have overwritten earlier ones.
Attribution to a specific report is unreliable, and re-running them may not reproduce either report
exactly. Treat them as a starting point, not evidence.

**The four on-disk result CSVs predate the current score columns.** They have no
`Stage1BiologyScore`, and `RelevanceScore` is populated for only ~206–300 of ~2,065 valid rows.
Current code **does** persist both — `dataset_identification.py:1394` writes `Stage1BiologyScore`,
and `row_relevance` (line 1333) gives Stage-1-only rows `RelevanceScore = Stage1BiologyScore`. So the
colour-by-relevance toggle will work on new runs but cannot be tested on the CSVs on disk. One
validation agent reported this as a design gap; that report is **stale on this point**.

**`sklearn.cluster.HDBSCAN` and the `hdbscan` package differ in `min_samples` semantics** — sklearn's
**includes the point itself**, so `min_samples=5` in one is `6` in the other. If the dependency is
swapped, this silently changes clustering behaviour.

**t-SNE cross-version identity must not be assumed.** sklearn changed the `init` default to `"pca"`
and `learning_rate` to `"auto"` in 1.2, renamed `n_iter`→`max_iter` in 1.5, and refactored the
distance kernels in 1.1/1.2.

**Method claims in this domain do not fail loudly.** The proposed pipeline imports cleanly, runs
without error, and returns `labels_` — it just returns all `-1`. Nothing raises. This is the same
failure shape as the previous session's path-arithmetic bug: correct-looking code, silent wrong
answer, discovered only by looking at real output. **Validate numeric methods against real data
before designing around them, not after.**

**Carried forward:** one test fails on a clean tree and that is expected
(`test_task_manager.py::test_task_submission`, `assert '10' == 10`). pyright reports 376 errors —
that is the baseline, not breakage. `ruff` is named in `CONSTITUTION-python.md` but is **not
installed**.

---

## 6. Commands to verify current state

```bash
# Tests — expect: 1 failed, 213 passed, 6 skipped (unchanged; no code touched this session)
uv run pytest -p no:cacheprovider --no-cov -q

# Types — expect: 376 errors, 1 warning
uv run pyright

# Live pipeline imports cleanly
uv run python -c "import uorca.graph.runner, uorca.analysis.agents.analysis, uorca.analysis"

# Numeric libs available for the theme-map work (umap is absent by design)
uv run python -c "import numpy, pandas, sklearn, hdbscan, plotly; print(sklearn.__version__)"

# Run the app (Streamlit, port 8501)
uv run uorca explore

# Reproduce the surrogate validation (no API calls, no cost)
cd docs/theme_map_validation/scratchpad_root && uv run python exp3_cluster.py
```

The 4.1 experiment is **not yet scripted** — it needs an OpenAI call added ahead of `exp3_cluster.py`,
and it spends money, so it should not be run unattended.

---

## 7. Files in scope

**Added (untracked, gitignored — see §5):** `docs/theme_map_validation/scratchpad_root/` (21 files)
and `docs/theme_map_validation/scratchpad_mine/` (7 files) — the two agents' experiment scripts and
outputs.

**Modified (tracked):** `HANDOFF.md` — this file.

**Modified (gitignored, local only):** `CLAUDE.md` — added the identification-funnel numbers, the
PCA→HDBSCAN anti-pattern, and the `hdbscan`/`sklearn.cluster.HDBSCAN` note.

**Deliberately untouched:** all of `uorca/` — no production code was written or changed. `tests/` —
no tests added, because there is nothing implemented to test. No spec was written to
`docs/superpowers/specs/`; the brainstorming flow was paused before that step, on the user's
instruction, and writing one would misrepresent an unvalidated design as settled.

**Transient, now gone:** the brainstorm visual-companion server (stopped). Its mockups persist in
`.superpowers/brainstorm/21018-1786521530/content/` — `layout.html` (the three tab-placement options)
and `encoding.html` (the three colour/size options). `.superpowers/` is gitignored.
