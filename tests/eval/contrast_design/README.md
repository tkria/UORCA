# Contrast-design evaluation set

One YAML case per dataset (plan `docs/plans/2026-10-27-03-contrast-design-and-default-selection.md`,
section 5). Every case is a **draft — owner to confirm** (`status` field) until the owner
checks the design, the expected contrasts, their types and which are shown by default.

Drafted 2026-10-27 from the stored metadata of the run `results/20261007_PPARg/` (not in the
repository). The committed fixtures in `tests/fixtures/contrast_design/` hold the sample
columns that the cases refer to. That run had no research question ("Not specified"), so each
case states the question it assumes in `research_question_assumed`.

| Case | Difficulty | Point of the case |
|---|---|---|
| GSE298581 | difficult | Needs a (cell line, replicate) `pair` block: `~0 + group + pair`, 33 residual df (owner, 2026-10-28) |
| GSE295568 | difficult | Batch must be dropped (groups nested in batch); rescue arms vs shPPARG (and DLX6 vs EVT shPPARG) are shown by default |
| GSE166308 | simple | Two groups, one contrast |
| GSE324244 | simple | Two groups, one contrast |
| GSE132902 | simple | Two groups, one contrast |

## Schema

- `groups`: short label -> `{metadata column: exact value}`. A group is the set of samples
  that match all its selectors. Scoring maps labels to whatever group levels a run produced by
  sample membership, so it does not depend on the run's level names.
- `design`: `group_columns`, `required_blocks`, `must_drop` (column + `reason_code` from
  `uorca.core.design_validation`), `expected_formula` (as `validate_design` writes it),
  `expected_residual_df`, and `needs_review_expected` (for `uorca.core.review_flags`, under the
  current pipeline's grouping).
- `design.derived_columns`: columns built before validation by joining source columns with
  `"_"` (GSE298581: `pair` = cell line x replicate).
- `cell_line_effects` (GSE298581 only): a second analysis with its own groups, blocks and
  required contrasts, and what `validate_design` produces for it today.
- `owner_answers`: answers the owner gave to earlier `owner_to_confirm` points.
- `contrasts.required`: each item has `target` and `reference` (lists of group labels; more
  than one label means an equal-weight average) and three fields (owner decision Q7b,
  2026-10-28, which replaced the earlier primary / supporting / not_relevant roles):
  - `contrast_type`: objective, one of `perturbation_vs_control`, `rescue_vs_perturbation`,
    `rescue_vs_control`, `dose_vs_control`, `dose_vs_dose`, `time_vs_baseline`,
    `drug_vs_drug`, `interaction`, `average`, `other`
    (`uorca.core.contrast_checks.CONTRAST_TYPES`). The weight shape is checked in code:
    `interaction` must be four groups weighted +1, +1, -1, -1, `average` a scaled average, the
    named comparison types pairwise, and `other` anything valid. Whether a pairwise contrast
    is a rescue, a dose or a drug comparison cannot be read from the design, so that part is
    not checked.
  - `show_by_default`: `true`, `false`, or `pending_owner` while the owner has not decided.
  - `reason`: one sentence tied to the research question.
  - `rule` (optional): `owner 2026-10-28` marks a value the owner set directly (the three
    simple cases, Q7c). `owner-default 2026-10-28` marks the rescue rule: rescue vs
    perturbation is shown by default; rescue vs control (is the rescue complete?) is in the key
    with `show_by_default: false`.
- `contrasts.optional`: acceptable when present; same fields.
- `contrasts.not_shown_by_default`: may be present, but must not be shown by default.
  `any_pair_within` means any pairwise contrast between those groups.
- `contrasts.must_be_absent`: contrasts or rules (`no_cross_block`) that should not appear.
- `current_run`: what the 2026-10-07 run produced, the "before" measurement.
- `owner_to_confirm`: open points for the owner.

## Scoring (proposed, not yet implemented)

Deterministic. Convert each run contrast with `uorca.core.contrast_checks.check_contrasts` and
take `canonical_pair()`; convert each case item to the same pair through the group selectors.
Score: required pairs present / required, agreement of `show_by_default` with the case,
`not_shown_by_default` pairs shown by default, `must_be_absent` hits, formula match, and residual df match. A reversed contrast (reference - target) should
count as present with a direction note. Run each case 3 times and record the model, provider,
date and cost.
