# Contrast-design evaluation set

One YAML case per dataset (plan `docs/plans/2026-10-27-03-contrast-design-and-default-selection.md`,
section 5). Every case is a **draft — owner to confirm** (`status` field) until the owner
checks the design, the expected contrasts and their roles.

Drafted 2026-10-27 from the stored metadata of the run `results/20261007_PPARg/` (not in the
repository). The committed fixtures in `tests/fixtures/contrast_design/` hold the sample
columns that the cases refer to. That run had no research question ("Not specified"), so each
case states the question it assumes in `research_question_assumed`.

| Case | Difficulty | Point of the case |
|---|---|---|
| GSE298581 | difficult | Needs a `cell_line` block: `~0 + group + cell line`, 42 residual df |
| GSE295568 | difficult | Batch must be dropped (groups nested in batch); rescue arms vs shPPARG are primary |
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
- `contrasts.required`: each item has `target` and `reference` (lists of group labels; more
  than one label means an equal-weight average), a `type` (the `contrast_type` values of plan
  section 3.2) and the expected `role` (`primary`, `supporting`, `not_relevant`; plan section
  3.3).
- `contrasts.optional`: acceptable when present; listed with their expected role.
- `contrasts.not_primary`: may be present but must not be `primary` or a default.
  `any_pair_within` means any pairwise contrast between those groups.
- `contrasts.must_be_absent`: contrasts or rules (`no_cross_block`) that should not appear.
- `current_run`: what the 2026-10-07 run produced, the "before" measurement.
- `owner_to_confirm`: open points for the owner.

## Scoring (proposed, not yet implemented)

Deterministic. Convert each run contrast with `uorca.core.contrast_checks.check_contrasts` and
take `canonical_pair()`; convert each case item to the same pair through the group selectors.
Score: required pairs present / required, `not_primary` pairs marked primary, `must_be_absent`
hits, formula match, and residual df match. A reversed contrast (reference - target) should
count as present with a direction note. Run each case 3 times and record the model, provider,
date and cost.
