# Phase 4H Frozen Outer Reporting v1

This stage executes the already-frozen cross-subject reporting policy on the
successful v1r1 subject aggregates.

## Inferential family

The inferential reporting unit is the predeclared family × severity macro for
each model, operating point, and metric.

For each condition it reports:

- C0 subject macro;
- CS subject macro;
- paired degradation `C0-CS`;
- deterministic 10,000-replicate subject-bootstrap 95% interval;
- frozen direction status;
- eligible and total subject counts;
- overall 61-subject, UniVR, and KFall strata;
- descriptive source denominators.

## Required detail

Individual fault variants and checkpoint seeds are retained as descriptive
subject-level detail with C0, CS, paired degradation point estimates, and
coverage. They are not promoted into additional inferential hypothesis
families.

The q25/median/q75 timing summaries are also retained descriptively.

## Governance

No arbitrary practical margin and no binary global robustness label are used.
`NO_DIRECTIONAL_CONCLUSION` is not an equivalence claim. Missing required
subject coverage produces an unresolved inferential condition rather than
silent deletion or zero imputation.

Outer results cannot change any already-frozen scientific or reporting rule.
