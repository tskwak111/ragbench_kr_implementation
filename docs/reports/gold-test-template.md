# Sealed Gold Test Report

This report is completed only from the aggregate public export produced by
`scripts/run_gold_test.py`. Never paste question text, answers, evidence, document IDs,
raw snapshot IDs, provider responses, or protected checkpoint paths here.

## Preregistration

- Signed preregistration public ID:
- Signature verification record and signer:
- Frozen code commit:
- Three frozen public configuration IDs, in declared order:
- Gold scope: 300 items, or the declared reduced scope of exactly 150 with justification:
- Frozen metric set: correctness, faithfulness, citation, abstention, latency, cost,
  Hit, Recall, and MRR.
- Primary paired comparison:
- Exclusions:
- Stopping rule:
- Bootstrap: 95% document-cluster paired bootstrap, fixed seed, at least 10,000 resamples.

The preregistration must exist and pass its immutable hash check before `ALLOW_GOLD_ACCESS=1`
is set. A dry run verifies the signed artifact, exact ordered configuration hashes, and
public-safe sealed metadata without accepting or opening the gold file.

## Controlled execution record

- Operator and UTC start/end:
- Explicit command and approved environment:
- Dry-run artifact hash:
- Protected result repository identifier:
- Restart/checkpoint events:
- Same-cohort assertion for all three configurations:
- Completed responses per configuration:
- Provider billing reconciliation:

The gold run is one-way: after unsealing there are no configuration, prompt, threshold,
or cohort tuning controls. Checkpoints are immutable and keyed by hashed item identity.

## Aggregate results

Insert aggregate-only tables for all frozen metrics, overall and by question type. Include
the paired effect, 95% interval, sample count, document-cluster count, and resample count.
Describe latency units and cost currency. Do not infer significance solely from point estimates.

## Invalidations and lineage

List append-only invalidation records. Any bug or scoring/output change invalidates the affected
run. State the previous public result identifier, reason, replacement code commit, and whether a
new snapshot was justified. Never silently replace an output or reuse its result identity.

## Optional Solar comparison

Run Solar Pro 3 vs Pro 4 only after the core result exists, on the fixed budgeted subset declared
before that comparison. Label it **exploratory** unless it was part of the original signed
preregistration. It cannot change the core top-three result.

## Limitations and release check

- Calibration and benchmark-defect limitations:
- Scope reduction, if any:
- Missing/failed items (must follow the frozen stopping rule):
- Restricted-field scan completed:
- Public identifiers are salted hashes; no raw config, item, document, corpus, or snapshot IDs:
