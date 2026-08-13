# Human validation and sealed gold protocol

This runbook defines the operator-only review that converts automatically validated synthetic
candidates into a restricted benchmark. The repository contains the protocol and an empty review
template only. It does not contain reviewed questions or a real gold file.

## Safety boundary

- Treat candidate questions, answers, evidence, reviewer notes, and gold item IDs as restricted.
- Keep restricted artifacts outside normal development logs and version control, according to the
  source documents' licenses.
- Never print, preview, sample, or log sealed gold content. Normal logs may contain only the output
  of `public_gold_metadata`: snapshot/version, file name, SHA-256, count, scope status, and sealing
  timestamp.
- `ALLOW_GOLD_ACCESS=1` alone is insufficient. Reading requires an explicit supported gold command
  and its explicit execution flag. Normal imports and ordinary test runs remain gold-blind.
- Use only `pytest -m gold` together with `ALLOW_GOLD_ACCESS=1` for tests that are deliberately
  marked `gold`. An inherited environment flag does not enable them.

## Select and randomize the review queue

1. Begin only with candidates that passed automated validation and whose immutable corpus, parse,
   and generation snapshots are known.
2. Use a recorded random seed with `plan_review_sample`. Select at least 300 items across question
   type, difficulty, document, parse-sensitive status, and answerability. The exported queue omits
   generator confidence so it cannot bias reviewers.
3. Preserve the generated queue's order. Do not reorder easy-looking questions first.
4. Fill a private copy of `data/benchmarks/review_template.csv`. The public file is header-only.

The review columns are: natural question; answer exists; evidence correct; page correct; answer
unambiguous; answerable label correct; type/difficulty correct; reviewer decision; corrected answer;
corrected evidence; notes; reviewer ID; and timezone-aware timestamp. Allowed decisions are
`accept`, `correct`, and `reject`. A correction must include corrected answer or evidence.

## Review every selected item

For every row, open the original PDF and the corresponding standard and enhanced parsed
representations. Independently verify the question, answerability, answer, verbatim evidence,
source page, type, and difficulty. Never approve an item using only the generated answer or parsed
text. Correct only when the intended question remains unambiguous and the original source supports
the correction; otherwise reject it. Record reviewer identity and a timezone-aware timestamp.

## Double review and adjudicate

Assign at least 50 identical items to two distinct reviewers. Before adjudication, calculate raw
categorical agreement and Cohen's kappa using `calculate_review_agreement`. Do not replace kappa
with raw agreement.

Disagreements require an independent identified adjudicator, an explicit final decision, and
written notes citing the rule applied after comparing the original PDF and both parses. The
`adjudicate_reviews` function rejects silent resolutions. Keep pre-adjudication reviews intact for
the agreement calculation.

## Leakage-safe split snapshots

Before assigning splits, give each accepted item a question-family ID and paraphrase-group ID. Use
`build_split_snapshots` with a recorded version and seed. It forms connected components across
shared documents, question families, and paraphrase groups, then assigns whole components to
`dev_auto`, `test_gold`, or `stress`. If the components cannot populate all splits without leakage,
the operation fails; revise the candidate pool rather than splitting a component.

Persist only the immutable split metadata in normal development storage. Restricted split content
stays in its licensed private location. `SplitSnapshot.model_dump()` excludes member item IDs and
contains only the split name/version, seed, item count, membership hash, and content-bound snapshot
ID. Never move a family, paraphrase, or shared-document item to another split after snapshot
creation; create a new version instead.

## Quality threshold and sealing

Seal exactly 300 accepted/adjudicated items only when the predeclared quality threshold is met. If
it is not met, seal exactly the approved scope floor of 150 and mark `scope_status` as `reduced`.
Anything below 150 is insufficient and must not be sealed. Do not describe the reduced snapshot as
full completion.

`seal_gold` sorts unique item IDs, writes canonical JSONL through a no-follow directory descriptor,
fsyncs a mode-0600 unpredictable temporary file, and atomically publishes it without replacement.
It returns content SHA-256 and public metadata. A published path is immutable: corrections require
a new version and path.

Store the returned public metadata as JSON without adding content-derived samples. To perform an
operator-authorized integrity check, use:

```console
ALLOW_GOLD_ACCESS=1 ragbench gold verify /restricted/gold-v1.jsonl \
  /public/gold-v1.metadata.json --execute --json
```

The command verifies file name, ownership/type, mode 0600, SHA-256, schema, and count. Its output is
public metadata only. There is intentionally no gold preview command.

## Completion record

Record the queue seed and distributions, reviewer assignments, agreement and kappa, adjudication
rule version, accepted/corrected/rejected counts, split metadata hashes, gold SHA-256, count, and
`full` or `reduced` scope. Manual review and sealing remain incomplete until an authorized human
operator performs these steps against the original documents.
