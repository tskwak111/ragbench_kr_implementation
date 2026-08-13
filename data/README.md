# Corpus data policy

`data/raw/` is an ignored local-only directory. Do not add PDFs, documents with unclear licensing, or any source bytes to Git.

`configs/corpus.yaml` is an intentionally empty **draft** manifest, not a frozen corpus. A corpus can be frozen only after `CorpusManifest.validate(freeze=True)` succeeds and a human has reviewed the source, license, provenance, and sampled first/middle/last pages for every PDF.

Use `scripts/collect_corpus.py` only with `--operator-approved` and an approved local root. It copies one PDF atomically into `data/raw/`, hashes it, counts pages with `pypdf`, refuses overwrites, and writes a review report. An `unknown` redistribution status is reportable but blocks freeze.

Publish manifests with `CorpusManifest.export_public(path)`. The export retains titles, source URLs, license status, hashes, page counts, rationale, and snapshot ID, while omitting local paths and document bytes. Documents marked `nonredistributable` must never have their bytes distributed.
