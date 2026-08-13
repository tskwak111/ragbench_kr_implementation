# Corpus data policy

`data/raw/` is an ignored local-only directory. Do not add PDFs, documents with unclear licensing, or any source bytes to Git.

`configs/corpus.yaml` is an intentionally empty **draft** manifest, not a frozen corpus. A corpus can be frozen only after `CorpusManifest.validate(freeze=True)` succeeds and a human has reviewed the source, license, provenance, and sampled first/middle/last pages for every PDF.

Use `scripts/collect_corpus.py` only with `--operator-approved`, an approved local root, and existing private raw/output directories. It requires POSIX no-follow descriptor primitives and fails closed on other platforms. It reads the source through a verified descriptor, stages in the verified raw directory, hashes and parses staged bytes with `pypdf`, then atomically publishes with a no-replace hard link. It refuses symlink traversal and overwrites. Review reports and manifest fragments must be direct, non-overwriting files inside `--private-output-dir`. An `unknown` redistribution status is reportable but blocks freeze.

Cooperating collector runs are serialized by an advisory `flock` in the trusted private output directory. A partial previous run is accepted only when its final artifact bytes exactly match the current staged artifact; conflicting output aborts the new run. Published outputs are never unlinked by failure cleanup. Temp cleanup verifies the staged inode first, but its safety assumes the operator-private directory is not being maliciously modified by another same-UID process; there is no claim of hostile same-UID atomic deletion protection.

Every record identifies a `sector` (`corporate` or `public`), a `content_stratum` (`table_heavy`, `text_heavy`, or `mixed`), and a `template_family`. Freezing requires both sectors, table-heavy and text-heavy coverage, and configurable organization/template-family concentration caps.

Publish manifests with `CorpusManifest.export_public(path)`. The export retains titles, source URLs, license status, hashes, page counts, rationale, and snapshot ID, while omitting local paths and document bytes. Documents marked `nonredistributable` must never have their bytes distributed.
