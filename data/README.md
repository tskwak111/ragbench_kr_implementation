# Corpus data policy

`data/raw/` is an ignored local-only directory. Do not add PDFs, documents with unclear licensing, or any source bytes to Git.

`configs/corpus.yaml` is an intentionally empty **draft** manifest, not a frozen corpus. A corpus can be frozen only after `CorpusManifest.validate(freeze=True)` succeeds and a human has reviewed the source, license, provenance, and sampled first/middle/last pages for every PDF.

Before collection, create the raw and private manifest directories and restrict both to the current operator:

```bash
mkdir -p data/raw /operator-private/ragbench-manifests
chmod 0700 data/raw /operator-private/ragbench-manifests
```

Use `scripts/collect_corpus.py` only with `--operator-approved`, an approved local root, and those existing private raw/output directories. The collector opens them without following symlinks, then fails closed unless each directory is owned by the current effective UID and has no group/world permissions. It requires POSIX no-follow descriptor primitives and fails closed on other platforms. It reads the source through a verified descriptor, stages in the verified raw directory, hashes and parses staged bytes with `pypdf`, rejects PDFs with no pages, then atomically publishes with a no-replace hard link. It refuses symlink traversal and overwrites. Review reports and manifest fragments must be direct, non-overwriting files inside `--private-output-dir`. An `unknown` redistribution status is reportable but blocks freeze.

Unpredictable `O_EXCL` staging names and exclusive directory permissions prevent other OS users from replacing staging names. Cooperating collector runs are serialized by an advisory `flock` in the private output directory. Same-UID processes are inside the trusted/cooperating boundary; the collector does not claim protection against a malicious process running as the same UID. A partial previous run is accepted only when its final artifact bytes exactly match the current staged artifact; conflicting output aborts the new run. Published outputs are never unlinked by failure cleanup. Temp cleanup removes only names generated and opened exclusively by that invocation.

Every record identifies a `sector` (`corporate` or `public`), a `content_stratum` (`table_heavy`, `text_heavy`, or `mixed`), and a `template_family`. Freezing requires both sectors, table-heavy and text-heavy coverage, and configurable organization/template-family concentration caps.

Publish manifests with `CorpusManifest.export_public(path)`. The export retains titles, source URLs, license status, hashes, page counts, rationale, and snapshot ID, while omitting local paths and document bytes. Documents marked `nonredistributable` must never have their bytes distributed.
