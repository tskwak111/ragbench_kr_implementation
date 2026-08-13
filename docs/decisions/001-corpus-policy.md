# ADR 001: Corpus provenance, redistribution, and freeze policy

## Status

Accepted for implementation; the starter corpus is not frozen.

## Decision

Each corpus document must record a stable identifier, title, organization, year, document type, language, `sector` (`corporate` or `public`), `content_stratum` (`table_heavy`, `text_heavy`, or `mixed`), `template_family`, source URL, download date, license label, redistribution state, local path, SHA-256, PDF page count, and inclusion rationale. `redistributable`, `nonredistributable`, and `unknown` are the only supported redistribution states.

The collection utility accepts only an operator-approved regular PDF beneath an explicit approved root. It uses POSIX no-follow directory descriptors for every source/destination component, stages with an unpredictable tempfile name, hashes and parses staged descriptor bytes before publishing with atomic hard-link no-replace semantics, and removes staging files on failure. It never uses a fallback that could weaken these guarantees. Reports/fragments use the same no-replace design in a verified private output directory. The tool reports unknown licenses for review; it never downloads documents or determines legal status itself.

Freezing requires successful local PDF/hash/page validation, unique content, source and download provenance, no unknown redistribution states, configured document/page/diversity targets, both sectors, table-heavy and text-heavy strata, template-family concentration limits, and `status: frozen`. A `frozen` manifest always runs these checks even if `validate()` is called without `freeze=True`. Before changing that status, a reviewer manually inspects first, middle, and last pages of every file for corruption. The empty starter manifest must fail this gate.

The snapshot ID is a SHA-256 hash of canonical, sorted content hashes and stable metadata. It does not depend on YAML order or local absolute paths. Public exports omit all local paths and document bytes; they retain provenance and redistribution state. Nonredistributable source bytes remain local.

## Consequences

Actual source selection, licenses, acquisition, page inspection, target compliance, and the frozen snapshot are data-operations work requiring a human reviewer. This repository intentionally makes no claim that a corpus has been assembled or frozen.
