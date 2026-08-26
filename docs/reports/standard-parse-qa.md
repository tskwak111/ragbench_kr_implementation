# Standard parse manual QA

Date: 2026-08-26 (Asia/Seoul)

## Decision

Standard is conditionally accepted as the baseline parse for prose retrieval and table-oriented
RAG. It is not accepted as sole evidence for chart-derived numeric answers. Enhanced parsing and a
blinded paired review remain required before choosing the production parse mode.

## Method

Thirty source pages were rendered and compared with the corresponding
`document-parse-260630` elements: six each for complex tables, financial statements, charts,
multi-column layouts, and ordinary/mixed prose. `Pass` means the page is usable without a material
content defect; `conditional` means useful content is present but structure, order, or metadata is
degraded; `fail` means chart-derived facts are unsafe even if surrounding prose remains useful.

## Results

| Stratum | Document/page | Result | Main finding |
|---|---|---:|---|
| Complex table | `sds-sustainability-2023` p51 | Pass | Values retained; navigation glyph noise |
| Complex table | `sds-sustainability-2022` p3 | Pass | Multi-column contents retained as separate tables |
| Complex table | `lge-sustainability-2024-2025` p80 | Pass | Content retained; minor cross-column ordering |
| Complex table | `kt-esg-appendix-2024` p4 | Conditional | Large table split; nested raw HTML in Markdown |
| Complex table | `mobis-climate-2024` p3 | Conditional | Main values retained; placeholders and label noise |
| Complex table | `samsung-electronics-sustainability-2024` p16 | Conditional | Tables useful; embedded chart semantics degraded |
| Financial statement | `kt-esg-appendix-2024` p3 | Conditional | Values retained; row labels repeated as nested HTML |
| Financial statement | `hyundai-sustainability-2024` p97 | Conditional | Values retained; several unit cells lost `십억` |
| Financial statement | `samsung-electronics-sustainability-2023` p101 | Conditional | Values retained; two tables wrapped in nested HTML |
| Financial statement | `samsung-electronics-sustainability-2024` p56 | Conditional | Values retained; nested-table structure |
| Financial statement | `sds-sustainability-2022` p92 | Pass | Financial, tax, and distribution tables retained |
| Financial statement | `mobis-sustainability-2023` p125 | Pass | Financial and R&D tables retained |
| Chart | `sds-sustainability-2021` p47 | Fail | One salary value lost a digit; decorative pies invented |
| Chart | `koreanair-esg-2024` p54 | Fail | Counts mislabeled as percentages/feet; series conflated |
| Chart | `mobis-sustainability-2023` p13 | Fail | Patent series and cumulative value conflated |
| Chart | `hyundai-sustainability-2024` p28 | Fail | Multi-series line chart reduced to one series; bogus units |
| Chart | `mobis-sustainability-2024` p59 | Fail | Pie categories mapped to incompatible values |
| Chart | `kt-esg-factbook-2024` p82 | Fail | Year blocks and chart/table values fragmented |
| Multi-column | `samsung-electronics-sustainability-2023` p13 | Conditional | Content retained but left/right metric order interleaves |
| Multi-column | `kt-esg-factbook-2024` p80 | Conditional | Director fields detached from their rows |
| Multi-column | `sds-sustainability-2022` p96 | Pass | Dense two-column opinion remains readable |
| Multi-column | `mobis-netzero-2023` p6 | Conditional | Prose retained; chart labels and values interleave |
| Multi-column | `lge-sustainability-2024-2025` p48 | Conditional | Process begins at Step 6 before Steps 1–5 |
| Multi-column | `sds-sustainability-2024` p146 | Pass | Two-column opinion retained in useful order |
| Prose/mixed | `kostat-household-2024q4` p3 | Pass | Bullets, numbers, and URLs retained |
| Prose/mixed | `mobis-climate-2023` p2 | Pass | Prose and scenario table retained |
| Prose/mixed | `koreanair-esg-2024` p47 | Pass | Three-column article retained in reading order |
| Prose/mixed | `hyundai-sustainability-2024` p92 | Pass | Three risk columns retained in reading order |
| Prose/mixed | `samsung-electronics-sustainability-2024` p49 | Pass | Three-column security text retained |
| Prose/mixed | `mobis-sustainability-2023` p53 | Pass | Three-column chemical-management text retained |

Totals: 13 pass, 11 conditional, 6 chart-fact failures. All 30 pages produced parse elements; the
six failures are scope failures for chart-grounded numeric use, not empty-page failures.

## Operational rule

Until paired QA is complete, chunk and retrieve Standard prose and tables, but exclude chart-derived
numeric claims unless the same value is independently present in prose or a table. No Standard vs
Enhanced quality claim is made from this baseline review.
