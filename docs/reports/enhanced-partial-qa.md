# Partial Enhanced parse paired QA

Date: 2026-08-26 (Asia/Seoul)

## Decision

Do not switch the corpus default from Standard to Enhanced yet. Enhanced materially improved three
chart-heavy pages, but it was not consistently better: 14 pairs were equivalent for core retrieval,
3 favored Standard, and 10 were mixed or unsafe. Enhanced visual descriptions must not be treated as
ground truth without checking the source page.

## Scope and method

Enhanced completed for 3 of 20 documents and 185 of 1,981 pages:

- `mobis-climate-2023`: 11 pages
- `mobis-climate-2024`: 9 pages
- `mobis-sustainability-2024`: 165 pages

Thirty source pages were rendered and compared with their successful Standard and Enhanced elements:
all 20 pages from the two short climate reports plus 10 visual/table-heavy pages from the sustainability
report. The reviewer knew the modes, so this is a preliminary paired engineering review, not the planned
corpus-wide blinded gate. `Tie` means no material difference for retrieval; `mixed` means one mode added
useful structure but also introduced unsupported, noisy, or incorrectly associated content.

## Results

| Document/page | Result | Main finding |
|---|---:|---|
| `mobis-climate-2023` p1 | Tie | Cover text identical |
| `mobis-climate-2023` p2 | Tie | Prose and scenario table retain the same facts; Enhanced mainly adds line breaks |
| `mobis-climate-2023` p3 | Tie | Scope and methodology tables retain the same facts |
| `mobis-climate-2023` p4 | Tie | Risk matrix content retained in both modes |
| `mobis-climate-2023` p5 | Tie | Opportunity table content retained in both modes |
| `mobis-climate-2023` p6 | Mixed | Numeric table retained; Enhanced embeds a verbose formula image description without a verified fact gain |
| `mobis-climate-2023` p7 | Mixed | Formula descriptions are richer but duplicate already retained table evidence |
| `mobis-climate-2023` p8 | Mixed | Same prose/table evidence; Enhanced adds visual-description volume |
| `mobis-climate-2023` p9 | Mixed | Same numeric evidence; Enhanced adds unneeded formula narration |
| `mobis-climate-2023` p10 | Mixed | Same scenario values; Enhanced description is substantially longer but not more reliable |
| `mobis-climate-2023` p11 | Standard | Enhanced expands a decorative closing page into 959 characters of visual narration |
| `mobis-climate-2024` p1 | Tie | Title-page text identical |
| `mobis-climate-2024` p2 | Tie | Contents-page text identical |
| `mobis-climate-2024` p3 | Mixed | Dense two-column content retained; Enhanced adds duplication without fixing reading-order noise |
| `mobis-climate-2024` p4 | Tie | Risk table facts retained in both modes |
| `mobis-climate-2024` p5 | Mixed | Map and heatmap semantics remain unsafe; extra descriptions do not establish correct values |
| `mobis-climate-2024` p6 | Tie | Risk-impact tables retain the same numeric facts |
| `mobis-climate-2024` p7 | Tie | Transition-risk tables retain the same numeric facts |
| `mobis-climate-2024` p8 | Tie | Opportunity tables retain the same numeric facts |
| `mobis-climate-2024` p9 | Tie | Closing text identical |
| `mobis-sustainability-2024` p10 | Mixed | Enhanced explains the strategy graphic, but invents unsupported stacked-bar segment values |
| `mobis-sustainability-2024` p11 | Enhanced | Recovers the 4.2/6.1/9.7/12.2 line series with units and useful process semantics |
| `mobis-sustainability-2024` p16 | Standard | Financial/compensation tables are identical; Enhanced adds a hallucinated navigation-icon description |
| `mobis-sustainability-2024` p27 | Tie | Stakeholder inputs/outputs retained; both modes still contain nested-table structure |
| `mobis-sustainability-2024` p36 | Enhanced | Recovers the 30/80/100% roadmap where Standard emitted bogus chart values; associations still need source review |
| `mobis-sustainability-2024` p37 | Mixed | Improves percentage labels but misreads `557 TJ` as `5557` and over-interprets the radial graphic |
| `mobis-sustainability-2024` p59 | Enhanced | Correctly separates Cradle-to-Grave and Cradle-to-Gate categories and values |
| `mobis-sustainability-2024` p108 | Tie | Process and corrective-action tables match; Enhanced only narrates decorative arrows |
| `mobis-sustainability-2024` p110 | Standard | Standard keeps two cost-saving rows separate; Enhanced merges them into one cell |
| `mobis-sustainability-2024` p125 | Mixed | Flowchart narration is richer, but repeated decorative-icon hallucinations add noise and no table gain |

Totals: 3 Enhanced, 3 Standard, 14 tie, 10 mixed.

## Operational status

The partial Enhanced snapshot records USD 6.105000 local gross cost and has no open reservations.
It is backed up at the ignored local path
`.ragbench/backups/ragbench-enhanced-partial-185p-2026-08-26.dump` (SHA-256
`416dbbfa170a4f4bde70e2ec9d2806cb55b98cd11eac84fb4a7ebe1698f9159d`). Remaining paid requests
were stopped after repeated provider `401 Unauthorized` responses; failed attempts produced no
successful API usage rows.

Until the identical-corpus run and blinded review are complete, use Standard for prose/tables and
treat chart-derived values from either mode as conditional evidence. Enhanced chart output may be
evaluated selectively, but its generated figure descriptions are not trusted facts.
