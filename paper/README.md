# ICASSP 2027 draft

Build: `latexmk -pdf main.tex` (IEEEtran conference mode; TeX Live 2023 suffices).
Camera-ready: swap to the official ICASSP template — `\documentclass{article}` +
`\usepackage{spconf}` (spconf.sty from the ICASSP author kit), move authors into
`\name{}/\address{}`, keep everything else.

Every number traces to `results/phase4/**` CSVs and `docs/experiments/EXP-0018..0028`;
the consolidated map is `docs/EXTERNAL_VALIDATION_RESULTS.md`. TODOs in main.tex:
author block/affiliations, blinding-rule check, SOTA citation spot-check
(flagged "verify" items in docs/paper_results_tables.xlsx).
