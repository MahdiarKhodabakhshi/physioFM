## Experiment log (MANDATORY for every agent — Claude Code, Cursor, Codex)

This project is driven by several coding agents, and the human needs one place to
track the latest progress and the *reasoning* behind it. The chronological lab
notebook lives in `docs/experiments/` (one file per experiment). Read
`docs/experiments/README.md` for the full convention. The rules apply to **all**
agents:

- **When you decide/start an experiment, create its entry first** (before results
  bias the reasoning). Scaffold it with the helper:
  `python docs/experiments/exp.py new --title "..." --agent <cursor|codex|claude-code>`
  then fill §1 *Why* and §2 *Setup*.
- **When it finishes,** fill §4 *Results* (with the run date), §5 *Interpretation*
  (your claim), set frontmatter `status`/`run_date`/`verdict`, list the commits in
  §7, then run `python docs/experiments/exp.py index`.
- **Never touch §6 or set `verified: yes`** — that section is reserved for Mahdiar
  to confirm or correct your interpretation.
- Record real outcomes including nulls/failures. If an experiment resolves a phase
  question, also fold its verdict into the matching `docs/PHASE*.md` table.
- The index is generated from each entry's frontmatter — edit frontmatter, then run
  `exp.py index`; never hand-edit the index table. `exp.py check` lists unfinished
  or unverified entries. The helper has no third-party dependencies.

(Claude Code users can invoke the `experiment-log` skill via `/experiment` instead
of calling the helper directly.)

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

When the user types `$graphify`, invoke the `skill` tool with `skill: "graphify"` before doing anything else.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- Dirty graphify-out/ files are expected after hooks or incremental updates; dirty graph files are not a reason to skip graphify. Only skip graphify if the task is about stale or incorrect graph output, or the user explicitly says not to use it.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
