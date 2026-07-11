---
name: experiment-log
description: Record and update PhysioFM experiments in the docs/experiments/ lab notebook. Use whenever an experiment is decided, started, or finished, or when the user asks to log/track/journal an experiment, capture why a run was done, record results & interpretation, or see the latest experiment progress across agents. Triggers include "/experiment", "log this experiment", "record the result", "what's the latest progress".
---

# PhysioFM Experiment Log

Maintain the chronological lab notebook at `docs/experiments/` so the human
(Mahdiar) can see, across Claude Code / Cursor / Codex, what was run, **why**, the
dated result, the agent's interpretation, and his own verification — plus the
related commits. Read `docs/experiments/README.md` for the full convention; this
skill is the Claude Code workflow over it.

The helper script does all mechanical work (IDs, dates, the index). Run it with
the repo interpreter from the repo root:

```bash
python docs/experiments/exp.py <cmd>
```

## Routing — pick the action from what the user is doing

### A. Starting / deciding an experiment → create the entry NOW
Do this *before* results exist, so the reasoning is captured unbiased.

1. `python docs/experiments/exp.py new --title "<concise title>" --phase <phase> --agent claude-code [--status running]`
   - This prints the new file path, recent commits, and recent result files.
2. Open the file and fill **§1 Why** (hypothesis/motivation — the most important
   part) and **§2 Setup** (exact command, data keys, variant, output dir). Pull
   the command from the user's request or the `scripts/` driver being run.
3. Leave §4–§6 as placeholders. Do **not** touch §6 (it's Mahdiar's).
4. `python docs/experiments/exp.py index` to refresh the index.

### B. An experiment finished → record the result
1. `python docs/experiments/exp.py result EXP-NNNN` (prints the path
   + recent commits/results to draw from).
2. Edit the entry:
   - **Frontmatter:** set `status: done` (or `blocked`), `run_date: <today>`,
     and a one-line `verdict:`.
   - **§3** append a dated run-log line. **§4** the actual numbers/tables + the
     `results/...` path, with the run date in the header. Report real outcomes,
     including nulls/failures.
   - **§5** your interpretation — clearly *a claim*, the takeaway and caveats.
   - **§7** the relevant commit SHAs (prune the printed list).
3. **Never** set `verified: yes` or tick §6 — that is Mahdiar's confirmation only.
4. If this resolves a phase question, also fold the verdict into the matching
   `docs/PHASE*.md` table (the curated results live there; this log is the journal).
5. `python docs/experiments/exp.py index`.

### C. "What's the latest progress?" → report
- `python docs/experiments/exp.py list` for the index, and
  `... check` for entries that are unfinished or awaiting Mahdiar's verification.
  Summarize newest-first; flag anything `running`, `blocked`, or done-but-unverified.

### D. Mahdiar verifying an interpretation
- Only when the user explicitly confirms: set `verified: yes`, tick §6's box, add
  his notes under §6, then `... index`.

## Guardrails
- Entry created when the experiment is decided, not retroactively, so §1 is honest.
- One file per experiment; the index is generated from frontmatter — edit frontmatter, then run `index`, never hand-edit the index table.
- Dates absolute (today is available in context); keep `run_date` and §4 in sync.
- §5 = agent claim, §6 = human verification. Keep them separate; agents stay out of §6.
- After committing experiment code, add the SHA to §7 and re-run `index`.
