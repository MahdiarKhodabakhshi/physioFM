---
id: EXP-0004
title: F4 — Matched downstream head
status: done
created: 2026-06-28
run_date: 2026-06-17
agent: unknown (pre-log backfill)
phase: phase2-followup
verified: no
tags: head, mlp, overfit, seed-iv, seed-v
commits:
verdict: The head is not the lever — a matched MLP stays in the ~41–55% band on every feature set; the gap to PC-SSL's 84–92% is not the downstream head.
---

# EXP-0004 — F4 — Matched downstream head

> **Status:** done · **Created:** 2026-06-28 · **Run:** 2026-06-17 · **Agent:** unknown (pre-log backfill) · **Phase:** phase2-followup

---

## 1. Why — hypothesis & motivation

Maybe PhysioFM-S representations carry nonlinear value a linear probe can't read,
and the gap to PC-SSL's published 84–92% is just a weak readout head. Test one
identical frozen-encoder + MLP head applied across raw-DE, PC-pretrained, and
random-init, so the head is held constant and only the features vary.

---

## 2. Setup — exactly what was run

```bash
PY=/home/mahdiar/.conda/envs/xcqa/bin/python
$PY scripts/phase2_f4_matched_head.py
```

- **Data:** SEED-V (chance 20%), SEED-IV (chance 25%), smoothed DE.
- **Variant / config:** identical 2-hidden-layer MLP head `(256,128)`, balanced oversampling, per-fold validation early stopping. `logreg` (linear probe) and `mlp` (un-balanced) shown for context.
- **Output dir:** `results/phase2/followup/f4/`

---

## 3. Status & run log

- 2026-06-17 — run completed (date inferred from result timestamps).
- 2026-06-28 — backfilled into experiment log.

---

## 4. Results  *(run date: 2026-06-17)*

**SEED-V (chance 20%)** — acc % / macro-F1 %:

| Features | logreg | mlp | mlp_bal |
| --- | ---: | ---: | ---: |
| raw_de | 51.40 / 49.92 | 40.10 / 37.43 | 40.81 / 38.92 |
| physiofm_pretrained | 45.50 / 44.01 | 44.37 / 42.15 | 45.10 / 43.07 |
| physiofm_random_init | 48.54 / 46.57 | 42.53 / 40.64 | 43.16 / 41.17 |

**SEED-IV (chance 25%)**:

| Features | logreg | mlp | mlp_bal |
| --- | ---: | ---: | ---: |
| raw_de | 62.75 / 54.76 | 48.93 / 41.58 | 54.64 / 45.85 |
| physiofm_pretrained | 57.49 / 48.93 | 51.72 / 44.25 | 55.06 / 46.40 |
| physiofm_random_init | 60.68 / 53.11 | 49.55 / 45.29 | 53.65 / 48.60 |

Results: `results/phase2/followup/f4/f4_matched_head.csv`.

---

## 5. Interpretation — agent's reading

The head is not the lever. The matched MLP does not reach the 80s on any feature
set; with balancing it lands at 40.8 / 54.6 (raw-DE) — *below* the linear probe
(51 / 63), confirming overfit on ~600 labels/fold. All three feature sets sit in
the same ~41–55% band. The gap to PC-SSL's 84–92% is **not** explained by the
downstream head, and PhysioFM-S carries no MLP-accessible value the probe hid. This
sharpens suspicion onto the PC-SSL number itself (→ F12).

---

## 6. ✅ Your verification — *(reserved for Mahdiar)*

> Leave the agent's interpretation above untouched. Confirm or correct it here.

- [ ] **Verified** (set `verified: yes` in frontmatter when ticked)
- **Notes / corrections:**


---

## 7. Commits

- (uncommitted as of 2026-06-28 backfill — `scripts/phase2_f4_matched_head.py`)

---

## 8. Links

- Related entries: [[EXP-0008]] (F12, the PC-SSL number), [[EXP-0007]]
- Docs / results: `docs/PHASE2_FOLLOWUP.md` (F4), `results/phase2/followup/f4/`
