---
id: EXP-0005
title: F5 — Input-context (p_in) sweep on un-smoothed SEED-IV
status: done
created: 2026-06-28
run_date: 2026-06-17
agent: unknown (pre-log backfill)
phase: phase2-followup
verified: no
tags: context, p_in, p_out, seed-iv-raw, C4
commits:
verdict: C4 revived on raw DE — PC−random gap is large everywhere and grows with context (peak 18 pts at p_in=8/p_out=16); multi-horizon p_out=16 beats single-step.
---

# EXP-0005 — F5 — Input-context (p_in) sweep on un-smoothed SEED-IV

> **Status:** done · **Created:** 2026-06-28 · **Run:** 2026-06-17 · **Agent:** unknown (pre-log backfill) · **Phase:** phase2-followup

---

## 1. Why — hypothesis & motivation

F1/F2 flipped the Stage-2 null on un-smoothed DE, so the source doc prescribes
re-running the ablation grid on raw DE. C4 ("more context / multi-horizon helps")
was "not supported" on smoothed DE. Hypothesis: with the dynamics restored,
context and multi-horizon prediction should now matter. Sweep p_in × p_out.

---

## 2. Setup — exactly what was run

```bash
PY=/home/mahdiar/.conda/envs/xcqa/bin/python
bash scripts/run_f5.sh          # -> scripts/phase2_f5_context.py
```

- **Data:** SEED-IV un-smoothed (`seed_iv_raw`).
- **Variant / config:** p_in ∈ {1,4,8} × p_out ∈ {1,16}, matched PC vs random-init, paired with F2 trial-level readouts. gap = PC − random (GRU readout).
- **Output dir:** `results/phase2/followup/f5/`

---

## 3. Status & run log

- 2026-06-17 — run completed (date inferred from result timestamps).
- 2026-06-28 — backfilled into experiment log.

---

## 4. Results  *(run date: 2026-06-17)*

acc %; gap = PC − random (GRU readout):

| p_in | p_out | GRU PC | GRU rand | **GRU gap** | last PC | logreg PC | logreg rand |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 50.28 | 45.00 | **5.28** | 48.89 | 53.26 | 41.97 |
| 1 | 16 | 58.61 | 45.00 | **13.61** | 59.72 | 54.67 | 41.97 |
| 4 | 1 | 56.94 | 43.33 | **13.61** | 48.89 | 53.21 | 36.98 |
| 4 | 16 | 58.06 | 43.33 | **14.72** | 55.00 | 55.04 | 36.98 |
| 8 | 1 | 52.50 | 40.28 | **12.22** | 48.89 | 51.44 | 38.28 |
| 8 | 16 | 58.33 | 40.28 | **18.06** | 55.56 | 53.96 | 38.28 |

Results: `results/phase2/followup/f5/f5_context.csv`.

---

## 5. Interpretation — agent's reading

On raw DE, context and multi-horizon both help — C4 revived. The PC−random gap is
large at every config and **grows with context** (peak 18.1 at p_in=8/p_out=16).
Multi-horizon `p_out=16` beats single-step `p_out=1` at every p_in (gap 13.6 vs 5.3
at p_in=1) — the opposite of smoothed DE. The multi-horizon predictive objective is
useful precisely where temporal dynamics survive.

---

## 6. ✅ Your verification — *(reserved for Mahdiar)*

> Leave the agent's interpretation above untouched. Confirm or correct it here.

- [ ] **Verified** (set `verified: yes` in frontmatter when ticked)
- **Notes / corrections:**


---

## 7. Commits

- (uncommitted as of 2026-06-28 backfill — `scripts/phase2_f5_context.py`, `scripts/run_f5.sh`)

---

## 8. Links

- Related entries: [[EXP-0001]], [[EXP-0002]], [[EXP-0006]]
- Docs / results: `docs/PHASE2_FOLLOWUP.md` (F5), `results/phase2/followup/f5/`
