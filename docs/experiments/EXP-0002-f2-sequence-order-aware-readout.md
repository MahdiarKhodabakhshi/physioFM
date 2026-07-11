---
id: EXP-0002
title: F2 — Sequence-level / order-aware readout
status: done
created: 2026-06-28
run_date: 2026-06-17
agent: unknown (pre-log backfill)
phase: phase2-followup
verified: no
tags: readout, gru, order-control, seed-iv, seed-v
commits:
verdict: Temporal order matters only where dynamics survive — on un-smoothed DE gru beats shuffled by ~5 pts and PC beats random; on smoothed DE order-invariant.
---

# EXP-0002 — F2 — Sequence-level / order-aware readout

> **Status:** done · **Created:** 2026-06-28 · **Run:** 2026-06-17 · **Agent:** unknown (pre-log backfill) · **Phase:** phase2-followup

---

## 1. Why — hypothesis & motivation

The Stage-2 negative result used segment-level probes that ignore temporal order.
Maybe a fair, order-aware readout would surface temporal value the linear probe
hid. Test trial-level classification from frozen per-window hidden states with an
explicit **order control**: a GRU pool vs the same GRU on time-shuffled windows.
Run on both smoothed DE and (post-F1) un-smoothed SEED-IV.

---

## 2. Setup — exactly what was run

```bash
PY=/home/mahdiar/.conda/envs/xcqa/bin/python
$PY scripts/phase2_f2_sequence_readout.py
```

- **Data:** SEED-V & SEED-IV (smoothed, combined-corpus encoders); SEED-IV un-smoothed (`seed_iv_raw`, F1 encoders).
- **Variant / config:** readouts `last` (causal last-state → logreg), `gru` (order-respecting pool), `gru_shuf` (same GRU, windows shuffled = order control). Subject-dependent folds, seed 42.
- **Output dir:** `results/phase2/followup/f2/`

---

## 3. Status & run log

- 2026-06-17 — run completed (date inferred from result timestamps).
- 2026-06-28 — backfilled into experiment log.

---

## 4. Results  *(run date: 2026-06-17)*

**Smoothed DE** (combined-corpus encoders), acc % / macro-F1 %:

| Dataset | Readout | pretrained | random-init |
| --- | --- | ---: | ---: |
| SEED-V | last | 47.50 / 45.73 | 48.40 / 46.77 |
| SEED-V | gru | 43.33 / 40.31 | 35.49 / 31.14 |
| SEED-V | gru_shuf | 42.08 / 39.08 | 36.18 / 31.89 |
| SEED-IV | last | 56.39 / 48.81 | 61.39 / 54.53 |
| SEED-IV | gru | 51.67 / 44.22 | 55.56 / 46.07 |
| SEED-IV | gru_shuf | 51.67 / 43.15 | 55.00 / 44.28 |

**Un-smoothed DE** (SEED-IV, F1 encoders):

| Dataset | Readout | pretrained | random-init |
| --- | --- | ---: | ---: |
| SEED-IV raw | last | **59.72 / 51.30** | 48.89 / 42.30 |
| SEED-IV raw | gru | **58.61 / 50.23** | 45.00 / 38.25 |
| SEED-IV raw | gru_shuf | 53.61 / 44.69 | 39.44 / 34.11 |

Results: `results/phase2/followup/f2/` (incl. `f2_sequence_unsmoothed.csv`).

---

## 5. Interpretation — agent's reading

Order matters only where dynamics survive. On **smoothed** DE `gru ≈ gru_shuf`
(shuffling time costs ~0–1 pt) and pretrained ≯ random — order-invariant/static,
null holds even with a fair temporal readout. On **un-smoothed** DE the order
control bites: pretrained `gru` (58.6) beats `gru_shuf` (53.6) by ~5 pts, and
pretrained beats random under every readout (last 59.7 vs 48.9). Temporal order is
genuinely discriminative once LDS smoothing is removed — corroborates F1 from the
readout side.

---

## 6. ✅ Your verification — *(reserved for Mahdiar)*

> Leave the agent's interpretation above untouched. Confirm or correct it here.

- [ ] **Verified** (set `verified: yes` in frontmatter when ticked)
- **Notes / corrections:**


---

## 7. Commits

- (uncommitted as of 2026-06-28 backfill — `scripts/phase2_f2_sequence_readout.py`)

---

## 8. Links

- Related entries: [[EXP-0001]], [[EXP-0005]]
- Docs / results: `docs/PHASE2_FOLLOWUP.md` (F2), `results/phase2/followup/f2/`
