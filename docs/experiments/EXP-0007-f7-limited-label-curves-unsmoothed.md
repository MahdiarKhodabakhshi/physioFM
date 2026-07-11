---
id: EXP-0007
title: F7 — Limited-label curves (decide C2) on un-smoothed SEED-IV
status: done
created: 2026-06-28
run_date: 2026-06-17
agent: unknown (pre-log backfill)
phase: phase2-followup
verified: no
tags: label-efficiency, C2, seed-iv-raw, low-label
commits:
verdict: A label-efficiency FM win on raw DE — at 10% labels PC-pretrained beats raw-DE by ~6 pts and random-init by ~12; C2 leans positive in the low-label regime.
---

# EXP-0007 — F7 — Limited-label curves (decide C2) on un-smoothed SEED-IV

> **Status:** done · **Created:** 2026-06-28 · **Run:** 2026-06-17 · **Agent:** unknown (pre-log backfill) · **Phase:** phase2-followup

---

## 1. Why — hypothesis & motivation

C2 asks whether the pretrained representation is more *label-efficient* than raw DE.
Smoothed DE showed no margin. With dynamics restored (raw SEED-IV), test whether the
FM representation pays off as labels shrink — the canonical foundation-model promise.

---

## 2. Setup — exactly what was run

```bash
PY=/home/mahdiar/.conda/envs/xcqa/bin/python
$PY scripts/phase2_f7_label_curves.py
```

- **Data:** SEED-IV un-smoothed (`seed_iv_raw`).
- **Variant / config:** frozen encoder + the F4 matched head (balanced 2-layer MLP, per-fold val early stopping, class weighting), trained on label fractions {10%, 50%, 100%} of each fold's train segments.
- **Output dir:** `results/phase2/followup/f7/`

---

## 3. Status & run log

- 2026-06-17 — run completed (date inferred from result timestamps).
- 2026-06-28 — backfilled into experiment log.

---

## 4. Results  *(run date: 2026-06-17)*

acc % / macro-F1 %:

| Features | 10% labels | 50% labels | 100% labels |
| --- | ---: | ---: | ---: |
| raw_de | 40.92 / 35.21 | 50.06 / 43.63 | 50.80 / 44.54 |
| physiofm_pretrained | **46.69 / 38.64** | 49.63 / 42.29 | 50.76 / 43.34 |
| physiofm_random_init | 35.08 / 30.05 | 38.46 / 32.84 | 38.85 / 32.80 |

Results: `results/phase2/followup/f7/f7_label_curves.csv`.

---

## 5. Interpretation — agent's reading

A label-efficiency FM win on raw DE. At full labels PC-pretrained ties raw-DE
(~50%), but the FM margin **grows as labels shrink**: at 10% labels PC-pretrained
beats raw-DE by ~6 pts (46.7 vs 40.9) and random-init by ~12 pts. On un-smoothed DE
the pretrained representation is genuinely more label-efficient — the positive FM
result Stage 2 could not surface on smoothed DE. C2 leans positive on raw DE in the
low-label regime.

---

## 6. ✅ Your verification — *(reserved for Mahdiar)*

> Leave the agent's interpretation above untouched. Confirm or correct it here.

- [ ] **Verified** (set `verified: yes` in frontmatter when ticked)
- **Notes / corrections:**


---

## 7. Commits

- (uncommitted as of 2026-06-28 backfill — `scripts/phase2_f7_label_curves.py`)

---

## 8. Links

- Related entries: [[EXP-0004]] (shared matched head), [[EXP-0005]]
- Docs / results: `docs/PHASE2_FOLLOWUP.md` (F7), `results/phase2/followup/f7/`
