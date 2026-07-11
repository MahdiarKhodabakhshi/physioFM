---
id: EXP-0006
title: F6 — Scale check on un-smoothed SEED-IV
status: done
created: 2026-06-28
run_date: 2026-06-17
agent: unknown (pre-log backfill)
phase: phase2-followup
verified: no
tags: scale, model-size, seed-iv-raw
commits:
verdict: The PC−random gap is real and scale-stable on raw DE — ~5 pts at 1M params, opening to ~13 pts by 4M and plateauing through 15M.
---

# EXP-0006 — F6 — Scale check on un-smoothed SEED-IV

> **Status:** done · **Created:** 2026-06-28 · **Run:** 2026-06-17 · **Agent:** unknown (pre-log backfill) · **Phase:** phase2-followup

---

## 1. Why — hypothesis & motivation

The PC−random gap on raw DE (F1/F5) could be a small-model artifact that vanishes
with scale, or one that only appears at scale. Test whether it is robust across the
model-size ladder at fixed p_in=1, p_out=16.

---

## 2. Setup — exactly what was run

```bash
PY=/home/mahdiar/.conda/envs/xcqa/bin/python
bash scripts/run_f6.sh          # -> scripts/phase2_f6_scale.py
```

- **Data:** SEED-IV un-smoothed (`seed_iv_raw`).
- **Variant / config:** fixed p_in=1, p_out=16; hidden/layers ∈ {128/4, 256/6, 512/8}; pretrained−random gap (GRU readout) vs size.
- **Output dir:** `results/phase2/followup/f6/`

---

## 3. Status & run log

- 2026-06-17 — run completed (date inferred from result timestamps).
- 2026-06-28 — backfilled into experiment log.

---

## 4. Results  *(run date: 2026-06-17)*

| hidden | layers | params (M) | GRU PC | GRU rand | **GRU gap** | logreg gap |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 4 | 1.08 | 53.61 | 48.61 | **5.00** | 8.82 |
| 256 | 6 | 3.72 | 58.61 | 45.00 | **13.61** | 12.71 |
| 512 | 8 | 15.31 | 57.78 | 44.44 | **13.33** | 13.11 |

Results: `results/phase2/followup/f6/f6_scale.csv`.

---

## 5. Interpretation — agent's reading

The gap is real and scale-stable. On un-smoothed DE the pretrained−random gap is
robustly positive at every size, opening from ~5 pts at 1M params to ~13 pts by 4M
and plateauing through 15M. The pretraining benefit is not a sub-million-parameter
artifact — present across the ladder (and the SEED corpus is tiny, so larger scale
neither creates nor destroys it).

---

## 6. ✅ Your verification — *(reserved for Mahdiar)*

> Leave the agent's interpretation above untouched. Confirm or correct it here.

- [ ] **Verified** (set `verified: yes` in frontmatter when ticked)
- **Notes / corrections:**


---

## 7. Commits

- (uncommitted as of 2026-06-28 backfill — `scripts/phase2_f6_scale.py`, `scripts/run_f6.sh`)

---

## 8. Links

- Related entries: [[EXP-0005]], [[EXP-0003]]
- Docs / results: `docs/PHASE2_FOLLOWUP.md` (F6), `results/phase2/followup/f6/`
