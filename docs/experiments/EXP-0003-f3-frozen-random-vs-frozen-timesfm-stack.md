---
id: EXP-0003
title: F3 — Frozen-random vs frozen-TimesFM stack
status: done
created: 2026-06-28
run_date: 2026-06-17
agent: unknown (pre-log backfill)
phase: phase2-followup
verified: no
tags: timesfm, transfer, ablation, smoothed-de
commits:
verdict: C6 refuted — a frozen random stack of TimesFM's exact 1280x20 shape matches the frozen pretrained TimesFM stack; pretrained weights add nothing over a big fixed mixer.
---

# EXP-0003 — F3 — Frozen-random vs frozen-TimesFM stack

> **Status:** done · **Created:** 2026-06-28 · **Run:** 2026-06-17 · **Agent:** unknown (pre-log backfill) · **Phase:** phase2-followup

---

## 1. Why — hypothesis & motivation

E1b showed a frozen pretrained TimesFM stack does slightly better than from-scratch.
Claim C6 reads that as "TimesFM's pretrained temporal priors transfer." Alternative:
the gain is just the trained structured I/O blocks + a high-dimensional random
projection, not the pretrained weights. Test by building a stack at TimesFM-2.5's
**exact** shape but with **random** frozen weights, and comparing matched.

---

## 2. Setup — exactly what was run

```bash
PY=/home/mahdiar/.conda/envs/xcqa/bin/python
# variant timesfm_rand in physiofm/physiofm_s.py
$PY scripts/phase2_pretrain.py --variant timesfm_rand --p_in 1 --p_out 16 ...
$PY scripts/phase2_extract_eval.py --model_dir results/phase2/followup/f3/timesfm_rand_pin1_pout16_linear ...
```

- **Data:** SEED-V & SEED-IV, smoothed DE.
- **Variant / config:** `timesfm_rand` = decoder stack at TimesFM-2.5 shape (d=1280, 20 layers, 16 heads, head_dim 80, intermediate 1280) but random + frozen, structured I/O blocks trained fresh (PC-MSE 0.0031 vs E1b 0.0039). Compared to frozen TimesFM (E1b) and from-scratch d=256 (E1a).
- **Output dir:** `results/phase2/followup/f3/`

---

## 3. Status & run log

- 2026-06-17 — run completed (date inferred from result timestamps).
- 2026-06-28 — backfilled into experiment log.

---

## 4. Results  *(run date: 2026-06-17)*

acc % / macro-F1 %:

| Stack | SEED-V logreg | SEED-V SVM | SEED-IV logreg | SEED-IV SVM |
| --- | ---: | ---: | ---: | ---: |
| frozen **TimesFM** (E1b) | 43.27 / 41.32 | 44.18 / 41.83 | 60.40 / 52.35 | 61.48 / 53.20 |
| frozen **random** (TimesFM shape) | 47.26 / 45.69 | 46.24 / 44.57 | 58.93 / 50.31 | 60.63 / 51.84 |
| from-scratch (E1a, d=256) | 45.58 / 44.10 | 46.13 / 44.53 | 57.49 / 48.93 | 57.41 / 49.12 |

Results: `results/phase2/followup/f3/`.

---

## 5. Interpretation — agent's reading

C6 refuted. A frozen *random* stack of the same 1280×20 shape matches (SEED-V: even
edges out) the frozen *pretrained* TimesFM stack. TimesFM's pretrained temporal
priors add nothing over "a big fixed nonlinear mixer" on smoothed DE; the apparent
"transfer" in E1b is the trained I/O blocks + a high-dimensional random projection,
not the pretrained weights. Consistent with random-init ≈ pretrained throughout
Stage 2. *Scope:* smoothed DE.

---

## 6. ✅ Your verification — *(reserved for Mahdiar)*

> Leave the agent's interpretation above untouched. Confirm or correct it here.

- [ ] **Verified** (set `verified: yes` in frontmatter when ticked)
- **Notes / corrections:**


---

## 7. Commits

- (the `timesfm_rand` variant in `physiofm/physiofm_s.py` — confirm/attach SHA if committed)

---

## 8. Links

- Related entries: [[EXP-0001]], [[EXP-0006]]
- Docs / results: `docs/PHASE2_FOLLOWUP.md` (F3), `results/phase2/followup/f3/`
