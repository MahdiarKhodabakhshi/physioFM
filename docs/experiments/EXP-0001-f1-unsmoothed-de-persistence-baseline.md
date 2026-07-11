---
id: EXP-0001
title: F1 — Un-smoothed DE + persistence baseline (SEED-IV)
status: done
created: 2026-06-28
run_date: 2026-06-17
agent: unknown (pre-log backfill)
phase: phase2-followup
verified: no
tags: smoothing, persistence, seed-iv, flip-path
commits:
verdict: Stage-2 null flips on un-smoothed DE — PC-pretrain beats random by ~10–13 pts; LDS smoothing was hiding the dynamics.
---

# EXP-0001 — F1 — Un-smoothed DE + persistence baseline (SEED-IV)

> **Status:** done · **Created:** 2026-06-28 · **Run:** 2026-06-17 · **Agent:** unknown (pre-log backfill) · **Phase:** phase2-followup

---

## 1. Why — hypothesis & motivation

Stage-2 concluded "temporal predictive-coding adds nothing" — but that was measured
on **LDS-smoothed** DE. Hypothesis: LDS smoothing destroys the within-trial dynamics
PC could exploit, so the null is an artifact of the feature, not of emotion being
static. Test by holding everything fixed and changing only smoothed (`de_LDS`) vs
un-smoothed (`de_movingAve`) DE — the only dataset with an un-smoothed key is SEED-IV.
A persistence/variance baseline quantifies how much learnable dynamics survive.

---

## 2. Setup — exactly what was run

```bash
PY=/home/mahdiar/.conda/envs/xcqa/bin/python
bash scripts/run_f1.sh          # -> scripts/phase2_f1_smoothing.py
```

- **Data:** SEED-IV only; `de_LDS` (smoothed) vs `de_movingAve` (un-smoothed, key `seed_iv_raw`).
- **Variant / config:** matched PhysioFM-S `scratch`, d=256 / 6L, `p_in=1 → p_out=16`, trained SEED-IV-only so smoothing is the sole changed variable. Persistence + variance computed in the per-(C,B) corpus-standardized space the model trains in.
- **Output dir:** `results/phase2/followup/f1/`

---

## 3. Status & run log

- 2026-06-17 — run completed (date inferred from `results/phase2/followup/f1/` timestamps).
- 2026-06-28 — backfilled into experiment log.

---

## 4. Results  *(run date: 2026-06-17)*

| Variant | persistence MSE 1-step | persistence MSE multi-step | model PC-MSE | within-trial var frac |
| --- | ---: | ---: | ---: | ---: |
| SEED-IV smoothed (LDS) | 0.00001 | 0.00054 | 0.02274 | **0.1%** |
| SEED-IV un-smoothed (movingAve) | 0.10249 | 0.29675 | 0.12811 | **17.2%** |

Zero-shot linear probe (acc % / macro-F1 %), SEED-IV:

| DE variant | PC-pretrained (logreg) | random-init (logreg) | PC-pretrained (lin-SVM) | random-init (lin-SVM) |
| --- | ---: | ---: | ---: | ---: |
| smoothed (LDS) | 61.41 / 52.00 | 55.98 / 48.37 | 61.85 / 54.29 | 61.09 / 52.19 |
| un-smoothed (movingAve) | **54.67 / 45.99** | 41.97 / 35.53 | **52.14 / 44.13** | 44.33 / 38.42 |

Results: `results/phase2/followup/f1/f1_smoothing.{csv,md}`.

---

## 5. Interpretation — agent's reading

The Stage-2 null flips on un-smoothed DE. Under LDS the within-trial (dynamic)
signal is ~0.1% of variance, persistence is near-perfect (MSE 1e-5), and
PC-pretrained ≈ random-init — exactly the original null. On un-smoothed DE the
within-trial fraction is **~17×** larger, persistence is far from optimal, and
PC-pretraining beats random-init by **~10–13 points**. So "temporal PC adds
nothing" is substantially an **artifact of LDS smoothing destroying learnable
dynamics**, not evidence emotion is intrinsically static. The static-emotion claim
must be scoped to smoothed DE. *Scope:* SEED-IV only (sole un-smoothed feature key).

---

## 6. ✅ Your verification — *(reserved for Mahdiar)*

> Leave the agent's interpretation above untouched. Confirm or correct it here.

- [ ] **Verified** (set `verified: yes` in frontmatter when ticked)
- **Notes / corrections:**


---

## 7. Commits

<!-- The follow-up scripts were untracked at backfill time; add SHAs once committed. -->
- (uncommitted as of 2026-06-28 backfill — `scripts/phase2_f1_smoothing.py`, `scripts/run_f1.sh`)

---

## 8. Links

- Related entries: [[EXP-0002]] (readout side of the same flip), [[EXP-0005]], [[EXP-0006]], [[EXP-0007]]
- Docs / results: `docs/PHASE2_FOLLOWUP.md` (F1), `results/phase2/followup/f1/`
