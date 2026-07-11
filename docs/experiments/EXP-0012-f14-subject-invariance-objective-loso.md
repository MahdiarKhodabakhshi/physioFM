---
id: EXP-0012
title: F14 — Subject-invariance objective on DE, evaluated on LOSO (the unsaturated regime)
status: done
created: 2026-06-28
run_date: 2026-06-28
agent: claude-code
phase: phase3
verified: no
tags: subject-invariance, loso, cross-subject, adversarial, coral, clisa, runnable-now
commits:
verdict: NEGATIVE (for the invariance objective). On strict LOSO the adversarial subject-invariance term does NOT lift the encoder above raw-DE: the only gain is on SEED (+3 pts) and it comes from training a deep encoder at all (dann_l0, lambda=0) as much as from the adversarial term; on SEED-IV/-V the FM ties or loses. Probe-time CORAL collapses accuracy (whitening removes the absolute band-power level — same mechanism as C5 instance-norm). Points at the DE feature as the bottleneck → motivates raw-EEG (F15). Scope caveat: only DANN+CORAL arms run; the CLISA contrastive and frozen-PC-FM arms were not.
---

# EXP-0012 — F14 — Subject-invariance objective on DE, evaluated on LOSO

> **Status:** done · **Created:** 2026-06-28 · **Run:** 2026-06-28 · **Agent:** claude-code · **Phase:** phase3

---

## 1. Why — hypothesis & motivation

The diagnosis (EXP-0010 F10) showed within-subject emotion is **linearly saturated**
— no learned representation can beat a linear DE probe at full labels, so the
proposed method *cannot* win there. The **one regime that is NOT saturated is
cross-subject (LOSO)**, where every classifier sits near the floor (~35–40%). The
literature's biggest cross-subject gains come not from forecasting but from
**subject-invariance** objectives — contrastive inter-subject alignment (CLISA),
adversarial domain alignment (MS-MDA), CORAL. The proposal's temporal-PC objective
optimized next-window prediction, which targets nothing about the actual hard
problem (cross-subject distribution shift).

**Hypothesis.** Adding a subject-invariance term — and/or aligning features at probe
time — improves cross-subject (LOSO) emotion recognition, and improves the FM
encoder *more* than it improves raw-DE, because the FM has capacity to absorb an
invariance constraint that a fixed feature cannot.

This is the concrete instantiation of the spec's F8/F11. **Cheap, existing infra,
no new data** — the highest-leverage thing to try before the raw-EEG rebuild (F15 /
EXP-0013).

**Pre-registered prediction.** On LOSO: `invariance-trained FM` > `PC-FM` ≈
`random-init` ≈ `raw-DE` (current LOSO ≈ 31.9 / 37.8, SEED-V / SEED-IV from
`docs/PHASE2.md`). Probe-time alignment (CORAL) lifts all methods somewhat; the
*adversarial/contrastive* training term is what should lift the FM specifically.

**Decision rule (from spec F8/F11).**
- Invariance machinery helps the FM **more** than raw-DE → cross-subject is where
  the FM's value lives; pivot the thesis to subject-invariant pretraining.
- Helps **all methods equally** → the shift is just hard; alignment is a generic
  trick, not an FM result.
- Helps **nothing** → invariance objective is insufficient on DE; the DE bottleneck
  itself is the limiter → motivates F15 (raw EEG).

---

## 2. Setup — exactly what to run (PLAN — not yet executed)

**Reuse:** `physiofm/physiofm_s.py` encoder, `physiofm/structured_data.py` DE
pipeline, and `physiofm/phase2_eval.py::loso_eval` (already the subject-independent
harness, seed 42).

**New:** `scripts/phase2_f14_invariance.py` — train the encoder + linear head with a
subject-invariance term, then score through `loso_eval`. Arms:
1. **raw-DE + CORAL** (probe-time alignment baseline; no model).
2. **PC-FM** (frozen, EXP-0001-style encoder) + CORAL probe.
3. **FM + subject-adversarial** training: gradient-reversal (DANN) head predicting
   subject id, so the encoder is pushed to be subject-invariant.
4. **FM + CLISA-style contrastive alignment**: pull together windows of *different
   subjects under the same emotion*, push apart different emotions.
5. **random-init** control for arms 3–4 (the matched no-pretrain baseline).

- **Data:** `seed`, `seed_iv`, `seed_v` (DE on disk). Primary metric: **LOSO** acc /
  macro-F1 (subject-dependent reported for context only).
- **Output dir:** `results/phase3/f14/`.

---

## 3. Status & run log

- 2026-06-28 — created (pre-registered). Runnable on existing infra; no data block.
- 2026-06-28 — built `scripts/phase2_f14_invariance.py` (DANN gradient-reversal +
  CORAL) + `scripts/run_f14.sh`; smoke test on SEED-IV passed (`raw_de` LOSO 40.0%
  ≈ the known raw-DE baseline, so the harness is correct). **Launched the full run**
  (seed/seed_iv/seed_v × {raw_de, raw_de_coral, dann_l0, dann_adv}, 40 epochs) →
  `results/phase3/f14/`.
- 2026-06-28 — full run completed → `results/phase3/f14/f14_invariance.{csv,md}`.
  **Scope note:** the four arms actually run were `raw_de` (fixed feature baseline),
  `raw_de_coral` (probe-time CORAL alignment), `dann_l0` (deep DE encoder + emotion
  head, **no** invariance term — the λ=0 control), and `dann_adv` (same encoder + a
  gradient-reversal subject classifier — the adversarial invariance arm). The planned
  **CLISA contrastive** arm and the frozen **PC-FM (+CORAL)** arm from §2 were **not**
  run.

---

## 4. Results  *(run date: 2026-06-28)*

Strict inductive LOSO, acc % / macro-F1 %, mean ± std over held-out subjects.
Full table: `results/phase3/f14/f14_invariance.md`.

| Arm | SEED LOSO | SEED-IV LOSO | SEED-V LOSO |
| --- | ---: | ---: | ---: |
| raw_de (fixed feature) | 54.41 / 49.37 | 40.02 / 34.80 | 34.24 / 28.39 |
| raw_de_coral (probe-time align) | 33.27 / 16.64 | 27.85 / 13.09 | 19.89 / 8.20 |
| dann_l0 (deep encoder, λ=0) | **57.81 / 52.43** | 37.73 / 30.64 | 31.13 / 23.91 |
| dann_adv (adversarial invariance) | 56.85 / 51.19 | 39.67 / 33.78 | 30.28 / 23.38 |

(Reference LOSO from `docs/PHASE2.md`: raw-DE ≈ 31.9 / 37.8 on SEED-V / SEED-IV.)

---

## 5. Interpretation — agent's reading

**The subject-invariance objective does not deliver the pre-registered win.** The
prediction was `invariance-FM > dann_l0 ≈ raw_de` on LOSO. Instead:

- **The adversarial term adds nothing over just training an encoder.** `dann_adv` never
  beats `dann_l0`: SEED 56.9 vs 57.8, SEED-IV 39.7 vs 37.7, SEED-V 30.3 vs 31.1. The
  gradient-reversal invariance penalty is a wash — whatever cross-subject robustness the
  encoder has comes from supervised training, not from being pushed subject-invariant.
- **The FM only beats raw-DE on one of three datasets.** On SEED the deep encoder gains
  ~+3 pts (57.8 vs 54.4), but on SEED-IV and SEED-V it *ties or loses* to the fixed
  raw-DE feature (37.7/31.1 vs 40.0/34.2). So there is no consistent "the FM has capacity
  a fixed feature lacks" effect — this lands on the **"helps all/nothing equally →
  the shift is just hard, alignment is not an FM result"** branch of the decision rule.
- **Probe-time CORAL is actively harmful** (SEED 54.4 → 33.3, SEED-V 34.2 → 19.9,
  macro-F1 near floor). Whitening each held-out subject to the train covariance destroys
  the discriminative signal — the *absolute per-(C,B) band-power level*. This is the
  **same mechanism as C5 / the instance-norm collapse** (`docs/PHASE2.md` E3.2): any
  alignment that removes the absolute level guts DE emotion. Alignment must preserve the
  level, which CORAL does not.

**Net:** targeting subject-invariance on DE is **not** the lever. Combined with
[[EXP-0010]] (DE is linearly saturated) and [[EXP-0011]] (no SSL objective helps on
smoothed DE), this points the finger at the **DE feature itself** as the bottleneck, not
the objective, the classifier, or the invariance framing — which is exactly the
motivation to move off DE onto the raw waveform ([[EXP-0013]] / F15). *Caveat:* the
CLISA *contrastive* inter-subject arm was not run, so the specific "pull same-emotion /
different-subject windows together" idea remains formally untested; but the adversarial
result and the CORAL collapse both argue against invariance-on-DE paying off.

---

## 6. ✅ Your verification — *(reserved for Mahdiar)*

> Leave the agent's interpretation above untouched. Confirm or correct it here.

- [ ] **Verified** (set `verified: yes` in frontmatter when ticked)
- **Notes / corrections:**


---

## 7. Commits

<!-- to be filled as F14 lands -->

---

## 8. Links

- Builds on: [[EXP-0010]] (LOSO = the unsaturated regime), [[EXP-0008]] (eval honesty)
- Next/parallel: [[EXP-0013]] (F15 — raw-EEG FM, the deeper fix)
- Spec: `docs/PhysioFM_Stage2_FollowUp_Experiments.md` (F8 / F11)
- Refs: CLISA (arXiv:2109.09559); MS-MDA; MS-DCDA (arXiv:2408.10235)
- ⚠️ Eval honesty: report **strict inductive LOSO** (no target data in training);
  many published cross-subject numbers are *transductive* DA (see target data) — cf.
  the leakage finding in [[EXP-0008]].
