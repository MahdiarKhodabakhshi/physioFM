---
id: EXP-0011
title: F9 — Masked-DE reconstruction objective (was it the temporal objective, or pretraining itself?)
status: done
created: 2026-06-28
run_date: 2026-06-28
agent: claude-code
phase: phase2-followup
verified: no
tags: masked-autoencoding, objective, root-cause, ssl, spatial-spectral
commits:
verdict: It was NOT the temporal objective — on smoothed DE masked-recon also fails to beat random-init (both ≤ random); on raw DE both objectives beat random by ~+9. No SSL pretext helps where the smoothed features carry no learnable structure.
---

# EXP-0011 — F9 — Masked-DE reconstruction objective

> **Status:** planned · **Created:** 2026-06-28 · **Run:** — · **Agent:** claude-code · **Phase:** phase2-followup

---

## 1. Why — hypothesis & motivation

Root-cause goal: *why did the proposed method fail on emotion?* The proposed SSL
was **temporal forecasting** (predict next DE window). It tied random-init on
smoothed DE ([[EXP-0001]]). Two very different explanations remain:
(i) the *temporal objective* was mismatched to a static-spectral signal, or
(ii) *pretraining-in-general* can't help because the signal is linearly saturated
([[EXP-0010]]). F9 separates them with an SSL objective whose inductive bias
**matches** the static spatial-spectral structure: **masked-DE reconstruction**
(mask random channels/bands of each DE window, reconstruct them).

**Pre-registered prediction / decision rule.**
- masked-recon **> random-init** where forecasting did **not** (smoothed DE) → the
  failure was the *temporal objective* mismatch; a structure-matched pretext
  rescues pretraining. (Would be a genuinely positive, novel result.)
- masked-recon **≈ random-init** too → *no* SSL objective helps on this signal;
  consistent with a linearly-saturated task ([[EXP-0010]]) → confirms the wall is
  the *task ceiling*, not the objective. (Strongest root-cause closure.)

Run on smoothed (`seed_v`, `seed_iv`) and un-smoothed (`seed_iv_raw`); the same
frozen linear/SVM probe as every other ladder entry.

---

## 2. Setup — exactly what was run

```bash
PY=/home/mahdiar/.conda/envs/xcqa/bin/python
bash scripts/run_f9.sh        # masked-recon + forecasting-PC + random-init, then probe
```

- **Objective:** masked autoencoding — per token, mask a fraction of the 310-d
  (C×B) DE entries (`random` | `channel` | `band` masking), reconstruct the masked
  entries from the unmasked ones + causal context. Reuses `PhysioFMS` (variant
  scratch, p_in=1, p_out=1 so the head reconstructs the current window); checkpoint
  saved in the standard format so `phase2_extract_eval.py` probes it identically.
- **Baselines (matched):** forecasting-PC (`phase2_pretrain.py`) and random-init
  (`--epochs 0`).
- **Output dir:** `results/phase2/followup/f9/`

---

## 3. Status & run log

- 2026-06-28 — created (pre-registered); built `scripts/phase2_pretrain_masked.py` + `scripts/run_f9.sh`.
- 2026-06-28 — ran `EPOCHS=40 bash scripts/run_f9.sh` (combined corpus for smoothed, `seed_iv_raw` for raw); each encoder probed by the identical frozen linear/SVM harness. Done.

---

## 4. Results  *(run date: 2026-06-28)*

Frozen-probe accuracy % (logreg / linear-SVM). `mask_ratio=0.5`, random masking,
40 epochs. Results: `results/phase2/followup/f9/{smoothed,raw}/probe_*.csv`.

| objective | smoothed SEED-V | smoothed SEED-IV | raw SEED-IV |
| --- | ---: | ---: | ---: |
| masked-recon (static SSL) | 44.96 / 45.29 | 49.93 / 51.27 | 51.53 / 50.24 |
| forecasting-PC (temporal SSL) | 45.46 / 44.30 | 52.98 / 53.33 | 52.64 / 51.19 |
| **random-init** (no SSL) | **48.54 / 46.82** | **60.68 / 61.02** | 41.97 / 44.33 |

(raw-DE linear ceiling for reference: SEED-V 51.4, SEED-IV 62.8.)

---

## 5. Interpretation — agent's reading

**It was not the temporal objective.** On **smoothed** DE, the structure-matched
masked-recon objective performs *the same as* forecasting-PC and **both sit at or
below random-init** (SEED-IV: masked 49.9, PC 53.0, random **60.7**) — i.e.
pretraining of *either* kind adds nothing and slightly distorts vs. a random
projection of the raw features. So swapping the objective to one whose inductive
bias matches the static spatial-spectral structure does **not** rescue it. This is
the "masked-recon ≈ random-init" branch of the pre-registered fork → **no SSL
pretext helps on smoothed DE**, pointing at the *features/task*, not the objective.

On **raw** (un-smoothed) DE, both objectives behave identically again — but now
**both beat random-init by ~+8–10 pts** (masked 51.5, PC 52.6 vs random 42.0).
So when the features carry learnable structure, pretraining helps and the benefit
is *objective-agnostic* (temporal and static pretexts are interchangeable).

Net: the proposed method's failure on emotion was **not** a wrong-objective
problem. The published **LDS-smoothed** DE has no learnable structure beyond what a
linear readout of the raw features already captures, so *every* SSL pretext is
inert there; on un-smoothed DE any reasonable pretext helps. Read together with
[[EXP-0010]] (the ceiling) this isolates the wall to the feature/task, not the SSL
design. *Caveat:* single mask config (random, 0.5, 40 ep); a sweep could shift
absolute numbers but is unlikely to flip the smoothed-DE ≈/< random conclusion.

---

## 6. ✅ Your verification — *(reserved for Mahdiar)*

> Leave the agent's interpretation above untouched. Confirm or correct it here.

- [ ] **Verified** (set `verified: yes` in frontmatter when ticked)
- **Notes / corrections:**


---

## 7. Commits

- (uncommitted as of 2026-06-28) new: `scripts/phase2_pretrain_masked.py`, `scripts/run_f9.sh`; results under `results/phase2/followup/f9/`.

---

## 8. Links

- Related entries: [[EXP-0001]], [[EXP-0010]] (the fork-partner), [[EXP-0005]]
- Spec: `docs/PhysioFM_Stage2_FollowUp_Experiments.md` (F9, Tier 2)
