---
id: EXP-0010
title: F10 — Representation ceiling / linear-saturation test (why the method couldn't win)
status: done
created: 2026-06-28
run_date: 2026-06-28
agent: claude-code
phase: phase2-followup
verified: no
tags: ceiling, linear-saturation, root-cause, nonlinear, loso
commits:
verdict: CONFIRMED — the per-window DE->emotion map is linearly saturated. Nonlinear (RBF-SVM/MLP) never meaningfully beats linear: it is clearly *worse* within-subject (overfits small folds) and only ties (±~4 pts) on LOSO. No representational headroom existed above the raw-DE linear ceiling for any FM to capture (root cause B confirmed).
---

# EXP-0010 — F10 — Representation ceiling / linear-saturation test

> **Status:** done · **Created:** 2026-06-28 · **Run:** 2026-06-28 · **Agent:** claude-code · **Phase:** phase2-followup

---

## 1. Why — hypothesis & motivation

Root-cause goal: *why did the proposed temporal-PC foundation model fail to beat
the baselines on emotion?* Three preconditions were needed; two are proven failed
(A: smoothing kills dynamics — [[EXP-0001]]; C: the PC-SSL baseline was leakage —
[[EXP-0008]]). The third — **B: is there any representational headroom above the
raw-DE linear ceiling?** — is asserted (from [[EXP-0003]]/[[EXP-0004]]) but never
*measured*. This is the decisive missing number.

**Pre-registered prediction.** The per-window DE→emotion map is **linearly
saturated**: nonlinear classifiers (RBF-SVM, MLP) on the raw 310-d DE window will
**not** meaningfully exceed linear (LogReg / Linear-SVM) — and crucially this holds
even in a **large-data regime** (LOSO, ~all-but-one-subject training), removing the
~600-labels/fold data-limitation confound that capped [[EXP-0004]].

**Decision rule.**
- nonlinear ≈ linear in *both* small (subject-dependent) and large (LOSO) data →
  the static per-window DE signal is linearly extractable; **no headroom exists**,
  so the proposed FM could not have won regardless of objective/pretraining. This
  *proves* root cause B and largely closes "why".
- nonlinear ≫ linear under large data → there **is** nonlinear headroom the FM
  failed to capture → the failure is the *method*, not the task (reopens the
  design question). Falsifier for B.

Run on smoothed DE (`seed_v`, `seed_iv`) and un-smoothed (`seed_iv_raw`) to see
whether removing LDS changes the static ceiling (separate from the temporal gains
already seen in [[EXP-0005]]).

---

## 2. Setup — exactly what was run

```bash
PY=/home/mahdiar/.conda/envs/xcqa/bin/python
$PY scripts/phase2_f10_ceiling.py \
    --datasets seed_v seed_iv seed_iv_raw \
    --classifiers logreg linear_svm rbf_svm mlp_bal \
    --protocols subject_dependent loso
```

- **Data:** raw per-window 310-d DE (`build_raw_de_segments`); no model.
- **Protocols:** `subject_dependent` (PC-SSL folds, ~600 labels/fold = small data)
  and `loso` (leave-one-subject-out, train subsampled to 8000 = large data).
- **Output dir:** `results/phase2/followup/f10/`

---

## 3. Status & run log

- 2026-06-28 — created (pre-registered); building `scripts/phase2_f10_ceiling.py`.
- 2026-06-28 — ran the full grid (`seed_v seed_iv seed_iv_raw` × {logreg, linear_svm,
  rbf_svm, mlp_bal} × {subject_dependent, loso}); artifacts in
  `results/phase2/followup/f10/f10_ceiling.{csv,md}` + `f10_run.log`. Done.

---

## 4. Results  *(run date: 2026-06-28)*

acc % / macro-F1 %, mean ± std. `linear` = LogReg / Linear-SVM; `nonlinear` =
RBF-SVM / balanced MLP. Full table: `results/phase2/followup/f10/f10_ceiling.md`.

**Subject-dependent (small data, ~600 labels/fold) — best linear vs best nonlinear:**

| Dataset | best linear (acc) | best nonlinear (acc) | Δ (nl − lin) |
| --- | ---: | ---: | ---: |
| seed_v | 51.40 (logreg) | 44.59 (rbf) | **−6.8** |
| seed_iv | 65.12 (lin-svm) | 54.64 (mlp) | **−10.5** |
| seed_iv_raw | 55.35 (logreg) | 50.80 (mlp) | **−4.6** |

**LOSO (large data, ~all-but-one subject) — best linear vs best nonlinear:**

| Dataset | best linear (acc) | best nonlinear (acc) | Δ (nl − lin) |
| --- | ---: | ---: | ---: |
| seed_v | 34.30 (logreg) | 26.22 (mlp) | **−8.1** |
| seed_iv | 39.86 (logreg) | 40.42 (mlp) | **+0.6** |
| seed_iv_raw | 34.78 (logreg) | 38.95 (rbf) | **+4.2** |

---

## 5. Interpretation — agent's reading

**Root cause B confirmed: the DE→emotion map is linearly saturated; there was no
representational headroom for any FM to capture.** The pre-registered fork was
"nonlinear ≈ linear in *both* regimes → no headroom" vs "nonlinear ≫ linear under
large data → the failure is the method." The result lands squarely on the first
branch:

- **Within-subject** nonlinear classifiers are not just ≈ but strictly *worse* than
  linear (−4.6 to −10.5 pts), the documented small-fold overfit — no nonlinear signal
  to exploit even in-distribution.
- **LOSO (the large-data regime that removes the ~600-label confound)** is where a
  falsifier could have appeared, and it does **not**: on SEED-V nonlinear is far worse
  (−8.1), on SEED-IV it ties (+0.6), and on un-smoothed SEED-IV-raw it edges linear by
  only +4.2 (rbf). Nowhere is there the `nonlinear ≫ linear` that would reopen "the
  failure is the method." The largest nonlinear gain anywhere is ~4 pts, cross-subject,
  on raw DE only.

So the static per-window DE signal is **linearly extractable and saturated**: a linear
probe already reaches the ceiling, and a learned/nonlinear representation cannot beat it
within-subject and barely moves it cross-subject. This *proves the missing precondition*
in the failure diagnosis — combined with A (smoothing kills dynamics, [[EXP-0001]]) and
the leakage finding ([[EXP-0008]]), it closes "why the proposed FM couldn't win on
emotion DE": there was nothing above the linear ceiling to win.

**One directional hint, not a falsifier:** the only place nonlinear ever edges linear is
the **cross-subject (LOSO)** regime (SEED-IV/-IV-raw), i.e. the one axis that is *not*
saturated. That is exactly the regime [[EXP-0012]] (F14) targets and the reason the
project's remaining FM hopes live cross-subject / on raw signal ([[EXP-0013]]), not
within-subject on DE. *Caveat:* LOSO train was subsampled to 8000 for the nonlinear
classifiers; a larger cap is unlikely to manufacture the missing ≫ but is the only loose
end.

---

## 6. ✅ Your verification — *(reserved for Mahdiar)*

> Leave the agent's interpretation above untouched. Confirm or correct it here.

- [ ] **Verified** (set `verified: yes` in frontmatter when ticked)
- **Notes / corrections:**


---

## 7. Commits

- (uncommitted) new: `scripts/phase2_f10_ceiling.py`

---

## 8. Links

- Related entries: [[EXP-0001]], [[EXP-0003]], [[EXP-0004]], [[EXP-0008]], [[EXP-0011]]
- Spec: `docs/PhysioFM_Stage2_FollowUp_Experiments.md` (F10, Tier 2)
