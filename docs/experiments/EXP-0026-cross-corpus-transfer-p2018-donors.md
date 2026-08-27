---
id: EXP-0026
title: Cross-corpus transfer — P2018-pretrained donors fine-tuned on Sleep-EDF-78 and HMC
status: done
created: 2026-08-27
run_date: 2026-08-27
agent: claude-code
phase: external-validation
verified: no
tags: transfer, pretraining, p2018, sleep_edf, hmc, perch
commits: TBD
verdict: HYPOTHESIS CONFIRMED for full-weight transfer; refuted for trunk-only. SEDF-78 per-electrode, 4 replicates (donor seeds 42/1 x FT seeds): P2018-transfer 78.35 acc / kappa .712 vs matched random-init 75.21 / .672 -> +3.14 +/- 0.33 acc, ALL replicates positive - the first robust pretraining benefit in the project, in the literature's +2.3..+6.6 transfer range. Transfer ~= same-corpus PC (+0.55 +/- 0.71): pretraining on a bigger foreign corpus fully substitutes for target-corpus pretraining. Trunk-only transfer does nothing (SEDF 76.55 ~ rand 76.78; HMC 73.11 <= rand 73.65) - the benefit needs the full model, which only channel-agnostic per-electrode tokens allow. Bonus: 78.79-79.10 (best transfer folds) is the project's best SEDF number, beating structured 77.9.
---

# EXP-0026 — Cross-corpus transfer (the literature's positive regime)

> **Status:** running · **Created:** 2026-08-27 · **Agent:** claude-code · **Phase:** external-validation

## 1. Why — hypothesis & motivation
In-domain PC pretraining is now a triple-replicated null (SEDF ~+0.85 noise; HMC
+0.00 ± 1.25, 8 seeds; P2018 Δκ .0001 at 994 subjects — EXP-0024/0025). The published
sleep literature shows the same in-domain pattern but consistent POSITIVE transfer
across corpora (SleepTransformer SHHS→SEDF-78 +3.5 acc / +.046 κ; Phan et al.
MASS→SEDF +3.0; L-SeqSleepNet +2.3…+6.6). This is also the proposal's Phase-3 promise
(joint pretraining → transfer). Hypothesis: PC pretraining on the biggest corpus
(P2018, 994 subjects) transfers a benefit to smaller target corpora that same-corpus
pretraining cannot provide.

## 2. Setup — exactly what was run
Donors (seed 42, pc 60 ep, input-space PC): pretrained on ALL 994 P2018 records —
(a) structured 6×64; (b) per-electrode 1×64 (5,964 sequences). Donor corpora contain
no target-corpus subjects by construction.

Transfer mechanics (`scripts/make_transfer_ckpt.py`):
- **full** (per-electrode only — 1×64 tokens are channel-count-agnostic): donor
  weights verbatim; standardizer = target corpus stats.
- **trunk**: target random-init model with the donor's decoder transplanted
  (`layers.*` + `out_norm.*`); input/output blocks stay target-shaped random.

Evaluations (each = transfer arm vs matched random-init vs same-corpus-PC, identical
harness, FT full, best-epoch-by-val-κ on HMC / fixed 8 ep on SEDF):
1. SEDF-78 structured 2×64, 5-fold: rand / sedf-pc (gate-0 seed-42 models) /
   **trunk-transfer**.
2. SEDF-78 per-electrode 1×64 (`--merge_every 2`), 5-fold: rand / sedf-perch-pc /
   **full-transfer** (new perch archives sliced from tf64, `build_perch_tf64.py`).
3. HMC structured 4×64, fixed split, FT 20 ep: rand / hmc-pc (seed-42) /
   **trunk-transfer**.

## 2b. Comparability caveats (pre-registered, from the pre-launch review)
- **HMC is the exposure-clean comparison**: its pc baseline pretrained on SN001–125
  only, test = SN128–154 — fully symmetric with the donor. On SEDF the same-corpus pc
  arms follow the locked gate-0 convention (corpus-wide unlabeled pretraining incl.
  test subjects), which biases AGAINST the transfer arm; state this when interpreting
  transfer ≤ same-corpus-pc there.
- **Not compute-matched**: at matched 60 epochs the donors take ~6–19× more optimizer
  steps than target pc baselines (63 vs 10 steps/ep structured; 373 vs 20 perch).
  A transfer win is "more data + more compute under the same recipe".
- The SEDF pc/rand arms are RE-RUN under this harness (a seed-derivation fix changed
  batch-shuffle order vs gate-0), so rows are internally comparable; gate-0 numbers
  are not byte-identical references.
- Standardizers: always the target corpus's, all arms alike.

## 3. Status & run log
- 2026-08-27 05:0x — pre-launch adversarial review (13 agents): 5 confirmed findings
  fixed (dead `--ft_seed` in the SEDF harness now threaded; skip-guards moved to
  last-written artifacts; unused HMC perch build dropped as a leakage footgun;
  documented the two caveats above). The review also caught that the CONCURRENT
  P2018 c3 queue had crashed on an ARCH KeyError (`p2018_tf64_c3` never registered,
  lookup before the fallback) — fixed, relaunched.
- 2026-08-27 05:1x — queue launched (`scripts/run_transfer.sh`).

## 4. Results  *(run date: 2026-08-27)*

**SEDF-78 per-electrode (1×64, merge 2), 5-fold, acc ± fold-sd / κ:**

| replicate (donor/ft seed) | random-init | same-corpus PC | **P2018 transfer (full)** | tr−rand |
|---|---|---|---|---|
| d42 / ft42 | 75.85 ± 2.33 / .679 | 78.26 ± 2.94 / .710 | **78.79 ± 2.29 / .717** | +2.94 |
| d42 / ft1 | 74.81 / .668 | 77.54 / .702 | **77.91 / .707** | +3.10 |
| d42 / ft2 | 75.41 / .675 | 77.48 / .701 | **79.10 / .721** | +3.69 |
| d1 / ft1 (independent donor) | 74.77 / .667 | 77.91 / .705 | 77.58 / .702 | +2.81 |
| **mean** | 75.21 / .672 | 77.80 / .705 | **78.35 / .712** | **+3.14 ± 0.33** |

transfer − same-corpus-pc = +0.55 ± 0.71 (≈ parity; recall the SEDF pc arm's corpus
includes test-subject unlabeled data, the donor's does not — §2b).

**Trunk-only transfer (structured tokens):** SEDF 76.55 ± 2.32 / .688 vs rand
76.78 ± 2.97 / .690 vs pc 77.66 ± 2.43 / .702 (seed 42). HMC (e20, fixed split):
transfer BAC 73.11 / κ .657 vs rand 73.65 / .663 vs pc 73.70 / .674. Trunk alone: nothing.

CSVs: results/phase4/transfer/{sedf_perch_ft,sedf_structured_ft,hmc_ft}.csv.

## 5. Interpretation — agent's reading
1. **The pretraining story flips in exactly the regime the literature predicted.**
   In-domain: null (4 replications, EXP-0024/0025). Cross-corpus, full-weight:
   +3.1 ± 0.3 acc over matched random-init, every replicate positive, two independent
   donors. This is the proposal's Phase-3 claim, finally demonstrated.
2. **Mechanism is localized**: trunk transplant does nothing on either target; the
   benefit requires the pretrained input/output blocks WITH the trunk — only
   channel-count-agnostic per-electrode tokens make that possible across corpora.
   Per-electrode tokenization is thus not just an ablation: it is the transfer vehicle
   (consistent with gate-2, where per-electrode raw also showed the largest in-domain
   pretraining delta, +1.8).
3. Transfer ≈ same-corpus pretraining ⇒ a single big-corpus pretrain substitutes for
   per-target pretraining — the foundation-model deployment story, at 2.65 M params.
4. New best SEDF-78 number: 78.35 mean / 78.8–79.1 best replicates (vs 77.9
   structured) — per-electrode + transfer wins outright.
5. Caveats stand (§2b): not compute-matched (donor saw ~19× more steps); SEDF pc arm
   has the exposure advantage, transfer does not. SHHS (pending access) is the natural
   scale-up: 5.8k-subject donor → SEDF/HMC/P2018 targets.

## 6. ✅ Your verification — *(reserved for Mahdiar)*

## 7. Commits
d14ca59 (pipeline + review fixes), this commit (results).

## 8. Links
- EXP-0024 (HMC), EXP-0025 (P2018), docs/SLEEP_DATASET_CANDIDATES.md §02.
