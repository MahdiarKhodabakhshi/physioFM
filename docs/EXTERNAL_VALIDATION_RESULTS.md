# External validation & cross-corpus transfer — results (2026-08-27)

One-day sequel to `docs/NEXT_PHASE_RESULTS.md`, executed after the dataset survey
(`docs/SLEEP_DATASET_CANDIDATES.md`). Notebook: EXP-0024 (HMC), EXP-0025 (P2018),
EXP-0026 (transfer). All fine-tuned, subject-disjoint, published-protocol evaluations.

## One-table summary

| Claim | Evidence | Status |
|---|---|---|
| The tf64 + causal-decoder architecture transfers across sleep corpora | HMC fixed split: BAC 73.8 / κ .668 / wF1 74.5 (8 seeds) — 2nd of 8 on BAC vs the LaBraM/CBraMod/REVE/CSBrain ladder at ~1/100th params; P2018 SleePyCo split: 76.1 acc / κ .684 (6 ch) | **Confirmed, 2 new datasets** |
| In-domain PC pretraining helps | HMC paired Δ +0.00 ± 1.25 BAC (8 seeds); P2018 Δκ ≤ .001 at 994 subjects (6 ch AND single-channel); SEDF +0.85 (4 seeds) reads as noise | **Null — 4th replication; scale ruled out as excuse** |
| Cross-corpus transfer (the literature's positive regime; proposal Phase 3) | P2018 donor → SEDF-78, full weights via per-electrode tokens: **+3.14 ± 0.33 acc over matched random-init** (4 replicates, 2 independent donors), ≈ parity with same-corpus PC; trunk-only transfer: nothing (SEDF & HMC) | **CONFIRMED — first robust pretraining benefit** |
| Best SEDF-78 number | per-electrode + P2018 transfer: 78.35 mean / 79.1 best; structured @ e16 FT: 78.4–79.0 (any head) | New project best |
| The SOTA gap is missing bidirectional context | EXP-0027: adding the SOTA models' sequence stage (bidirectional transformer head, matched window) moves ≤1 acc at e8 and ≈0 at matched budget, on all three datasets — consistent with EXP-0023's twin (+0.9 offline). Residual gap = the epoch-level feature extractor (fixed Welch tokens vs learned intra-epoch encoders) | **Refuted — strengthens the streaming claim: causality costs ≈1 point** |
| Gap to bidirectional supervised SOTA | P2018 c3 (input-identical): −6.6 acc to SleePyCo; 6-ch recovers +1.9; residual ≈ model class/context — the quantity the causal/streaming claim (Gate 3) trades against | Quantified |

## Protocol notes (for the paper's reproducibility appendix)
- HMC: NeuroLM positional split (SN001–102 / 103–127 / 128–154 = 100/25/26), no wake
  trim, all scored epochs (91,248/22,124/23,871 — matches NeuroLM Table 1 exactly);
  metrics BAC / unweighted κ / weighted-F1 pooled over test epochs; best FT epoch by
  val κ. Our input pipeline (tf64 Welch @ native 256 Hz) replaces theirs by design.
- P2018: SleePyCo's published fold file, vendored (MIT) at
  `data/physiofm/splits/p2018_sleepyco_folds.npy`; metrics pooled across the 5 folds
  (acc/MF1/unweighted κ). Label extraction verified against both distributed formats;
  892,200 scored epochs vs SleePyCo's 892,262 — the −59 traced to 3 records whose WFDB
  annotations lack initial stage marks present in the sample-wise vectors
  (tr03-0314 +12 REM, tr05-0326 +46 N2, tr07-0602 +1 N2); their residual +3 unexplained.
- Transfer caveats (pre-registered): donors not compute-matched (~6–19× more optimizer
  steps at matched 60 epochs); SEDF same-corpus-pc arms include test-subject unlabeled
  data in pretraining (locked gate-0 convention) while donors see no target subject —
  biased AGAINST the transfer arm; HMC is the exposure-clean comparison.

## Where results live
results/phase4/{hmc,p2018,transfer}/ (CSVs + models); datasets/{HMC,P2018} (16 + 134 GB,
SHA-verified); pipelines physiofm/{hmc,physio2018}.py, scripts/{build_hmc_dataset,
build_p2018_dataset,build_perch_tf64,make_transfer_ckpt,phase2_hmc_finetune,
phase2_p2018_finetune,run_hmc,run_p2018,run_transfer}*.py/sh.

## Next (queued on SHHS access — NSRR request pending)
shhs1 (5,793 nights) as donor: per-electrode pretrain → transfer to SEDF/HMC/P2018;
the SleepTransformer precedent says the transfer delta grows with donor size.
