# PhysioFM — complete account & paper-alignment brief

*For the 2026-08-19 supervisor meeting. Purpose: (1) explain every step of the project and its
evidence in one place; (2) agree how the paper is framed — what we claim, what we report as
negative, what we leave out. Every number traces to `docs/experiments/EXP-0001…0023` and
`results/`; the two summary documents behind this brief are `docs/FINAL_REPORT.md` (Jul 29) and
`docs/NEXT_PHASE_RESULTS.md` (Aug 19).*

---

## 0. Read this first — naming and protocol

| Term | Meaning |
|---|---|
| **DE** | differential entropy = log band-power in δ/θ/α/β/γ per channel per epoch (a hand-designed feature; 30 s × 100 Hz × 2 ch = 6 000 samples → 10 numbers on sleep). Baked into the proposal (inherited from PC-SSL). |
| **tf64** | 64 log-spaced spectral bins per channel per epoch (new, Aug 18) — DE with 13× finer resolution. |
| **raw tokens** | 200 ms of raw EEG, all channels, as one token (new, Aug 18). |
| **PhysioFM (pretrained)** | our structured-patch causal decoder transformer (TimesFM-2.5 layers, d = 256, 6 layers, 2.4 M params) **with** predictive-coding pretraining (input-space next-window MSE). Code name `physiofm_pc`. |
| **PhysioFM (latent)** | same, pretrained with the JEPA/data2vec-style *latent-target* objective (new). |
| **PhysioFM (no pretrain)** | identical architecture, random init — the control that isolates pretraining. |
| **raw features → logreg** | no model at all: DE (or tf64) straight into logistic regression — the baseline the model must beat. |
| **frozen probe** | encoder frozen, logistic regression on its states. **Inflates pretraining gains 3–12× — never a headline number.** |
| **fine-tuned** | encoder + head trained end-to-end on the task (what every published competitor does) — **the protocol we report**. |
| Splits | sleep: subject-disjoint 5-fold, 78 subjects; seizure: leave-one-patient-out, 24 patients; emotion: PC-SSL subject-dependent folds; MI: session hold-out. Seeds: pretraining seeds 42/1/2/3 where "multi-seed". |
| ⚠️ **PC ≠ PC-SSL** | PC-SSL = the prior ICASSP'26 paper we build on; "PC" in our tables = *our* pretrained model. |

---

## 1. What the proposal promised (URE/PhysioFM_Proposal.docx.pdf)

**Central question (verbatim):** *"whether a decoder-only transformer, pretrained with a predictive coding objective on structured physiological signal features, can learn transferable representations across multiple EEG tasks."*
Combine **PC-SSL** (predictive coding on DE, emotion only, one step ahead) with **TimesFM** (decoder-only, variable context, multi-horizon).

| Phase | Promise | Status |
|---|---|---|
| 1 | Fine-tune TimesFM on 310 univariate DE series; compare with PC-SSL 84.5 / 92.4 % | ✅ done — at chance (refuted, as expected); the 84–92 % target itself turned out ~80 % leakage |
| 2 | Structured (C×B) patch in the decoder; PC objective; p_out > p_in | ✅ done — patching works; multi-horizon/context help only on un-smoothed DE |
| 3 | **Joint multi-dataset pretraining** (SEED-IV/V + Sleep-EDF + CHB-MIT + BCI-IV-2a, channel-agnostic patch, +20 % synthetic ARMA) → zero-shot + fine-tuned (10/50/100 % labels) on 4 tasks vs TS-TCC/ContraWR/DeepSleepNet, EEG-GNN-SSL, BENDR/EEGNet/EEG-Conformer; attention-map interpretability | ❌ **never built**: every pretraining run is single-dataset; no cross-task zero-shot transfer; no synthetic mixing; no named-baseline table; no interpretability |
| Timeline / venue | 4 months; NeurIPS / ICML / ICASSP / JBHI | ~4 months elapsed (May–Aug); ICASSP Sep 16 is the live target |

**Honest one-liner vs the proposal:** the *architecture* half of the bet is confirmed and now stronger; the *pretraining* half — the actual proposed contribution — is falsified in this regime, with the mechanism measured; the *foundation-model* half (one jointly-pretrained model that transfers) was never tested.

---

## 2. What was built

- **Data (all local, all leakage-free splits):** SEED / SEED-IV / SEED-V emotion DE (public, LDS-smoothed; SEED-IV also un-smoothed); Sleep-EDF Cassette (153 nights, 78 subj, 195 k 30-s epochs, 2 EEG ch); CHB-MIT (24 patients, 682 files, 1.76 M 2-s epochs, 0.32 % seizure, 18 ch); BCI-IV-2a MI (9 subj, 22 ch). Plus new tf64 and raw-token archives for sleep and CHB-MIT.
- **Model:** structured token = whole (channels × features) matrix per epoch → linear embed → causal decoder (RoPE, SDPA) → predict the next 16 tokens (pretraining) / per-epoch classifier (downstream). No per-series instance norm (the Phase-1 lesson): fixed corpus standardisation.
- **One frozen evaluation harness** for everything; balanced logistic regression; paired per-fold / per-patient tests; multi-seed for the headline comparisons.
- **Lab notebook:** 23 experiments (`docs/experiments/EXP-0001…0023`), each with pre-registered hypothesis, exact commands, dated results, interpretation, and a verification slot reserved for you.

---

## 3. The story, step by step (each row = "did X → found Y → therefore Z")

| # | What we did | What we found | Therefore |
|---|---|---|---|
| 1 | **Phase 1**: each channel-band DE trace → TimesFM as a univariate series | **chance (20–28 %)** — RevIN strips absolute band power; flattening kills channel identity | redesign the input |
| 2 | **Phase 2**: token = full 62×5 DE matrix, no instance norm | **chance → 46–61 %** zero-shot on emotion | structured patching works (strongest, best-supported claim) |
| 3 | pretrained vs *identical random-init* on emotion | **PC ≈ random ≈ raw-DE logreg** (~51 % SEED-V / 63 % SEED-IV) | ablate everything |
| 4 | ablations: MLP head (F4), frozen random vs TimesFM stack (F3), masked objective (F9), nonlinear vs linear on DE (F10) | all null; **DE→emotion is linearly saturated** | the limiter is the DE feature, not the method |
| 5 | **un-smoothed DE** (F1; SEED-IV only) | LDS leaves 0.08 % within-trial variance; un-smoothed 17.6 % → PC − rand flips to **+11** (frozen); grows with context, scale-stable, dies when time is shuffled | "PC helps ∝ temporal dynamics" (turned out too simple) |
| 6 | **audit the PC-SSL SOTA** (F12) | 84–92 % is ~80 % temporal-neighbour leakage; same code, clean split → **40–45 %** (a replication of Brookshire et al. 2024, not a first) | the yardstick was inflated 2× |
| 7 | **sleep staging** (F13, pre-registered, full corpus) | frozen: PC 73.0 · raw 67.9 · rand 58.5 (+14.5, p<1e-4) | keystone positive (frozen) |
| 8 | **motor imagery** (F16) / **seizure** (F17) | MI null (41.7 vs 43.5, raw 51.1); seizure frozen +8.1 bal-acc / +0.08 AUC over rand, ties raw-DE at full labels | "graded spectrum": sleep +14.5, emo-raw +11, seizure +8.1, emo-smooth +2.4, MI −1.8 (frozen) |
| 9 | **protocol parity** (F16-parity): same pretraining recipe on all tasks; τ = k-step predictability | data-only predictability does *not* predict the gain (MI has the highest τ, no gain); shuffle control valid only for per-epoch labels | the ICML "temporal learnability" reframing lost its premise |
| 10 | **Control A** — dimension-matched (raw DE → random 256-d projection) | pretrained ≈ random projection on emotion (−0.2); MI −9.3; sleep +3.3 | on emotion the encoder adds nothing beyond the current window |
| 11 | **Control B — fine-tuning** (what competitors do) | sleep +9.8 → **+2.2**; seizure +8.1 → **≈ 0 (3 seeds)**; random-init gains +10–13 from fine-tuning, PC +3 | **frozen probes inflate SSL gains ~5×** |
| 12 | **Control C** — label efficiency under fine-tuning | frozen gap widens 9.8 → 12.7 at 1 % labels; fine-tuned it is **flat** (+1.9…+2.4) | the "20×/100× label efficiency" claim is a frozen-probe artefact — dropped |
| 13 | **root cause** — is the pretext learned? | yes: 40–65 % below persistence on 4/5 datasets — but **pretext skill anti-correlates with transfer** (best forecaster = MI = worst transfer) | proposed mechanism: "what is predictable ≠ what is discriminative" |
| 14 | **falsifiable test — seizure prediction** (downstream task *is* forecasting) | patient-specific: raw 0.710 · PC 0.769 · **rand 0.793 AUC** — random-init wins | mechanism's prediction **failed**; simplest reading: a random-init structured transformer is already a strong encoder |
| 15 | **Final report (Jul 29)** | architecture works; pretraining ≈ +2 on sleep, ≈ 0 elsewhere; label-efficiency retracted; below SOTA everywhere | plan: replace DE, keep the architecture |
| 16 | **Next-phase plan executed (Aug 18–19)**: Gate 0 tf64 · Gate 1 latent targets · Gate 2 raw tokens + per-electrode · Gate 3 streaming | see §4 | neither fix rescues pretraining; architecture + streaming claims get stronger |

---

## 4. The numbers that matter (fine-tuned unless stated; sleep = accuracy %, seizure = bal-acc / AUC)

### 4a. Sleep-EDF-78, subject-disjoint

| input | PhysioFM (pretrained) | PhysioFM (no pretrain) | pretraining Δ | raw features → logreg | best non-deep baseline |
|---|---:|---:|---:|---:|---:|
| DE (10-d) — 4 seeds | **75.5 ± 0.3** (κ .672) | 73.1 ± 0.5 | **+2.5** | 67.9 | HGB 69.6 |
| tf64 (128-d) — 4 seeds | **77.9 ± 0.4** (κ .71) | 77.0 ± 0.2 | **+0.85** | 72.8 | HGB 75.2 |
| raw 200 ms tokens — 1 seed, 10+3 ep | 75.5 (κ .676) | 74.2 | +1.2 (frozen: +14.7!) | — | — |
| per-electrode raw tokens — 1 seed | 76.4 | 74.6 | +1.8 | — | — |
| latent-target objective (DE, 4 seeds) | 73.7 ± 0.5 | — | +0.6 (i.e. **−1.9 vs input-PC**) | — | — |
| streaming (Gate 3): causal vs bidirectional twin, random-init | causal **73.2 online = offline** | bidir 74.1 offline → **70.4 online** | causal +2.8 online (+5.0 with PC) | — | — |
| published SOTA (raw-signal, unverified web figures) | 81–85 (κ .77–.83) | | | | |

### 4b. CHB-MIT seizure detection, LOPO 24 patients

| arm | frozen | fine-tuned |
|---|---:|---:|
| raw-DE → logreg | 72.4 / .806 | — (HGB: 74.0 / .851) |
| PhysioFM (pretrained) | 75.5–77.4 / .82–.85 | 78.4 / .863 (Jul 3-seed 78.2) |
| PhysioFM (no pretrain) | 67.5 / .741 | **80.2 / .874** (Jul 3-seed 79.1) |
| PhysioFM (latent) | 71.4 / .777 | 79.7 / .879 |
| tf64 | linear 72.4 / .809 (= DE); no headroom | ladder skipped |
| SOTA (cross-patient, unverified) | AUC 0.91–0.99 | |

→ pretraining Δ ≈ 0 (per-patient sd ±12–16); the architecture beats raw-DE by ~+6 bal-acc / +0.06 AUC.

### 4c. Everything else (frozen; never fine-tuned)

| task | PhysioFM (pretrained) | no pretrain | raw features | note |
|---|---:|---:|---:|---|
| Emotion SEED-IV, smoothed | 59.3 | 56.9 | **62.8** | 3 seeds; = random 256-d projection (dim-matched) |
| Emotion SEED-IV, un-smoothed | 51.7 | 40.7 | **55.3** | 3 seeds |
| Motor imagery BCI-IV-2a | 41.7 | 43.5 | **51.1** | 3 seeds |
| Seizure *prediction* (patient-specific) | 66.8 / .769 | **71.3 / .793** | 65.2 / .710 | 1 seed; the falsification test |
| Leakage audit (PC-SSL code) | leaky 65.8 / 70.4 → clean **39.8 / 44.7** (SEED-V / SEED-IV) | | | published 92.4 / 84.5 |

### 4d. Mechanism diagnostics (why pretraining does not transfer)

- Pretext IS learned in input space (model/persistence 0.35–0.60) but **skill anti-correlates with transfer**.
- Latent-target objective: on the long per-epoch corpora the predictor **never beats copy-last / AR-shrinkage** (skill −0.2 sleep, −0.6 seizure; collapses without time-normalised targets); where it is learned (emotion, MI, raw: skill 0.4–0.6) the frozen gain is negative or ~0.
- Dimension-matched control: emotion −0.2, MI −9.3, sleep +3.3 (DE), +3.5 (tf64).
- Frozen → fine-tuned inflation: DE +9.8 → +2.5; tf64 +11.2 → +0.85; raw +14.7 → +1.2; seizure +8.1 → ≈ 0.

---

## 5. Claims, graded by evidence

| claim | evidence | strength | keep in paper? |
|---|---|---|---|
| Structured (C×features) patching + no instance-norm recovers what univariate TimesFM destroys | chance → 46–61 %; instance-norm control collapses raw DE to chance | multi-dataset, decisive | ✅ core |
| A 2.4 M-param structured causal transformer with tf64 tokens reaches 77.9 % / κ .71 on Sleep-EDF-78 (subject-disjoint), 78–80 bal-acc / .87 AUC on CHB-MIT LOPO | 4 seeds (sleep) / 3 seeds (seizure) fine-tuned | strong | ✅ core |
| Causal design has a real streaming advantage (+2.8/+5.0 online, 1/190 compute) | 5 folds, 1 seed, DE tokens | good, single setting | ✅ (say single-seed) |
| Predictive-coding pretraining is worth ~+2.5 (DE) / +0.85 (tf64) / +1.2 (raw) on sleep and ≈ 0 on seizure/emotion/MI under fine-tuning | 4 seeds sleep, 3 seeds seizure | strong **negative** | ✅ report plainly (as the ablation) |
| Frozen-probe evaluation inflates SSL gains 3–12× | every representation | strong | ✅ methodological finding |
| Latent-target / raw-EEG / richer-spectral variants do not rescue pretraining | 4 seeds (latent DE, tf64), 1 seed (raw) | good | ✅ (short) |
| Published PC-SSL emotion SOTA is ~80 % leakage | same code, clean split | strong; **a replication of Brookshire 2024** | ✅ as one paragraph, cite Brookshire |
| Emotion / MI: pretrained ≈ random projection; below raw features | dim-matched, 3 seeds, frozen only | good | ✅ as scope/failure cases (or appendix) |
| "PC helps ∝ temporal structure" / "label efficiency 20–100×" / "objective-misalignment mechanism" | frozen-only or falsified | **retracted** | ❌ do not claim |
| Structured spatial patching > per-electrode decomposition | 2-ch sleep says the opposite (76.4 vs 75.5) | not shown | ❌ do not claim (needs many-channel corpus) |
| Beats published SOTA | no (sleep −3 to −7; seizure −0.05 to −0.12 AUC; SOTA figures unverified) | — | ❌; put an honest positioning table (verify numbers first) |
| A cross-task foundation model (joint pretraining, zero-shot transfer) | never built | — | ❌ (state as future work / limitation) |

---

## 6. Paper-framing options (the decision I need from you)

**Option A — "architecture" paper (recommended for ICASSP, 4 pages).**
*A compact structured-token causal transformer for EEG: spectral tokens, streaming inference, and what predictive-coding pretraining does (not) add.*
- Positive core: structured patching (chance → 60 %), Sleep-EDF-78 77.9 % / κ .71 and CHB-MIT LOPO 80 / .87 with 2.4 M params, causal streaming advantage (+2.8/+5.0 online at 1/190 compute), tf64 > DE.
- Honest ablation section: pretraining Δ (+0.85…+2.5 sleep, ≈ 0 elsewhere), fine-tuned, multi-seed; one line on latent/raw variants; frozen-probe inflation as the reason earlier SSL numbers looked big.
- Positioning table vs raw-signal SOTA (after verifying the numbers) — below SOTA, 30–100× smaller.
- Leave out or appendix: emotion/MI failure cases (or one paragraph), leakage audit (one paragraph, cite Brookshire), the retracted claims, the mechanism story.
- Risk: reviewers ask "why pretrain at all?" — answer is in the ablation; and "why not compare with LaBraM/CBraMod?" — 100× larger, raw-signal, different regime; state it.

**Option B — "diagnostic / methodology" paper.**
*When does forecasting-SSL help EEG? Frozen probes inflate gains 3–12×, pretext skill anti-correlates with transfer, and neither latent targets nor raw tokens fix it.*
- Strongest scientific content, genuinely new controls; but you said you cannot present a purely negative paper, and it is a harder sell at ICASSP.

**Option C — hybrid (A's positive core + B's controls as the "why we don't claim SSL" section).** This is what §5 above already is; it fits ICASSP's 4 pages if the controls are one table + one figure.

**Venue/timing (from the July analysis):** ICASSP 2027 (Toronto) — deadline **Sep 16** (professor's pick, 4 weeks; the work exists, only writing + SOTA verification remain); ML4H 2026 Sep 10; ICLR 2027 Sep 24 (needs the foundation-model story we do not have); AISTATS ~Oct 10; ICML Jan 2027 (the "temporal learnability" theory route lost its premise — MI has the highest predictability and no gain).

---

## 7. What to mention / what not to mention (proposed)

**Mention (with the number and the seed count):** fine-tuned results only as headlines; multi-seed means ± sd; the random-init control in every table; the dimension-matched control; frozen-probe inflation (once, as a methodological point); pretraining as a small positive on sleep and null elsewhere; tf64 > DE; streaming result (single seed, DE tokens); leakage audit as a replication; limitations: single-dataset pretraining, no joint/zero-shot transfer, 2-ch sleep for the raw ablation, seizure raw not run, SOTA figures to be verified.

**Do not mention as claims:** frozen-probe gains (+9.8/+14.5) except as the inflation example; "20×/100× label efficiency"; "PC helps in proportion to temporal structure"; the objective-misalignment mechanism as established (its prediction failed); structured > per-electrode; "foundation model"; SOTA parity; seizure prediction beyond "a falsification test, single seed, patient-specific".

**Ambiguous — your call:** whether emotion/MI stay in the main text as "where the encoder does not help" (they strengthen honesty, cost space); whether the leakage audit gets a table or a sentence; whether Gate 3 is a headline figure or a paragraph.

---

## 8. Decisions / questions for the meeting

1. **Framing:** A (architecture + honest ablation), B (diagnostic), or C (hybrid)? My recommendation: A/C.
2. **Venue and clock:** commit to ICASSP Sep 16? Internship end date vs writing time.
3. **What is "the contribution" sentence** we agree on? Candidate: *structured spectral tokens + causal decoder give a compact, streaming-capable EEG encoder that reaches 77.9 %/κ .71 on Sleep-EDF-78 and .87 AUC cross-patient on CHB-MIT; predictive-coding pretraining, evaluated fairly, adds ≤ 2.5 points — most reported SSL benefit in this setting is frozen-probe inflation.*
4. **SOTA table:** who verifies the published numbers (protocol, subset, #classes) — needed before any positioning claim.
5. **Remaining experiments worth 1–3 days each (do any before submission?):** (a) raw-token multi-seed + longer schedule; (b) seizure raw / many-channel per-electrode ablation (the only way to test "structured spatial patching"); (c) **joint multi-dataset pretraining + cross-task zero-shot** — the proposal's Phase-3 centrepiece, still untested (channel-agnostic embedder needed); (d) Gate 3 with tf64 tokens and 3 seeds; (e) latency benchmark with a KV cache for the streaming claim.
6. **Emotion/MI:** main text, appendix, or out?
7. **How to present the negative:** as an ablation inside a positive paper (A) vs as a finding (B).
8. **Naming:** "PhysioFM" implies a foundation model — keep, or rename to what it is (e.g. a structured causal EEG transformer)?
9. **Journal extension:** if ICASSP, what would the +30 % be (joint pretraining? more tasks?).

---

## 9. Appendix — where everything lives

- Curated docs: `FINAL_REPORT.md` (Jul 29 verdict + Aug 19 addendum), `NEXT_PHASE_RESULTS.md` (Gates 0–3), `RESULTS_POSITION.md` (SOTA scorecard, partly stale: label-efficiency §), `PAPER_RESULTS.md` (Jul 15 draft — superseded), `NEXT_PHASE_PLAN.md`, `ICML_proposal.md` (reframing idea, premise now weakened), `PHASE1.md`, `PHASE2.md`, `PHASE2_FOLLOWUP.md`, `STRATEGY.md`, `PROGRESS_REPORT.md`.
- Lab notebook: `docs/experiments/EXP-0001…0023` (§6 of every entry is your verification slot — none ticked yet).
- Results: `results/phase2`, `results/phase3`, `results/phase4/{gate0,gate1,gate2,gate3}`; figures `results/figures`.
- Code: `physiofm/` (de, spectral, raw_eeg, structured_data, physiofm_s, phase2_eval, sleep_edf, chbmit, bci_iv_2a), `scripts/` (build_*, phase2_*, gate*, diagnose_*, run_*.sh).
- Compute used: local H100-20C; RunPod H100 80 GB (July runs; Aug 18–19 raw-EEG stage). Timeline: May 8 (Phase 1) → Jul 29 (final report) → Aug 18–19 (next-phase plan).
