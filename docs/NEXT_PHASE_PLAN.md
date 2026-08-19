# Making the contribution work: replacing DE, keeping the architecture

*Constraint: keep the decoder-only causal transformer (TimesFM heritage) + the predictive-coding
objective (PC-SSL heritage). That pairing IS the contribution. Only the input representation and
the prediction target are on the table.*

Written 2026-07-29, after the root-cause diagnostic ([[EXP-0017]]).

> **OUTCOME (executed 2026-08-18/19 — see `docs/NEXT_PHASE_RESULTS.md`, [[EXP-0020]]–[[EXP-0023]]).**
> Gate 0: tf64 fails the headroom rule (sleep +2.3 vs DE +1.7; seizure none) but lifts the fine-tuned
> architecture to 77.9 % / κ .71 on sleep while pretraining shrinks to +0.85. Gate 1: latent targets do
> **not** beat input-space PC (sleep 73.7 vs 75.5 over 4 seeds; seizure indistinguishable) — five variants,
> all at random-init level; the latent objective degenerates into smoothness on the per-epoch corpora.
> Gate 2: raw 200 ms tokens reach parity with DE (75.5), pretraining +1.2 fine-tuned (+14.7 frozen);
> per-electrode ≥ structured on 2-ch sleep. Gate 3: **confirmed** — causal beats its bidirectional twin
> online by +2.8 / +5.0 at 1/190 the compute. Net: R1/R2 fixes do not rescue predictive-coding
> pretraining; the architecture + streaming claims and the frozen-probe-inflation finding get stronger.

---

## 1. What the diagnosis says any fix must satisfy

Two hard requirements, both derived from measurements we made, not intuition:

**R1 — The discriminative information must NOT be linearly accessible from the input.**
F10 showed the DE→label map is linearly saturated, and the downstream probe sees the whole DE
vector. So any encoder trained on DE can only *reorganise* what the probe already has. DE
pre-solves the problem, leaving SSL no headroom. A replacement representation must contain the
class information in a form a linear model cannot read off directly.

**R2 — The prediction target must not be the raw input.**
Our anti-correlation result: the better the model forecasts DE, the worse it transfers (MI best
forecaster / worst transfer; sleep worst forecaster / best transfer). Forecasting in *input
space* rewards modelling the smooth autocorrelated component, which is not the discriminative
component. The fix is to predict in **latent space** — where the target is learned, so the model
is free to represent what matters instead of what is easy to predict.

R1 kills DE. R2 kills input-space MSE forecasting. Both are fixable without touching the
architecture.

---

## 2. Candidate replacements for DE

| Option | Satisfies R1? | Preserves our patching? | Risk |
| --- | --- | --- | --- |
| **A. Raw EEG, structured multi-channel patches** (token = all C channels × ~200 ms) | ✅ morphology, phase, spindles/K-complexes/spike-waves are not linearly readable | ✅ same structured-patch idea, applied to raw signal | forecasting raw EEG in input space is near-impossible (low SNR) → **requires R2** |
| **B. Rich time–frequency** (STFT/multitaper, 64–128 bins instead of 5 bands) | ⚠️ partially — far less compressed than DE, but still no phase/morphology | ✅ direct drop-in for the `(C×B)` patch | may inherit DE's saturation at higher resolution; cheapest to test |
| **C. Learned VQ tokenizer** (LaBraM-style neural codes) | ✅ | ⚠️ adds a tokenizer stage | LaBraM owns this; heavy |

**Recommendation: A, with B as the cheap pilot.** B is a two-hour change to
`physiofm/de.py` and directly tests whether "less compression" alone buys headroom — a clean
go/no-go before committing to a raw-signal pipeline.

**And in both cases, switch the objective from input-space MSE to latent prediction (JEPA-style):**
keep the causal decoder predicting the *next* patch, but score the prediction against the
(EMA/stop-gradient) embedding of the future patch rather than its raw values. This keeps
"predictive coding in a decoder-only transformer" exactly intact while removing the failure mode
we measured.

---

## 3. ⚠️ Honest novelty assessment — the space has filled up

This is the part that matters most, and it is not comfortable.

| Prior work | What it does | Overlap with our plan |
| --- | --- | --- |
| **BrainGPT / EEGPT** (arXiv 2410.19779) | **"First autoregressive EEG pre-trained model"**: raw EEG, **causal decoder-only**, **next-signal prediction with MSE**, 37.5M samples / ~1B tokens, 138 electrodes; +5–11% over specialists on 12 datasets incl. sleep, emotion, MI | **This is our Option A, already done and scaled.** Direct collision. |
| **Laya: LeJEPA for EEG** (arXiv 2603.16281) | Latent prediction over reconstruction for EEG | **This is our R2 fix, already done.** |
| **CaMBRAIN** (arXiv 2605.28792) | Causal, streaming, real-time EEG; causal predictive pretraining + latent JEPA | Takes the causal/streaming niche (with an SSM, not a transformer) |
| **LaBraM / CBraMod** | Masked modelling on raw signal, ICLR 2024/2025 | The masked branch |

**Conclusion: "decoder-only causal transformer + autoregressive prediction on raw EEG" is no
longer novel.** BrainGPT is that paper. Simply swapping DE→raw signal would reproduce published
work at 1/100th the scale.

### The one genuine gap left

BrainGPT **decomposes multi-electrode signals into individual electrode sequences** — each
electrode becomes an independent training sample. That is exactly the mistake our Phase 1
diagnosed and Phase 2 fixed: it discards spatial structure. Our *actual* contribution was never
"causal AR on EEG" — it was **structured multi-channel patching that preserves spatial/spectral
topology**, which we showed takes emotion from chance to ~60%.

So the defensible position is narrow but real:

> **Structured spatial patching + causal autoregressive pretraining + latent (non-reconstructive)
> targets** — i.e. BrainGPT's objective with our spatial structure and JEPA's target, motivated by
> a measured diagnosis of why input-space forecasting fails.

Whether that is enough for a top venue is genuinely uncertain. It is an *ablation-scale*
contribution over BrainGPT, not a new paradigm.

---

## 4. The experiment plan

### Gate 0 — cheap pilot: does less compression buy headroom? *(1–2 days, no new data)*
1. Build a **rich time–frequency** archive: same pipeline as DE but 64 log-spaced bins instead
   of 5 bands (`physiofm/de.py` → `compute_spectrogram`). Sleep + seizure first.
2. **Re-run the linear-saturation test (F10)** on it. *This is the go/no-go.* If a nonlinear
   model still cannot beat a linear one, the representation is saturated like DE and **R1 fails
   — stop, do not proceed to raw EEG with this objective.**
3. If unsaturated: run PC pretraining + the dimension-matched control (`diagnose_encoder.py`).
   Does PC now beat a random projection by more than the +3.3 we saw on sleep?

### Gate 1 — the objective fix (latent targets) *(3–5 days)*
4. Implement **latent-target prediction**: add an EMA target encoder; the causal decoder predicts
   the target encoder's embedding of the next `p_out` patches; loss = cosine/MSE in latent space,
   with stop-gradient on the target. Guard against collapse (monitor embedding variance/rank).
5. **Re-run the pretext diagnostic** (`diagnose_pretext.py` analogue in latent space) and, most
   importantly, **re-measure the anti-correlation**: does pretext skill now *correlate* with
   downstream gain instead of anti-correlating? That is the direct test of R2.
6. Compare on sleep + seizure: input-space PC vs latent PC vs random-init, **fine-tuned** (not
   frozen — we know frozen inflates by ~5×).

### Gate 2 — raw EEG *(1–2 weeks; only if Gates 0–1 pass)*
7. Raw-signal structured patches: token = all C channels × 200 ms, straight into the existing
   decoder. Sleep-EDF and CHB-MIT raw are already on disk.
8. Same ladder: latent-PC vs random-init vs raw-DE baseline vs (if feasible) a published
   checkpoint, all fine-tuned, subject/patient-disjoint.
9. **Ablation that carries the paper:** structured multi-channel patches vs BrainGPT-style
   per-electrode decomposition, everything else identical. This isolates our actual contribution.

### Gate 3 — the defensible claim *(1 week)*
10. **Streaming/causal evaluation**: latency-constrained inference (decision at time *t* using
    only data ≤ *t*), compared against bidirectional models (LaBraM/CBraMod-style) forced into
    the same constraint. This is where a causal model can legitimately win.
11. Multi-seed + paired per-subject tests throughout; report fine-tuned numbers only.

---

## 5. What to do about the ICASSP deadline (Sep 16)

**Do not attempt this plan before Sep 16.** Gates 0–2 are 3–5 weeks of work with real failure
risk at every gate. Submit the diagnostic paper we already have:

> *When does predictive-coding SSL help EEG? Pretext skill anti-correlates with transfer, and
> frozen-probe evaluation inflates reported gains ~5×.*

That contribution is **complete, measured, and not owned by anyone else** — BrainGPT/Laya/CaMBRAIN
all report gains without ever testing whether pretext skill predicts transfer, or whether their
evaluation protocol inflates it. Our anti-correlation measurement and the frozen-vs-fine-tuned
comparison are genuinely new and would apply to *their* models too.

Then run this plan for the next cycle.

---

## 6. Honest bottom line

The idea can be made to work — R1 and R2 are both fixable and the architecture survives. But
**the version that works is one BrainGPT has largely published**, and our remaining edge
(structured spatial patching + latent targets, justified by a measured diagnosis) is an
ablation-scale delta, not a new paradigm. The strongest asset this project now has is not the
model — it is the **diagnostic methodology** that revealed why the objective fails, which nobody
in this literature has applied.
