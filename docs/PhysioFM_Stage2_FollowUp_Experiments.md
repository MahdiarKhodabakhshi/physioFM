# PhysioFM — Stage 2 Follow-Up Experiments

*The experiments a careful researcher would run before drawing final conclusions — organized by purpose: (Tier 0) de-confound the central negative result, (Tier 1) alternative configurations that could flip a verdict, (Tier 2) new objectives the results motivate, (Tier 3) the honesty keystone, (Tier 4) the Phase-3 hand-off.*

Each entry states the **hypothesis**, the **surprise / configuration / idea it addresses**, the **method**, a **decision rule** (what each outcome would mean), **cost**, and whether it is **blocking for a writeup**. The Tier-0 set is small, cheap, and — in my judgment — mandatory before any external claim.

Legend: 🔴 blocking for writeup · 🟡 high-value · 🟢 exploratory.

---

## Tier 0 — Make the central negative result bulletproof

The headline claim ("temporal PC pretraining adds nothing for emotion") currently rests on three confounds (see Interpretation §3). These four experiments remove them. They are cheap (minutes each on the existing harness) and decide whether the negative result is *fundamental* or an *artifact of setup*.

### F1 🔴 — Un-smoothed DE + persistence baseline *(the most important experiment)*

- **Hypothesis.** The forecasting pretext was near-trivial because it ran on **LDS-smoothed** DE (`*_de_LDS.npz`), where `F_{t+1} ≈ F_t`. On smoothed features, persistence is near-optimal, so pretraining *cannot* learn dynamics that were smoothed out — which is why pretrained ≈ random.
- **Addresses.** The surprise that pretraining = random init, and the leap from there to "emotion is static."
- **Method.**
  1. Build **raw (un-smoothed) per-window DE** archives (the SEED pipeline before LDS), same `time × 62 × 5` layout.
  2. Compute a **persistence baseline MSE**: predict `F_{t+1} = F_t` (and the multi-step analogue) on both smoothed and raw DE. Report alongside the model's PC-MSE.
  3. Decompose variance: per-(channel,band) **within-trial variance vs cross-trial/cross-subject variance**. This quantifies how much "predictable" signal is just the trial's static level.
  4. Re-pretrain E1a on raw DE; re-run the zero-shot probe and the random-init control.
- **Decision rule.**
  - If model PC-MSE ≈ persistence MSE on smoothed DE → the pretext was trivial; the Stage 2 null is (partly) a smoothing artifact, and the static-emotion claim must be re-tested on raw DE.
  - If on **raw** DE the pretrained model now beats random-init → temporal PC *does* help once dynamics are present; the real conclusion becomes "LDS smoothing destroys the learnable dynamics," which is a cleaner and more interesting story.
  - If pretrained still ≈ random-init on raw DE *and* persistence MSE ≫ model MSE → the static-emotion conclusion is genuinely robust; now it's bulletproof.
- **Cost.** Low (feature rebuild + one pretrain + probe). **Do this first.**

### F2 🔴 — Sequence-level / order-aware readout

- **Hypothesis.** A per-window probe with trial-constant labels cannot reward temporal modeling; a sequence-level readout can.
- **Addresses.** Confound that the evaluation is static by construction (Interpretation §3.3).
- **Method.** Keep the frozen encoder, but classify a **trial** from: (a) the model's accumulated causal hidden state at the last window; (b) an order-respecting pool (e.g. GRU/attention over the window-embedding sequence); and as a control (c) the same with **window order shuffled**. Compare pretrained vs random-init under (a)/(b), and (b) vs (c).
- **Decision rule.**
  - Pretrained > random-init under (a)/(b) but tie under per-window probe → temporal pretraining helps, but only a temporal readout exposes it. Big result.
  - (b) ≈ (c) (shuffling doesn't hurt) → confirms the signal is order-invariant/static; the negative result holds even with a fair readout.
- **Cost.** Low–medium (small readout heads; no re-pretraining needed).

### F3 🟡 — Frozen-random vs frozen-TimesFM stack (disentangle C6)

- **Hypothesis.** E1b matching E1a may be because the trained I/O blocks do the work, not because TimesFM's weights transfer.
- **Addresses.** The undecidable C6 claim (Interpretation §5).
- **Method.** Three matched runs, identical structured I/O blocks trained fresh: (1) frozen **TimesFM** stack (E1b, exists), (2) frozen **random-init** stack of the *same* d=1280/20-layer shape, (3) fully-trained from-scratch (E1a, exists). Same probe.
- **Decision rule.**
  - frozen-TimesFM > frozen-random → TimesFM priors genuinely transfer (C6 supported).
  - frozen-TimesFM ≈ frozen-random → the pretrained weights add nothing; "transfer" is really "a big fixed nonlinear mixer." C6 refuted, and it ties the whole story together (consistent with the random-init finding).
- **Cost.** Low (one extra frozen run + probe).

### F4 🔴 — Matched downstream head (fair PC-SSL-style comparison)

- **Hypothesis.** The PhysioFM-S-vs-PC-SSL and FM-vs-raw-DE comparisons are unfair because PC-SSL uses a frozen encoder + **2-layer MLP**, while PhysioFM-S uses a linear probe.
- **Addresses.** Confound C (Interpretation §3.4) and the PC-SSL gap framing (§4).
- **Method.** Apply **one identical downstream protocol** — frozen encoder + the *same* 2-hidden-layer MLP head, same regularization, same clean splits — to: raw DE, PhysioFM-S (pretrained), PhysioFM-S (random-init). Use proper per-fold validation and class weighting; report mean ± std.
- **Decision rule.**
  - If raw-DE + MLP now reaches the 80s → the PC-SSL gap is mostly the *head*, not the representation. Reframe the entire FM contribution.
  - If PhysioFM-S + MLP > raw-DE + MLP → there *is* representation value the linear probe was hiding.
  - If all three tie near the linear ceiling → confirms the head isn't the lever either; the static structure is genuinely the ceiling.
- **Cost.** Low. *(Note: the earlier "MLP overfits on raw DE" result was under a linear-probe regime and ~600 labels/fold — re-test it carefully here, because PC-SSL trains an MLP on the same-size data and does not collapse.)*

---

## Tier 1 — Alternative configurations that could change a verdict

These ask "would a different but still-reasonable setup flip the result?" Run after Tier 0; some become moot depending on F1/F2.

### F5 🟡 — Larger input context (`p_in`)

- **Hypothesis.** `p_in=1` gives the transformer no history per token; the proposal's `p_in=4` (or whole-trial context) might let attention exploit temporal context.
- **Addresses.** Whether "no temporal benefit" is partly a context-length artifact, and re-tests C4.
- **Method.** Sweep `p_in ∈ {1, 4, 8, full-trial}` × `p_out ∈ {1, 16}`, paired with the **sequence-level readout** from F2 (otherwise the extra context can't surface).
- **Decision rule.** Monotone gain with context (esp. on raw DE from F1) → dynamics matter; flat → confirms static. **Pre-register** that you expect flat on smoothed DE and possibly positive on raw DE.
- **Cost.** Medium (small grid; minutes each).

### F6 🟢 — Scale check

- **Hypothesis.** Null pretraining benefit may be a sub-million-parameter, sub-minute-training artifact; FM benefits are scale-emergent.
- **Addresses.** The scale caveat (Interpretation §3.4).
- **Method.** Two or three points on a scale ladder (params × epochs), tracking the **pretrained-minus-random gap** as the metric of interest, not absolute accuracy. Do this only on raw DE (post-F1) so there's signal to scale into.
- **Decision rule.** Gap stays ≈0 as scale grows → strong evidence the null is fundamental for this task. Gap opens → the FM bet needs scale, not abandonment.
- **Cost.** Medium (still small by FM standards; SEED is tiny).

### F7 🟡 — Supervised fine-tuning, done properly (decide C2)

- **Hypothesis.** Frozen probes underestimate the model; the proposal's fine-tuned mode (10/50/100% labels) was attempted but unstable on ~600 labels/fold and was never conclusive.
- **Addresses.** C2 (does FM beat raw-DE?) is still formally **open**; the head-only SGD probe hit 52.6% on SEED-V (just over raw DE) but only on V and optimizer-dependent.
- **Method.** Stabilize fine-tuning: low LR + layer-wise decay, strong weight decay/dropout, early stopping on a per-fold val split, class weighting; report **limited-label curves** (10/50/100%) for raw-DE-head, PhysioFM-S, random-init — all with the F4 head.
- **Decision rule.** A consistent FM > raw-DE margin that *grows as labels shrink* would be the genuine FM win (label efficiency); no margin → C2 closed negative.
- **Cost.** Medium.

### F8 🟡 — Subject normalization / domain adaptation for LOSO

- **Hypothesis.** The FM "loses" on LOSO (27.6/34.7 vs raw-DE 31.9/37.8) because nothing in the objective targets **subject-invariance**, and LOSO failure is driven by cross-subject DE distribution shift — a known SEED problem, not a model failure per se.
- **Addresses.** The claim "FM gives no cross-subject advantage" — currently true but tested with a model never asked to be invariant.
- **Method.** Add per-subject standardization / alignment (e.g. EA/Euclidean alignment, CORAL) at probe time for *all* methods; separately, try a subject-adversarial or CORAL pretraining term. Compare deltas.
- **Decision rule.** If invariance machinery helps the FM more than raw DE → this is where the FM's value actually lives, and it reframes the project around invariance rather than forecasting. If it helps all methods equally → the shift is just hard and not the model's job.
- **Cost.** Medium.

---

## Tier 2 — New objectives the results actually motivate

The Stage 2 finding (signal is static-spectral) is itself a hypothesis generator: if the discriminative content is static structure, the *objective* should match static structure, not forecasting.

### F9 🟡 — Masked-DE reconstruction instead of (or with) forecasting

- **Hypothesis.** A **masked-autoencoding** pretext over the `(C×B)` matrix (mask random channels/bands, reconstruct) matches the static-structural nature of the signal better than temporal forecasting, and may learn spatial-spectral relationships that transfer.
- **Addresses.** The new idea raised by "static, not dynamic": pick an objective whose inductive bias is spatial-spectral, not temporal.
- **Method.** Pretrain the same backbone with channel/band masking (BERT-/MAE-style) on DE; probe identically. Compare to PC and random-init.
- **Decision rule.** Masked-recon > random-init where PC failed → the problem was the *objective*, not pretraining per se. Clean, novel, and directly tied to the Stage 2 mechanism.
- **Cost.** Medium (new loss, same infra).

### F10 🟢 — Supervised-contrastive upper bound + static contrastive SSL

- **Hypothesis.** Establish the *representation* ceiling: how separable can emotion get with a structure-aware objective?
- **Method.** (a) Supervised-contrastive head as an **upper bound** on what any frozen representation could achieve here; (b) an augmentation-based contrastive SSL (channel dropout, band jitter) as a static SSL comparison to PC.
- **Decision rule.** If sup-contrastive ≈ linear ceiling → the ceiling is the data, not the method (and the whole 84–92% PC-SSL number gets even more suspicious — see F12). If sup-contrastive ≫ ceiling → there's headroom a better SSL objective might reach.
- **Cost.** Medium.

### F11 🟢 — Subject-invariant representation as the FM's thesis

- **Hypothesis.** The defensible "foundation model" value for biosignal emotion is **cross-subject generalization**, not within-subject temporal prediction.
- **Method.** Combine F8's invariance objective with F9's static pretext; evaluate primarily on LOSO and few-shot-new-subject.
- **Decision rule.** A real LOSO / few-shot-subject win would be a *positive* foundation-model result the project can build a paper around — pivoting the thesis from "temporal PC" to "structure-preserving, subject-invariant pretraining."
- **Cost.** Medium–high.

---

## Tier 3 — The honesty keystone

### F12 🔴 — Clean PC-SSL re-replication + leakage audit

- **Hypothesis.** The published 84.48 / 92.39 is inflated by the **~80% train/test segment overlap** flagged in the SEED-V replication; the SEED-IV replication never reached published numbers (44.7–70.4%).
- **Addresses.** Every PhysioFM-S-vs-PC-SSL statement currently leans on a number we don't trust (Interpretation §4). This is the single highest-leverage item for a publishable comparison.
- **Method.**
  1. **Audit** the original split: quantify exact train/test segment overlap (windows sharing a trial, or overlapping sliding windows). Report the leakage fraction explicitly.
  2. Re-run PC-SSL under **leakage-free** splits (trial-disjoint train/test; ideally also subject-disjoint LOSO) using the authors' code.
  3. Report PC-SSL clean numbers next to PhysioFM-S and raw DE under the *same* clean protocol.
- **Decision rule.**
  - Clean PC-SSL drops toward the raw-DE/PhysioFM-S band → the project's "gap to beat" was largely leakage; the honest contribution is the mechanistic decomposition, and PhysioFM-S is competitive.
  - Clean PC-SSL stays high → conv+SE+MLP genuinely extracts more, and the open question becomes *why* a conv mixer beats a transformer on this static task (architecture, not pretraining).
- **Cost.** Medium (their code exists; mostly careful split engineering). **Do not write the PC-SSL comparison without this.**

---

## Tier 4 — Phase-3 hand-off (where temporal PC should finally pay off)

### F13 🟡 — Pre-registered temporal-PC test on a genuinely dynamic task

- **Hypothesis.** Temporal predictive coding fails on emotion DE *because the task is static*; it should help on tasks with real temporal dynamics. **Sleep staging** is the cleanest first test: strong temporal structure (stage transitions, sequence context), large public data (Sleep-EDF Expanded), low channel count.
- **Addresses.** Turns the Stage 2 negative into a *prediction* and tests it — the scientifically strongest move.
- **Method.** Same PhysioFM-S machinery, DE (or band-power) features, sleep-staging readout (which is inherently sequence-level). **Pre-register** the directional prediction: pretrained > random-init here, unlike emotion.
- **Decision rule.**
  - Pretrained > random on sleep but not emotion → a clean, generalizable thesis: *time-series FM objectives help biosignal tasks in proportion to their genuine temporal dynamics.* That is the paper.
  - No benefit on sleep either → the temporal-PC-in-a-transformer idea is in deeper trouble and the project should pivot fully to structure/invariance (Tier 2).
- **Cost.** High (new dataset pipeline) — but it's the proposal's Phase 3 anyway, now with a sharp hypothesis instead of a vague "pretrain on more data."
- **Blocker.** Sleep-EDF / CHB-MIT / BCI-IV-2a are **not uploaded**; Phase 3 is data-blocked. Getting Sleep-EDF in is the unlock.

---

## Suggested order (and the minimal path to a writeup)

1. **F1** → **F4** → **F2** → **F3**  *(Tier 0: ~1–2 days of compute; decides whether the negative result is fundamental or an artifact, and makes every comparison fair).*
2. **F12** in parallel *(the keystone number; independent of the above).*
3. If F1/F2 keep the null on raw DE with a fair readout → the negative result is **publishable as-is**, scoped per Interpretation §7. Add **F13 (sleep)** to turn it from "we found a null" into "we found *when* the method works."
4. If F1/F2 **flip** the result on raw DE → the story becomes "LDS smoothing was hiding the dynamics," and you re-run the ablation grid (F5–F7) on raw DE.

**Minimal blocking set for any external claim: F1, F2, F3, F4, F12.** Everything else strengthens or extends; these five close the confounds that a reviewer (or a careful co-author) would raise first.
