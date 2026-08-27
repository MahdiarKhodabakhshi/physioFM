# Second sleep dataset — survey & recommendation (2026-08-27)

**Question:** the tf64 + causal-decoder + PC-pretraining workflow works on Sleep-EDF-78
(77.9 % / κ 0.71, fine-tuned, subject-disjoint). Which other sleep dataset(s) should we
use to validate it, given the ICASSP deadline (Sep 16) and our constraints?

**Constraints checked against every candidate:** raw EEG ≥ 100 Hz (tf64 Welch bins
0.5–49 Hz; `compute_log_spectrogram` is rate-agnostic, no resampling needed), per-30-s
AASM/R&K-mappable 5-class labels, ≥ 20 subjects for subject-disjoint splits, EDF-ish
format (our loader is MNE-based; `feature_fn` injection keeps epoching/labels shared),
disk (437 GB free local, 2 TB pod), and — critically — **published baselines on a
comparable split** so the result means something.

---

## TL;DR — ranked

| Rank | Dataset | Why | Access | Size | Effort | Comparison ladder |
|---|---|---|---|---|---|---|
| **1** | **HMC** (PhysioNet) | 151 subjects, 4 EEG @ 256 Hz, EDF, patient cohort; fixed published split with ~8 modern foundation-model baselines; best published κ ≈ 0.70 → our κ-0.71-class model is *competitive*, not embarrassed | CC-BY, zero registration | 16 GB | ~½ day loader; ladder ≈ same compute as Sleep-EDF-78 (151 vs 153 recordings) | NeuroLM/LaBraM/CBraMod/CSBrain/REVE, BAC/κ/wF1, full FT — matches our protocol |
| **2** | **Physio2018** (CinC training set) | 994 labeled subjects, 6 EEG @ 200 Hz — the scale test (6.5× pretraining data); lowest-SOTA major benchmark (best κ .737) → we'd sit ~0.03 from SOTA, vs 0.08 on SEDF-78 | ODC-By, zero registration | 135 GB | wfdb loader (not EDF) + big compute (994 nights) | XSleepNet 80.3/κ.732, SleePyCo 80.9/κ.737, SeqSleepNet 79.4/κ.719 — byte-identical published protocol (C3-A2, 5-fold) |
| **3** | **SHHS** (NSRR) | Reviewer gold standard; 5,793 EDFs, 2 EEG @ 125 Hz; enables the **cross-dataset transfer** experiment (the one setting where the literature says pretraining reliably pays) | Free DAUA, intern-signable, **no REB**; official "up to 2 weeks", typically days → **submit today** | 230 GB slice (shhs1 EDFs + profusion XMLs) | XML stage parser (simple); heavy compute | κ .81–.84 (L-SeqSleepNet .838) — we can't win absolute; role = credibility + transfer source |
| **4** | **ISRUC-SG1** | 100 patients, 6 EEG @ 200 Hz, two scorers; CBraMod fixed split (1–80/81–90/91–100) with a full FM ladder | Open MEGA links | ~14 GB | .rec→EDF rename, txt labels, MEGA throttling | CSBrain BAC .793/κ.741, CBraMod .787/.744, LaBraM .763/.723, U-Sleep .759/.721 |

**Recommendation: do HMC now (this week, guaranteed), submit the NSRR request for SHHS
today (free, likely approved before the deadline), and treat Physio2018 as the
scale/second-open option if pod compute allows. Skip MASS for this paper.**

---

## The strategic point the survey surfaced

The published SSL-for-sleep literature shows **exactly our pattern**: in-domain,
full-label pretraining gains are small or null (BENDR: pretrained ≈ random-init;
mulEEG: supervised > SSL in-domain; neuro2vec: +2.3; TS-TCC: supervised ≈ SSL). Our
+0.85 (tf64) / +2.5 (DE) is *consistent with the field*, not a private failure.

Where pretraining **does** reliably pay in the literature is **cross-corpus transfer**:
SleepTransformer SHHS→SEDF-78 +3.5 acc / +.046 κ; Phan et al. MASS→SEDF +3.0;
L-SeqSleepNet SHHS-init +2.3 to +6.6. **We have never run this** — it is the proposal's
Phase-3 promise (joint pretraining, zero-shot transfer). A second sleep corpus is not
just validation: it enables the one experiment where our pretraining has its best shot
(pretrain on HMC/SHHS/DCSM → fine-tune on SEDF-78, and the reverse). If the effect
appears, the paper's pretraining story flips from "≈ null" to "pays off under domain
shift, as in the supervised literature".

Cross-database difficulty is real and quantified: Alvarez-Estevez & Rijsman 2021 (the
HMC paper) found local κ 0.80 → cross-database κ 0.54 average; SleepDG (AAAI 2024) does
leave-one-dataset-out over SEDF/HMC/ISRUC/SHHS/P2018. Both are ready-made framing.

---

## Calibration warning for our own SEDF-78 number

Sleep-EDF-78 has two epoch-selection conventions worth ~4 acc points: **A** = keep ±30
min wake around sleep (the SOTA-headline convention), **B** = in-bed only. **Our
pipeline is Convention A** (`trim_wake_min=30.0` in `physiofm/sleep_edf.py`,
verified 2026-08-27). Fair comparisons at our operating point:

| Model (Convention A, 1-ch Fpz-Cz) | acc | κ |
|---|---|---|
| SleepTransformer (SHHS-pretrained) | 84.9 | .789 |
| SleePyCo | 84.6 | .787 |
| XSleepNet2 | 84.0 | .778 |
| TinySleepNet | 83.1 | .77 |
| SeqSleepNet | 82.6 | .760 |
| SleepTransformer (scratch) | 81.4 | .743 |
| U-Time | 81.3 | .745 |
| **PhysioFM-S tf64 (2 ch, causal, online-capable)** | **77.9** | **.71** |

All of those use 10–21-epoch **bidirectional** context; ours is causal (and we showed
causal wins the streaming regime by +2.8/+5.0). That is the honest positioning: a
different operating point, not a worse model on the same axis. Metric families never to
mix: supervised ladder = plain acc/MF1/κ; foundation-model ladder (HMC/ISRUC) =
**balanced acc**/κ/weighted-F1, full fine-tune; DOD = consensus-F1.

---

## Tier 1 details

### 1. HMC — Haaglanden Medisch Centrum (PhysioNet)
- https://physionet.org/content/hmc-sleep-staging/1.1/ — CC-BY 4.0, direct download, 12.9 GB zip.
- 151 recordings = 151 subjects (85 M/66 F, age 54 ± 15), heterogeneous sleep-disorder
  patients → population complement to SEDF's healthy-leaning cohort.
- EEG F4-M1, C4-M1, O2-M1, C3-M2 @ 256 Hz; EDF signals, EDF+/txt hypnograms, AASM 5-class.
- Fixed published split (NeuroLM protocol): subjects 1–100 train / 101–125 val / 126–151 test.
- Published ladder (BAC / κ / wF1, full fine-tune): REVE-Base .740/.698/.764 ·
  CSBrain .735/.682/.751 · LaBraM-Base .729/.681/.755 · CBraMod .727/.669/.740 ·
  EEGPT .703/.658/.732 · BIOT .686/.630/.709 · NeuroLM-B .674/.619/.713.
  **Best published κ ≈ 0.70 on this split** — a κ-0.71-class architecture (ours, on
  SEDF-78) lands in genuinely competitive territory against billion-token foundation
  models, with a 2.4 M-parameter model.
- Gotchas: use v1.1 (v1.0 had 3 bad recordings — don't mix subject lists); mixed
  in-lab/ambulatory.
- Effort: pairing function + stage map + channel names (~½ day); pretrain+FT ladder ≈
  Gate-0 sleep cost (151 ≈ 153 recordings).

### 2. Physio2018 — CinC Challenge training set
- https://physionet.org/content/challenge-2018/1.0.0/ — ODC-By, open; training portion ≈ 135 GB.
- 994 labeled subjects (MGH sleep lab, suspected apnea); test-set labels never released.
- 6 EEG (F3-M2, F4-M1, C3-M2, C4-M1, O1-M2, O2-M1) @ 200 Hz + EOG/EMG/resp/SaO2/ECG.
- AASM stages incl. "undefined" (drop), delivered as sample-indexed WFDB annotations →
  expand to 30-s grid; **format is MATLAB v4 .mat + .hea, not EDF** → loader uses
  `wfdb`-python instead of MNE (moderate, self-contained).
- Published protocol shared *identically* by XSleepNet2 (80.3/κ.732) and SleePyCo
  (80.9/κ.737), plus SeqSleepNet 79.4/.719, U-Time 78.8/.714: C3-A2, 5-fold
  subject-disjoint, 50-subject val. Lowest SOTA κ of any major benchmark.
- Why it matters for us: 6.5× the pretraining data of SEDF-78 — the honest test of
  "does PC pretraining scale", and a big-N architecture validation.

### 3. SHHS via NSRR — submit the request today
- https://sleepdata.org/datasets/shhs — free; per-dataset web DAUA, **individual
  requester e-signs, no PI/institutional signature, IRB question can be answered
  "not required"** for secondary use; one request can cover several datasets (tick
  MESA/CCSHS too, costs nothing). Official guidance "up to two weeks"; staff report
  most requests clear in ≤ 1–2 weeks, often days. Use the nrc-cnrc.gc.ca email.
- shhs1 = 5,793 EDFs, 228 GB; + profusion XML stage annotations 0.6 GB = **~230 GB
  canonical slice** (fits local disk; the `nsrr` ruby gem downloads per-folder).
- EEG C4-A1 + C3-A2 @ 125 Hz (Nyquist 62.5 > our 49 Hz fmax — fine).
- SOTA (70/30 split): L-SeqSleepNet 88.4/κ.838, SleePyCo 87.9/.830, SleepTransformer
  87.7/.828. We cannot win that table with 2.4 M params; SHHS's role here is
  (a) reviewer credibility, (b) **transfer source** for the Phase-3-style experiment.
- Note: NSRR currently shows a US-federal review banner (Aug 2026) — argues for
  submitting immediately.

### 4. ISRUC (backup / alternative to HMC)
- https://sleeptight.isr.uc.pt/ — open MEGA downloads (use mega-cmd; quota throttling).
- SG1: 100 patients, 1 night; SG3: 10 healthy. 6 EEG @ 200 Hz, `.rec` = EDF renamed,
  two independent scorers (papers use expert 1). ~14 GB.
- CBraMod-protocol ladder (BAC/κ/wF1): CSBrain .793/.741 · CBraMod .787/.744 ·
  REVE .782/.750 · LaBraM .763/.723 · U-Sleep .759/.721 · BIOT .753/.719.
- Gotcha: signal/label length off-by-a-few at tails (trim to min).

---

## Tier 2 (use if a specific angle is wanted)

| Dataset | N / spec | Access | Angle |
|---|---|---|---|
| **DOD-H/O** | 25 healthy + 55 OSA, 250 Hz, **5 scorers + consensus** | Zenodo (58 GB, HDF5) | Scorer-noise/agreement-aware metrics; SimpleSleepNet κ .846/.823, RobustSleepNet MF1 85.1/82.7 |
| **DCSM** | 255 patient PSGs, 6 EEG @ 256 Hz, EDF | Open (ERDA archive) | Extra pretraining corpus for multi-corpus pretrain → transfer; U-Sleep F1 .81 |
| **BOAS (Bitbrain)** | 128 nights / 108 healthy subjects, 6 EEG, 3-scorer consensus, BIDS EDF | OpenNeuro ds005555 | Modern healthy cohort; map nights→subjects before splitting |
| **CCSHS / CFS / MESA** | 515 / 730 / 2,056 PSGs @ 128–256 Hz | Same NSRR DAUA (tick on the same form) | CCSHS = cleanest signals (adolescents); MESA = diverse 54–95 |
| **CAP** | 108 PSGs, rates 100–512 Hz vary per record | PhysioNet open, 40 GB | Only 2 staging baselines — robustness test, not a benchmark |
| **UCD/St. Vincent's** | 25 subj, 2 EEG @ 128 Hz | PhysioNet open, 1.3 GB | Cheap sanity check only |
| **DREAMS** | 20 subj, 20-s epochs | Zenodo, CC-BY-NC-ND | 20-s grid breaks our pipeline; skip |

**Skip for this paper:** **MASS** (email application + REB accreditation + license
agreement, weeks-to-months, no Canadian fast-track — revisit for a journal extension;
SS3 is the classic DeepSleepNet benchmark), **NCH** (credentialed, 2.3 TB, pediatric),
**HSP/BDSP** (120k recordings; DUA + training; the "next phase" resource).

---

## Proposed 3-week plan

1. **Today:** NSRR account + one DAUA covering SHHS (+ MESA/CCSHS); download HMC
   (16 GB); optionally start Physio2018 (135 GB) in the background.
2. **Aug 28–30:** HMC loader (`physiofm/hmc.py` mirroring `sleep_edf.py`), tf64
   archive, pretrain pc + epochs-0 control (seed ladder), fine-tune on the fixed
   1–100/101–125/126–151 split → first external validation, reported as BAC/κ/wF1
   against the NeuroLM-protocol table.
3. **Week of Sep 1:** cross-corpus transfer: pretrain on HMC (+DCSM if downloaded) →
   fine-tune SEDF-78, and reverse; this is the experiment the literature says should
   finally show a pretraining effect. Physio2018 ladder if pod compute allows.
4. **When SHHS clears:** shhs1 slice → pretrain → transfer to SEDF-78/HMC (Phase-3
   claim, SleepTransformer precedent).

Verification pointers: all dataset claims from hosting pages (PhysioNet/Zenodo/NSRR/
Borealis); baseline numbers from the papers' own tables — extracted texts cached in
the session scratchpad (`papers/*.txt`) for spot-checking.
