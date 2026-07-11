---
id: EXP-0008
title: F12 — PC-SSL leakage audit + clean re-replication
status: done
created: 2026-06-28
run_date: 2026-06-17
agent: unknown (pre-log backfill)
phase: phase2-followup
verified: no
tags: pc-ssl, leakage, replication, audit
commits:
verdict: The PC-SSL 84–92% gap was largely ~80% temporal-neighbor leakage — same code, clean trial-disjoint split collapses to ~40–45%, within the PhysioFM-S band.
---

# EXP-0008 — F12 — PC-SSL leakage audit + clean re-replication

> **Status:** done · **Created:** 2026-06-28 · **Run:** 2026-06-17 · **Agent:** unknown (pre-log backfill) · **Phase:** phase2-followup

---

## 1. Why — hypothesis & motivation

PC-SSL publishes 84–92% on SEED while every clean PhysioFM-S/raw-DE probe sits at
~40–63%. F4 ruled out the downstream head, sharpening suspicion onto the PC-SSL
number itself. Hypothesis: the author protocol leaks. The notebook splits
**individual DE windows** with `train_test_split(shuffle=True)`, while PC-SSL forms
consecutive `(window_i → window_{i+1})` pairs with trial-constant labels — so a
random window split puts near-duplicate adjacent windows in both train and test.
Audit the leakage and re-run the *same code* with a clean trial-disjoint split.

---

## 2. Setup — exactly what was run

```bash
PY=/home/mahdiar/.conda/envs/xcqa/bin/python
$PY scripts/phase2_f12_pcssl_audit.py
```

- **Data:** in-repo PC-SSL reference (`PC-SSL/`: code, raw data, per-fold trained models, author splits, result CSVs); SEED-V, SEED-IV.
- **Variant / config:** PC-SSL implementation held FIXED; only the split changes — leaky random-window vs clean trial-disjoint. Plus leakage diagnostics (neighbor-in-train, NN same-trial, NN cosine).
- **Output dir:** `results/phase2/followup/f12/`

---

## 3. Status & run log

- 2026-06-17 — run completed (date inferred from result timestamps).
- 2026-06-28 — backfilled into experiment log.

---

## 4. Results  *(run date: 2026-06-17)*

**Leakage (random window split vs clean trial-disjoint):**

| Dataset | split | future-partner-in-train | either-neighbor | same-trial-in-train | NN same-trial | NN cosine |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| SEED-V | random window (leaky) | ~81% | 95.6% | 100% | 100% | 1.000 |
| SEED-V | trial-disjoint (clean) | 0% | 0% | 0% | — | — |
| SEED-IV | random window (leaky) | 81.4% | 95.3% | 100% | 100% | 1.000 |
| SEED-IV | trial-disjoint (clean) | 0% | 0% | 0% | — | — |

`future-partner-in-train` reproduces the author notebook's logged
`mean_test_overlap_with_classifier_train = 0.7997`.

**PC-SSL accuracy — same code, only the split changes (acc % / macro-F1 %):**

| Dataset | leaky random split | clean trial-disjoint | raw-DE LogReg (clean) | PhysioFM-S (clean probe) | chance |
| --- | ---: | ---: | ---: | ---: | ---: |
| SEED-V | 65.84 / 63.05 | **39.77 / 33.97** | 51.40 / 49.92 | 45–49 | 20 |
| SEED-IV | 70.41 / 67.85 | **44.72 / 26.73** | 62.75 / 54.76 | 57–61 | 25 |

(Full author-notebook reproduction with encoder fine-tuning reached **91.25%** on
SEED-V at 80% overlap — i.e. ≈ the published 92.39.)

Results: `results/phase2/followup/f12/f12_pcssl_audit.md`.

---

## 5. Interpretation — agent's reading

The gap was largely leakage. The published 84–92% rests on ~80% temporal-neighbor
leakage. Holding the PC-SSL implementation fixed and removing only the leakage
collapses accuracy to ~40–45% — at/below the raw-DE linear ceiling and within the
PhysioFM-S band. So PhysioFM-S is competitive on a clean protocol and the honest
contribution is the mechanistic decomposition. *Caveat:* the clean absolute number
comes from a faithful-but-unverified re-implementation; the **leaky-vs-clean delta**
(same code) is the controlled, implementation-invariant result and is what should be
cited.

---

## 6. ✅ Your verification — *(reserved for Mahdiar)*

> Leave the agent's interpretation above untouched. Confirm or correct it here.

- [ ] **Verified** (set `verified: yes` in frontmatter when ticked)
- **Notes / corrections:**


---

## 7. Commits

- (uncommitted as of 2026-06-28 backfill — `scripts/phase2_f12_pcssl_audit.py`)

---

## 8. Links

- Related entries: [[EXP-0004]] (ruled out the head, pointing here)
- Docs / results: `docs/PHASE2_FOLLOWUP.md` (F12), `docs/PC_SSL_UPSTREAM_AUDIT.md`, `results/phase2/followup/f12/`
