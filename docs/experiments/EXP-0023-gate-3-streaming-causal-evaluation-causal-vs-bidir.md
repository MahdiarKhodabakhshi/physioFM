---
id: EXP-0023
title: Gate 3 — Streaming/causal evaluation: causal vs bidirectional twin under a latency constraint
status: done
created: 2026-08-18
run_date: 2026-08-19
agent: claude-code
phase: next-phase
verified: no
tags: gate3, streaming, causal, bidirectional, latency
commits: 951a916, d4e100a
verdict: CONFIRMED. Sleep DE, same 5 subject-disjoint folds and fine-tuning recipe: the bidirectional twin (identical stack, full attention) scores 74.10 / k .655 offline but 70.40 / k .607 when only the past is visible; the causal model scores 73.18 (random-init) and 75.37 (input-PC) in both settings (online == offline verified) at 1 token per decision (KV cache) vs 190 for the twin. Under a streaming constraint the causal decoder wins by +2.8 (random-init) / +5.0 (pretrained); offline the twin is +0.9 better. The plan's 'defensible claim' holds.
---

# EXP-0023 — Gate 3 — Streaming / causal evaluation

> **Status:** planned · **Created:** 2026-08-18 · **Agent:** claude-code · **Phase:** next-phase

## 1. Why
The plan's "defensible claim": a causal decoder can legitimately win where decisions must be
made at time *t* from data ≤ *t*. We build the bidirectional twin of the same stack
(`PhysioFMS(causal=False)`), fine-tune both on the same folds, and evaluate (a) offline
(whole window visible) and (b) online (only the past visible at each epoch, i.e. the
bidirectional model is re-run on the truncated prefix). Prediction: offline the bidirectional
twin ≥ causal; online the causal model ≥ bidirectional and is O(1) per decision vs O(L).

## 2. Setup
`scripts/gate3_streaming_eval.py` on sleep (DE, and raw if Gate 2 lands): same 5 subject-
disjoint folds, --mode full fine-tuning, per-epoch accuracy/κ offline vs online, wall-clock per
decision.

## 3. Status & run log
- 2026-08-18 — created; code review found the SDPA fallback that made `causal=False` silently causal on unpadded batches — fixed (`layer.self_attn.is_causal = causal`) and verified before running.
- 2026-08-19 — run locally (`scripts/gate3_streaming_eval.py`), 4 arms × 5 folds; done.

## 4. Results *(2026-08-19; `results/phase4/gate3/streaming.csv`)*

| arm | attention | offline acc / κ | online acc / κ | tokens/decision |
|---|---|---:|---:|---:|
| causal_rand | causal | 73.18 / .645 | 73.18 / .645 | 1 (190 without cache) |
| bidir_rand | bidirectional | 74.10 / .655 | 70.40 / .607 | 190 |
| causal_latent | causal | 74.12 / .656 | 74.12 / .656 | 1 |
| causal_pc | causal | 75.37 / .672 | 75.37 / .672 | 1 |

## 5. Interpretation — agent's reading

Where decisions must be made from data ≤ t, the causal decoder is the right inductive bias: it loses nothing online, whereas the bidirectional twin loses 3.7 points, and it does so at 1/190 of the compute per decision. This is a legitimate, measured advantage of the architecture (not of the pretraining) and is the one gate whose promised claim survives intact. See `docs/NEXT_PHASE_RESULTS.md` §4.

## 6. ✅ Your verification — *(reserved for Mahdiar)*
- [ ] **Verified**

## 7. Commits

## 8. Links
- Plan: `docs/NEXT_PHASE_PLAN.md` §4 Gate 3
