# Analysis Note (pre-run): Regime-Analyst Pilot — Phase A of "The Intelligence Cost of Surveillance"

*(Paper retitled 2026-06-11; was "The Dictator's Microphone". Note content unchanged — written before the pilot ran.)*

*Written 2026-06-10, before any analyst inference was run. Mirrors the CLEAN_RUN preregistration discipline.*

## Question

Does surveillance at the message-writing stage degrade the informativeness of intercepted messages **to the regime**? Paper 1 established the citizen-side effect (receivers join less). This pilot tests the regime-side effect on the same archived message corpora.

## Hypotheses

- **H-A1 (blinding):** Analyst models reading surveilled messages recover the state and senders' intentions less accurately than when reading baseline messages from matched cells (same θ, same senders, same signals).
- **H-A2 (crisis asymmetry):** Degradation is larger in weak-regime states (θ < 0), where subversive content would otherwise have been written.
- **H-A3 (codedness alone insufficient or sufficient — two-sided):** On the direct/coded message bank (codedness induced without monitoring framing), θ-recovery degradation is measured without a directional prior; paper 1 found coded style alone did not move *receivers*, so whether style alone blinds *analysts* is an open empirical question.

## Data (archived; no new sender-side generation)

- **Primary:** nested Llama 3.3 70B paired corpus — `output/revision-nested-comm/.../experiment_comm_log.json` (baseline) vs `output/revision-nested-surv/.../experiment_comm_log.json` (surveillance). 500 matched cells, bitwise-identical θ and per-agent signals across arms; per-sender messages and JOIN/STAY decisions recorded.
- **Codedness contrast:** `CLEAN_RUN/message_banks/direct_coded_pairs.parquet`, 210 cells × 25 senders, direct vs coded variants; θ truth only (sender decisions unrecoverable — verified before this note).

## Design

Per (cell × arm × analyst): one bundled prompt presenting the cell's intercepted messages in deterministic-shuffled order with neutral "Sender k" labels. Blinding: identical template across arms; no arm labels, no θ, no signals, no briefings, no sender reasoning. Elicited, strict line format:

1. `FALL: 0–100` — P(regime falls within the month)
2. `JOIN_PERCENT: 0–100` — share of shown senders who chose JOIN
3. `SENDER k: JOIN|STAY p` — per-sender verdict + P(JOIN)

Targeting is derived (top-5 by stated P(JOIN)). Temperature 0.0. Cells: 150 of 500, stratified by θ decile, sampled once before arm/analyst expansion (seed 5150). Sender set per cell: intersection of usable senders across arms (non-empty message, valid decision, no API error), capped at 25.

**Analyst roster (capability gradient):** workhorse fleet — `deepseek/deepseek-v4-flash-20260423`, `meta-llama/llama-4-maverick`, `qwen/qwen3.7-plus`, `z-ai/glm-5.1-20260406`; gradient anchors (nested corpus only) — `meta-llama/llama-3.3-70b-instruct` (2025-era) and one frontier model (`openai/gpt-5.5` or `anthropic/claude-opus-4.8`).

## Estimands and tests

Primary metrics per analyst × arm:

1. **Strength:** Brier((FALL/100 − coup_success)²); AUC(FALL → coup_success); |Spearman ρ(FALL, θ)|. FALL is never scored in θ units.
2. **Join fraction:** MAE(|JOIN_PERCENT/100 − jf_true_shown|); calibration slope.
3. **Per-sender:** accuracy; pooled AUC of P(JOIN).
4. **Targeting:** precision@5 **lift** = prec@5 − jf_true_shown (lift, because base rates differ across arms: 0.471 vs 0.391).

Primary contrast: per-cell paired Δ(baseline − surveillance). Decomposable metrics: paired t-test and 10,000-draw sign-flip permutation test (seed 42). Pooled AUC and ρ: cluster bootstrap over cells (10,000 resamples). Heterogeneity: θ<0 vs θ≥0 split (H-A2). Robustness slice: Δ-metrics vs Δ-mean-message-length (surveilled messages are shorter; length is part of the treatment, reported transparently).

## Go/no-go rule (stated in advance)

**GO** if surveillance degrades per-sender AUC **and** precision@5 lift in the predicted direction with permutation p < 0.05 for at least 2 of the 3+ analyst models on the nested corpus. Otherwise: NO-GO for the headline as designed; diagnose via the difficulty knob (`--n-messages 10` sparse-interception variant) before abandoning — ceiling effects on 25-message bundles are the pre-identified failure mode.

## Exclusions / handling

Unparsed sender lines excluded from sender metrics (parse rates reported per analyst; sensitivity check dropping cells with >20% unparsed). Missing per-sender probabilities imputed 75/25 with flags. API-error analyst responses re-run via cache-aware retry; cells with unrecoverable analyst errors dropped pairwise. Degenerate sender messages (symbol-spam glitch output: alphabetic fraction < 0.75 or > 1,500 chars; 41/12,500 baseline, 21/12,500 surveillance) excluded at the usability layer, which via the intersection rule drops those senders from both arms symmetrically.
