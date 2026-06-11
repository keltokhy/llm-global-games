# Analysis Note v2 (pre-run): Confirmatory Regime-Analyst Experiments — "The Intelligence Cost of Surveillance"

*Written 2026-06-11, before any confirmatory inference was run.*

## Relation to v1 — what changed and why (stated openly)

The Phase A pilot (prereg v1, `analyst_pilot_prereg.md`) ran 2026-06-10 on 150 matched nested cells × 5 analyst models. Two lessons force a revision; the pilot is hereby labeled **exploratory** and the runs below are the confirmatory test.

1. **The v1 targeting metric (precision@5 lift) was mis-designed.** Lift = prec@5 − jf_true is mechanically compressed toward zero in cells where nearly all or nearly no senders join (most cells). It cannot move even when ranking quality collapses. **Replaced by per-cell ranking AUC** of stated P(JOIN) against actual decisions, computed only on cells with both classes present — decomposable, so it admits the paired sign-flip permutation test.
2. **The pilot's effect is crisis-conditional, as the theory's selection mechanism (Lemma 1, cutoff coding) predicts**: blinding concentrated at θ < 0 (all 5 analysts on Brier, p ≤ 0.001), absent or mildly reversed at θ ≥ 0. v1's unconditional primaries diluted the effect with strong-state cells where the model says no effect should exist. **v2 primaries are conditional on θ < 0**, with the θ ≥ 0 cells serving as the model-implied placebo.

## Confirmatory hypotheses

- **H-C1 (crisis blinding, primary):** On weak-regime cells (θ < 0), analysts reading surveilled messages perform worse than on matched baseline messages, on (a) regime-survival Brier and (b) per-cell sender-ranking AUC.
- **H-C2 (strong-state placebo):** On θ ≥ 0 cells, deltas are indistinguishable from zero (or reversed). A significant blinding effect at θ ≥ 0 would *contradict* the selection mechanism.
- **H-C3 (sparse robustness):** H-C1 survives when the analyst sees only k = 10 intercepted messages per cell (realistic partial interception; also decompresses ceiling effects).
- **H-C4 (capability gradient):** The crisis-blinding delta does not vanish for a frontier analyst — i.e., a smarter watcher cannot simply read through the code. (Directional; exploratory if only one frontier model is run.)
- **H-C5 (sender generalization):** H-C1 holds when the senders are a different model family (Qwen3-30B corpus).

## Runs (all decided before execution)

| Run | Corpus | Analysts | Items |
|---|---|---|---|
| R1 sparse | nested, `--n-messages 10`, same 150 cells, seed 5150 | deepseek-v4-flash-20260423, llama-4-maverick, qwen3.7-plus, llama-3.3-70b | 300 × 4 |
| R2 frontier | nested, full 25 messages, same 150 cells | anthropic/claude-opus-4.8 | 300 |
| R3 cross-sender | clean_qwen30 (250 matched cells × ≤10 senders) | deepseek-v4-flash-20260423, llama-4-maverick | 500 × 2 |

GLM-5.1 is dropped from the confirmatory roster (highest cost per item in the pilot, flat sender-level metrics); its pilot results remain reported. Temperature 0.0, blinded prompts v1 (unchanged), deterministic per-arm shuffles, same usability and degenerate-message filters as v1.

## Estimands and tests

Primary (per analyst, nested corpora): per-cell paired Δ(baseline − surveillance), **restricted to θ < 0 cells**, for (a) Brier of FALL vs coup outcome, (b) per-cell sender-ranking AUC. Paired t and 10,000-draw sign-flip permutation (seed 42). Placebo: same deltas on θ ≥ 0 cells, expected ≈ 0.

Secondary: pooled sender AUC (cluster bootstrap, as v1); join-fraction MAE (expected null, as in pilot); aggregate-vs-individual wedge (I_A unaffected while I_θ, I_i degrade).

**Confirmatory success criterion:** for ≥ 3 of the 4 R1 analysts, *both* primary metrics degrade in the predicted direction at θ < 0 with permutation p < 0.05, *and* the θ ≥ 0 placebo is not significantly degraded in ≥ 3 of 4. R2 and R3 are reported as robustness, not gates.

## Exclusions / handling

As v1 (unparsed senders excluded with rates reported; 75/25 imputation flagged; degenerate messages filtered symmetrically; API-error cells dropped pairwise). Per-cell AUC additionally requires both classes among parsed senders; cells failing this in either arm are dropped pairwise for that metric and counted.
