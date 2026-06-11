# Analysis Note v3 (pre-run): Held-Out Confirmation — "The Intelligence Cost of Surveillance"

*Written 2026-06-11, before any held-out inference was run.*

## Purpose and relation to v1/v2

v1 (exploratory pilot) and v2 (confirmatory, **NOT CONFIRMED** as preregistered — 1/4 passed the conjunctive primaries, with the binding failure isolated to the per-cell ranking AUC metric) are on record. Across v1/v2, crisis-state Brier degradation was significant in 10/10 analyst×corpus tests, but no single preregistered gate has yet passed cleanly. v3 is the decisive test, designed to be immune to the metric-shopping critique on two grounds: (i) a **single primary**, chosen before this run from the margin that was robust in *both* prior waves; (ii) **fresh data** — the 348 nested cells never sampled by any prior analyst run (complement of the seed-5150 150-cell sample).

## Hypothesis and primary

**H-V3:** On held-out weak-regime cells (θ < 0), analysts reading surveilled messages show higher Brier error on regime-survival judgment (FALL vs. coup outcome) than on matched baseline messages.

**Single primary estimand:** per-cell paired Δ Brier (baseline − surveillance), θ < 0 cells, per analyst. Test: 10,000-draw sign-flip permutation, seed 42. **Success criterion: Δ < 0 with p < 0.05 for all 4 analysts.** Placebo (must hold): θ ≥ 0 Brier delta not significantly negative for ≥ 3 of 4.

Everything else (sender accuracy, AUCs, join-fraction MAE, |ρ|) is reported as secondary, no gates.

## Run

All 348 held-out nested cells (full 25-message interception), analysts: deepseek-v4-flash-20260423, llama-4-maverick, qwen3.7-plus, llama-3.3-70b. Temperature 0.0, prompts v1 unchanged, same usability/degeneracy filters, `--holdout` selection (complement of `sample_cells(pairs, 150, seed=5150)`).
