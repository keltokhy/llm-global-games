# Paper 2 — Monitoring and the Common-Knowledge Function of Communication

**Thesis.** In a Morris–Shin global game of regime change played by LLM agents,
monitoring the communication channel suppresses collective action — not by
hiding information, not by deterring directly, and not by making messages
illegible — but by causing agents to **downgrade the confident public assertion
that the regime is beatable**. That dampens the common-knowledge signal of
vulnerability, which is the focal input coordination requires in a global game.

This is the mechanistic, validated deepening of Paper 1 (which establishes the
effect). Paper 2's contribution is the **mechanism**, identified causally and
validated against competing explanations.

## What is locked (empirics)

1. **The effect is message-mediated.** Surveillance messages replayed verbatim
   to naive receivers (no surveillance framing) suppress joining by **−8.6pp**
   (paired t=−16.1, p=3e-47; 95% CI [−9.6, −7.5]). Same magnitude as the live
   effect. It is the *content* of what agents say, not the decision prompt.

2. **The mechanism is a beatability downgrade.** An independent, different-family
   LLM judge scores how confidently each message asserts the regime is beatable.
   Surveillance lowers it (≈ −0.65 to −0.85 on 0–10), and that downgrade
   **mediates the join drop** (~67% in the replay, ~34% in the live nested set).
   Validated across **two designs** (expB replay, nested 500-cell) × **two judge
   families** (Qwen3-30B, Mistral-small-24B; writers are Llama-3.3-70B) × a
   lexicon proxy (judge↔certainty r≈0.45–0.51).

3. **Three competing mechanisms refuted.**
   - *Illegibility / covert channel* — a TF-IDF classifier separates
     surveillance from no-surveillance messages at **AUC≈1.0** (even both-coded),
     so the channel is overt, not hidden.
   - *Hedging / conditionality* — null and wrong-signed within matched cells
     (p=0.77).
   - *Collective "we" framing* — judges disagree on its sign (Qwen Δ=0.00,
     Mistral Δ=+0.67); not a mechanism.

## Structure

- §Effect: surveillance suppresses joining; replay isolates the message channel.
- §Mechanism: beatability mediation; refutation of illegibility/hedging/collective.
- §Theory: global game — monitoring lowers the public signal of vulnerability,
  raising the effective coordination threshold; the behavioral drop matches the
  comparative static. (No new data.)
- §Generality: cross-model (existing `output/xmodel-*` Llama-writes/Qwen-reads
  matched runs) + ONE new non-political coordination game (bank run / evacuation)
  — the single data gap, needed to rule out "political-content safety tuning."

## Code

- `mechanism_analysis.py` — deterministic pipeline (no API): replay effect,
  illegibility refutation (AUC), lexicon mediation, data-named top terms.
- `judge_mechanism.py` — API-based judge validation (OpenRouter); caches per
  message to `output/paper2_judge_cache/`. `MODE`/`SURV`/`COMM`/`JUDGE` via env.

## Remaining work

1. Second coordination game (new generation) — the generality gap.
2. Global-games theory section.
3. Cross-model mediation on existing `xmodel` data.
4. Manuscript.
