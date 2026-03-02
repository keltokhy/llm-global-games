# LLM Global Games — Website Content Reference

## Paper Title
**LLMs Can Play (Global) Games**

## Author
Khaled Eltokhy, Department of Economics, The Graduate Center, CUNY

## One-Line Summary
Seven LLM agents embedded in a regime-change coordination game play threshold strategies consistent with Bayesian Nash Equilibrium — and a social planner can manipulate outcomes by redesigning the information environment.

## Abstract (shortened)
The communication channel that enables coordination is also the channel through which regimes suppress it. I embed seven LLMs in the Morris–Shin (2003) regime change global game, conveying private signals as natural-language intelligence briefings. Across 1,800 country-periods and 45,000 decisions, join rates track the theoretical attack mass with mean r = +0.80 (p < 0.001 for every model), collapse to r ≈ 0 under scrambled briefings, and invert to r = -0.72 under flipped signals. The pattern reflects payoff structure, not sentiment: varying benefit/cost narratives shifts cutoffs in lockstep with theory (r = +0.97) — a 50 pp swing invisible to any text classifier. Surveillance poisons the communication channel: expressed joining falls by ~13.5 pp while second-order beliefs shift by less than 1 pp. Agents self-censor while interpreting others' silence as genuine.

## The Setup (for non-economists)
Imagine a country where citizens must independently decide whether to JOIN an uprising or STAY home. The regime has some hidden "strength" θ. If enough people join (mass > θ), the regime falls and joiners win. If not enough join, the regime survives and joiners are punished. Each citizen gets a noisy private signal about regime strength — delivered as a natural-language intelligence briefing.

The key insight from global game theory (Morris & Shin 2003): there's a unique equilibrium where each citizen has a threshold — join if your signal suggests the regime is weak enough, stay otherwise. The sigmoid curve below θ* (the theoretical cutoff) tells you what fraction of people join at each regime strength.

## Key Numbers
- 7 LLM models tested (Mistral, Llama, GPT-OSS, Qwen, DeepSeek, Trinity)
- 1,800 country-periods in pure treatment
- 45,000 individual JOIN/STAY decisions
- Mean correlation with theory: r = +0.80 (p < 0.001)
- Scramble destroys correlation: r ≈ 0
- Flip inverts it: r = -0.72
- BNE vs Level-k: BNE RMSE = 0.199 vs L1 = 0.307, L2 = 0.348
- Cost/benefit sweep: cutoff tracks theory with r = +0.97
- Surveillance chilling effect: -13.5 pp average across 5 models
- Second-order beliefs unchanged by surveillance (< 1 pp shift)
- Propaganda saturates: doubling plants from 5→10 adds 0 pp effect
- Surveillance × censorship: super-additive interaction

## Paper Structure (9 sections)
1. **Introduction** — The information channel is a trap
2. **The Global Game of Regime Change** — Theory: threshold equilibria, attack mass A(θ)
3. **Experimental Design** — Signal → text → decision pipeline
4. **Results** — Sigmoid alignment, BNE vs Level-k, identification tests
5. **Communication** — Pre-play messaging preserves signal structure
6. **Information Design** — Stability, instability, censorship, public signals
7. **Surveillance** — Belief-action wedge, self-censorship, instrument interactions
8. **Propaganda** — Mechanical dilution vs behavioral effect
9. **Conclusion**

## The Core Figure: The Sigmoid
The most important visualization: empirical join fraction vs regime strength θ. It's a sigmoid (S-curve) — when the regime is weak (low θ), most agents JOIN; when strong (high θ), most STAY. The empirical curve closely tracks the theoretical prediction. Under scrambled briefings, the sigmoid collapses to noise. Under flipped signals, it inverts.

## The Pipeline Diagram
θ (regime strength) → x_i = θ + noise → z-score → Briefing Generator (8 evidence domains, 3 latent sliders) → natural language briefing → LLM → JOIN/STAY

## Key Concepts to Convey
1. **Threshold Policy**: Each agent has an internal cutoff — join when signal says "weak regime"
2. **Attack Mass A(θ)**: Theoretical prediction of join fraction at each regime strength
3. **Scramble/Flip Falsification**: Breaking/inverting the signal destroys/inverts the pattern
4. **Cost/Benefit Sweep**: Same briefing text, different stakes narrative → 50pp swing proves it's not text classification
5. **Information Design**: A planner can shift equilibrium by redesigning signals (stability/instability/censorship)
6. **Surveillance → Self-Censorship**: Agents change behavior without changing beliefs
7. **Strategic Update Gap**: Each agent self-censors but thinks others' silence is genuine
8. **Belief-Action Wedge**: Beliefs shift ~0 pp, actions shift ~13.5 pp

## Equations (simplified for web)
- Regime falls when: A > θ (attack mass exceeds regime strength)
- Payoffs: Join + regime falls = +B; Join + regime survives = -C; Stay = 0
- Signal: x_i = θ + ε_i where ε ~ N(0, σ²)
- Threshold: Join if x_i < x* where x* = θ* + σΦ⁻¹(θ*)
- Attack mass: A(θ) = Φ((x* - θ)/σ)
- θ* = B/(B+C) = 0.50 when B=C=1

## Color Palette Suggestion (from figures)
The paper figures use: blue for pure/baseline, orange for communication, green for various treatments, red for flip/negative results. The sigmoid is typically plotted in blue with orange theoretical overlay.

## Links
- Paper PDF: (local, paper/paper.pdf)
- GitHub: https://github.com/keltokhy/llm-global-games
