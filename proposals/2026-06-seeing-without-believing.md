# Seeing Without Believing: Synthetic Media and the Collapse of Common Knowledge

*Proposal draft — June 2026. JMP candidate following "Speaking in Code."*

## Pitch

Morris and Shin (2002) showed that public signals punch far above their informational weight in coordination because they generate common knowledge: everyone knows that everyone saw the same thing. Synthetic media attacks exactly that property. When any video might be fabricated, the first casualty is not belief in lies — it is the coordinating power of *true* evidence, because each citizen doubts that others will treat it as credible. The liar's dividend (the regime crying "deepfake" at authentic footage) then emerges endogenously as the cheapest counter-coordination technology ever invented: it works even if no fake is ever produced. Headline: **synthetic media doesn't have to fool anyone to demobilize everyone.**

Paper 1 showed surveillance strips the second-order content of *private* peer messages. This paper shows authenticity doubt strips the second-order content of *public* signals. Same agenda, bigger object: the claim applies to any institution that runs on public evidence — protests, elections, courts, central-bank communication — not only to autocracies.

## Theoretical framework

**Primitives.** Regime-change global game: citizens $i \in [0,1]$, strength $\theta$, private signals $x_i = \theta + \varepsilon_i$, plus a public signal $y$ (a circulating video/document).

**Synthetic-media technology.** $y$ is authentic ($y = \theta + \nu$, $\nu \sim N(0, \tau^2)$) with probability $1-q$; with probability $q$ it is fabricated, drawn from an adversary-chosen distribution that mimics the authentic one (Denter–Ginzburg-style mimicry).

**Four channels, in increasing order of novelty:**

1. *Informativeness (mechanical).* With common knowledge of $q$, Bayesian weight on $y$ falls — a mixture-updating result. Necessary baseline, not the contribution.
2. *Clarity/commonality (core contribution).* Citizens receive private authenticity assessments $a_i$ (forensic skill, source trust, heterogeneous priors $q_i$). The *interpretation* of $y$ becomes dispersed: a common underlying signal read with idiosyncratic error. This maps exactly onto Myatt–Wallace (ReStud 2012) signal *clarity*: synthetic media converts a clear (public) signal into a murky (quasi-private) one **holding each individual's accuracy fixed**. Coordination decays in interpretation dispersion even at constant average informativeness. Experimental signature: actions and second-order beliefs move while first-order posteriors stay flat — precisely the wedge paper 1's restricted-elicitation machinery detects.
3. *Liar's dividend (endogenous).* After damaging authentic $y$, the regime can issue a denial ("that's a deepfake") at low or zero cost. In equilibrium the regime denies both real and fake damaging signals (pooling), so denials move authenticity posteriors by Bayes' rule, and denial effectiveness is *increasing in ambient $q$*. Target result: $\partial(\text{attack})/\partial q < 0$ even on paths where no fake is ever produced — **the liar's dividend is paid off-equilibrium.** Ambient synthetic-media prevalence subsidizes every authoritarian denial.
4. *Status-quo asymmetry.* Attacking requires coordination; surviving does not. Authenticity doubt therefore lowers attack incidence at every $\theta$ — epistemic pollution structurally favors incumbents, even when both sides can fabricate. Doubt is not symmetric ammunition: challengers need common belief, defenders need only dissensus.

**Theory anchors.** Morris–Shin (AER 2002) — publicity multiplier. Myatt–Wallace (ReStud 2012) — accuracy vs. clarity; the model's formal backbone. Morris–Shin–Yildiz (JET 2016) — rank beliefs as the language for "mutual legibility." Rubinstein (AER 1989) email game; Chwe, *Rational Ritual* — fragility of almost-common knowledge. Angeletos–Werning (AER 2006) — endogenous public information in crises. Edmond (ReStud 2013) — regime manipulation of signals. Kamenica–Gentzkow (AER 2011).

**Must-cite-and-differentiate.**
- **Denter–Ginzburg, "Troll Farms" (2024, rev. May 2026).** Nearest formal neighbor: fabricated messages mimicking genuine signals — but elections and first-order dilution, no coordination game, no common-knowledge channel, no denial stage.
- **Ui, "Strategic Ambiguity in Global Games" (2023).** Ambiguity about signal quality in coordination — Knightian, no authenticity structure, no adversary.
- **Schiff–Schiff–Bueno, "The Liar's Dividend" (APSR 2025).** The experimental fact (denials work, via informational uncertainty) — measured on candidate support, never on coordination. This paper formalizes their mechanism and embeds it where it matters most.
- **Altay–Gilardi (PNAS Nexus 2024).** "AI-generated" labels tax *true* headlines — the per-message first stage exists in humans.
- **Acemoglu–Ozdaglar–Siderius (ReStud 2024).** Misinformation in equilibrium sharing — networks, not coordination.
- Scoop status (verified June 10, 2026): no formal model of authenticity-uncertain public signals in a coordination game; no collective-action experiment with possibly-synthetic signals. The lane is open but theorist-attractive — stake it early.

## Experimental design

Receiver-side extension of the existing pipeline: a **broadcast layer**. All 25 agents in a cell see the same public item (e.g., "a video circulating tonight shows an elite army unit refusing orders in the capital"), rendered in the briefing register.

**Arms.**
1. Private-only baseline (exists).
2. \+ Authentic broadcast, no doubt manipulation.
3. \+ Broadcast under ambient doubt: context states forensic estimates that a share $q \in \{\text{low}, \text{med}, \text{high}\}$ of circulating clips are synthetic.
4. Heterogeneous verification: fraction $f$ of agents additionally receive "independent forensic verification: authentic"; vary $f$. Holds the broadcast and individual content fixed, varies the *distribution* of interpretations.
5. **Publicity manipulation (the crucial arm):** same verification, but framed either "this verification was distributed to all citizens" or "this verification reached you alone." Identical first-order content; pure second-order variation. This is the cleanest test in the paper — the global-games analogue of Chwe's ritual.
6. Liar's dividend: regime denial appended after the authentic broadcast, crossed with ambient $q$ from arm 3.

**Outcomes.** Join rates; first-order beliefs (P(regime falls), restricted elicitation); second-order beliefs (expected join share, restricted elicitation); with the communication layer on, whether peer messages *cite the broadcast as common ground* (a text-level "rational ritual" measure — does the public signal appear as a shared reference point, and does doubt kill that?).

**Predicted signatures.**
- Arm 3 vs. 2: joining falls by more than first-order beliefs move (multiplier loss).
- Arm 4: joining increasing and convex in $f$ (threshold flavor).
- Arm 5: "everyone received this" $>$ "you alone received this," with identical individual information — the publicity effect in isolation.
- Arm 6: denial neutralizes authentic evidence more effectively at higher ambient $q$; at $q=0$ denials are inert.

**Human anchor — zero-budget ladder (solo researcher, no participant funds).** The silicon + theory package is the JMP; the human anchor is built in tiers, each free or nearly free, and "human arm registered and in progress" is a perfectly normal state for a JMP talk:

- *Tier 0 — published human data (free, do first).* Calibrate the first stages against existing archives rather than new subjects: Schiff–Schiff–Bueno's liar's-dividend experiments are deposited in the APSR Dataverse (denial effects on belief in true stories = the arm-6 first stage); Altay–Gilardi's data (PNAS Nexus, OSF) gives the provenance-label tax on true content (arm-3 first stage); the human global-games lab literature (Heinemann–Nagel–Ockenfels and successors) anchors baseline threshold behavior. This is exactly the Ludwig–Mullainathan–Rambachan validation move — against data someone else already paid to collect.
- *Tier 1 — classroom experiment (≈ free).* The one genuinely new human contrast the paper needs is arm 5 (publicity framing). A simplified threshold game runs in oTree in CUNY sections in a single class period; unpaid or lottery-incentivized classroom designs are standard and IRB-cheap (file the protocol early — filing costs nothing). Even n ≈ 80–120 students on the arm-5 contrast is a real anchor.
- *Tier 2 — small grants sized exactly for this (apply in parallel, costs only time).* CUNY GC doctoral student research grants; IFREE small grants (experimental economics, funds graduate students); Russell Sage behavioral-econ small grants; APSA small research grants; Emergent Ventures (fast decisions, favors AI-and-institutions topics); NSF dissertation improvement grants if the program is running. Any one of these funds a Prolific PPI anchor (Broska et al. 2025) — and PPI means a few hundred human subjects anchor the entire LLM fleet, so the ask is small.

Preregister the LLM arms regardless; cross-model multiverse on the LLM side as in paper 1.

**Cost.** Core arms are receiver-side only (no communication layer required), so token costs are modest and within a solo budget — paper 1 is the existence proof. New human-subject spending: \$0 required for the JMP version; Tier 2 money upgrades the publication version.

## Risks and fallbacks

1. *"Isn't this just Myatt–Wallace re-skinned?"* The clarity mapping is the backbone, not the contribution. The contributions are the endogenous denial result (liar's dividend off-equilibrium), the status-quo asymmetry, and the experiment. If a theorist referee pushes, channels 3–4 carry the paper.
2. *Scoop risk.* Highest of the candidate ideas — misinformation theory is crowded-adjacent and the coordination angle is findable. Mitigation: 2-page model note this month, arXiv working paper with the LLM results by winter, circulate at seminars before the human arm completes.
3. *First-stage failure: LLM agents may be miscalibrated about authenticity cues* (too credulous or too cynical). Pilot arm 3's belief first-stage before building out; if the LLM first-stage is weird, recalibrate it against the Tier-0 published human moments and let the classroom anchor carry the headline contrast, with the LLM fleet as the mechanism lab.
4. *Arm 5 could fail in LLMs* (publicity framing too subtle). It works in humans in adjacent designs (Chwe-style lab games); pilot early since it is the make-or-break exhibit.

## Milestones

1. **Week 1–2:** Two-page model note — verify the Myatt–Wallace mapping delivers the multiplier-decay comparative static cleanly; sketch the denial pooling equilibrium.
2. **Week 2–4:** Broadcast-layer prototype + first-stage pilot (arm 3 beliefs; arm 5 framing check).
3. **Month 2–3:** Full LLM design surface + prereg; theory draft.
4. **Month 4–6:** Tier-0 calibration against published archives; classroom session (IRB filed in month 1 — filing is free); grant applications out (Tier 2).
5. **Winter 2026–27:** Working paper on arXiv/SSRN; seminar circuit.
