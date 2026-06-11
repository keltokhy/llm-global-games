# Pre-Submission Referee Report

**Paper**: The Intelligence Cost of Surveillance
**Authors**: Khaled Eltokhy (The Graduate Center, CUNY)
**Date**: 2026-06-11
**Review Standard**: REStud (Review of Economic Studies)

---

## Overall Assessment

The paper prices visible surveillance two-sidedly — chilling citizens and blinding the regime, with the blindness crisis-concentrated by a selection mechanism — and backs it with an unusually disciplined preregistered measurement program whose numerical spine verifies exactly against the replication archive. Its principal strength is the conception (the calibration trap and self-confirming over-monitoring are genuinely original) combined with exemplary internal discipline (a public preregistration failure on record, held-out confirmation, placebos, model-free replication). The single most critical issue is a cluster of four theory statements whose proofs do not deliver them — most notably Proposition 2 claims a monotonicity that neither follows from the proved bound nor matches the paper's own hump-shaped Figure 1 — compounded by an external-validity gap (all quantitative claims are simulator-internal, the abstract does not disclose the LLM instrument, and the headline h\*=0.26 travels without its modeling assumptions). For REStud specifically, the adversarial referee recommends desk rejection with an encouraging letter: the idea is REStud-grade, the execution is currently calibrated to APSR/AJPS/JEEA.

**Preliminary Recommendation**: Substantial revision required (for REStud); Revise before submitting (for APSR/AJPS/JEEA-tier targets).

---

## 1. Spelling, Grammar & Style

### Critical Issues (must fix before submission)
1. §2.3: "$W = L_C - L_R$" → $L_R$ never defined; should be $I_R$ (or rename pair consistently).
2. Proof of Prop 1: "attack mass is weakly **decreasing** in the informativeness of peer messages" → must be "increasing" for the logic chain to deliver chilling; as written the proof asserts the opposite of what (i) requires.
3. "one-fiftieth as much" (abstract, intro) vs "two orders of magnitude less" (§3.2, Fig. 1 caption) — 50× ≠ 100×; pick one everywhere.
4. §2.4 post-Prop-5 text uses $h$, $h^*$ three sections before they are defined.
5. "an euphemism treadmill" → "a euphemism treadmill"; parallel-structure break "could let … and lets".
6. §4.2 dangling modifier: "Combined with the companion paper's finding…, the blinding travels…" → recast with "Combining… I conclude…".

### Minor Issues (15)
"surveillor" → "surveiller"/recast; "hold a policy" → "retain"; "mis-designed" → "misdesigned"; ambiguous "analyst-model and corpus combination"; "incumbent model" phrasing; \citet possessive "Kuran (1991)'s"; §5.1 dose increments conflate unconditional (−6.3/−9.5pp) with conditional (~3pp) ranges; "retires the concern"; "\citet{yang2025}, who"; "bit-identical" vs "bitwise-matched"; "deterministic-shuffled" → "deterministically shuffled"; double "calm-state"; intro sentence three-appositives-deep (split); "as much as … as much" in abstract; model-section fragment.

### Style Patterns to Fix Throughout
(1) "pp" vs "percentage points" — spell out in prose, "pp" in tables only. (2) Define "strong/calm" = θ≥0 and "weak/crisis" = θ<0 once; one label thereafter. (3) Convert weaker colon-led verbless constructions to sentences. (4) Citation-as-author convention: pick paper-referent ("which") uniformly. (5) After fixing the 50×/100× claim, grep both phrases.

---

## 2. Internal Consistency & Cross-Reference Verification

**Verified against the replication archive:** all 14 ΔBrier estimates in Table 1 match analyst_results.json / v3_verdict.json exactly; 14 = 6+4+4; Opus reversal (+0.022, p=0.012) ✓; dose join fractions and Brier deltas ✓ (recomputed); h\*=0.26/0.15/0.08 ✓; 498 cells, coup 0.582/0.566, q̂ 0.69/0.59 ✓; Lemma-1 stats ✓; classifier 0.922/0.844/0.909 ✓; cross-sender ✓; codedness ✓; companion claims (−8.0pp, t=−16.2, two-fifths) ✓; all 28 citation keys resolve; all \ref targets exist.

### Critical Inconsistencies
1. §3.2/intro: "frontier analyst is the best model on every level metric (e.g., |ρ|=0.93, the highest)" — **false per archive**: Maverick 0.936 and Llama 3.3 0.934 > Opus 0.931; Opus is among the worst on level Brier. Underpins "blindness is in the data, not the reader" as stated. HIGH.
2. §3.3 "MAE 0.17–0.25" vs own Appendix Table 3 "0.17–0.40" vs JSON (0.166–0.401). HIGH.
3. Table 3 "−5.7 to −10.8 pp; 4/6 significant" — archive: **3/6** at p<0.05 (GLM 0.062, DeepSeek 0.080, Qwen 0.869 ns with +0.5pp outside the stated range). HIGH.
4. 50× vs 100× frontier-cost drift (abstract/intro vs §3.2/Fig 1). MEDIUM-HIGH.

### Cross-Reference Errors
None. Labels tab:decomp, tab:prereg, prop:selfconfirm never \ref'd from text; Discussion cites "Propositions 1–4," omitting Prop 5.

### Terminology Drift
$L_R$ undefined in the wedge definition (should be $I_R$). Otherwise clean.

### Minor Inconsistencies
Table 3 row 1 mixes full-sample baseline range with crisis deltas; classifier "double the calm gap" is ~2.7×; Fig 2 caption "75–85%" doesn't match per-analyst ratios (73/75/97%; pooled ~79%); text quotes join_fraction but figure plots join_fraction_valid (none/full differ at 3rd decimal; companion paper quotes valid); v1 gate "≥2/5" vs 6 pilot analysts (Opus outside v1 roster — say so); archive contains an unreported gpt-5.5 5-cell smoke; pilot Maverick calm placebo p=0.065 borderline given Opus reversal highlighted at p=0.012.

---

## 3. Unsupported Claims & Identification Integrity

### Causal Overclaiming (must address)
1. **Abstract never discloses the subjects are LLMs** and attributes mental states ("believe they are monitored"). Insert "LLM-agent citizens"/"LLM analyst models"; "are warned."
2. **h\*=0.26 travels without assumptions** (linear closure H=hq̂, pooled-analyst posterior, experimental G). §6.2's "with every input measured" is false — H is assumed, h free. Attach the conditional clause at every appearance (abstract, intro, §6.2).
3. Stasi opening's final sentence asserts the paper's conclusion as the historical cause of 1989 ("trained every East German… precisely no information"); staffing figures uncited. Recast as the paper's reading; cite (e.g., Gieseke).
4. Intro ¶2 asserts the finding as fact about monitored citizens generally, before evidence; historical "legibility" interpretation stated as fact. Add "the experiments below show…"; "arguably."
5. **"no reader can extract content that senders never encoded"** — impossibility claim; contradicted in degree by own evidence (retrained classifier recovers 0.844→0.909 vs 0.922 baseline ≈ 80% of the sender-level loss; decoding test restores two-fifths). State recovered fractions honestly; reserve "never encoded" for theory.
6. **"destroys information only where incriminating information existed"** — "only" contradicted by own codedness control (directional degradation, p=0.06); the style-vs-content inference rests on significant-vs-nonsignificant comparison (Gelman–Stern). Drop "only"; test the difference or hedge.
7. "Step functions of monitoring salience" — three hand-written warnings, no cardinal salience scale; cannot distinguish visibility step from steep concavity; dose curves rest on one sender family. Soften; state sender family.
8. "aimed… at exactly the information the monitor values most" asserted in intro ahead of theory/data; "exactly" overstates (concentration in tipping band).

### Generalization Issues
9. Prescriptive indicatives about real regimes ("optimal posture is the quiet one"; conclusion's "the bill exceeds the purchase"; abstract's tail claim) need "within the model, with measured inputs" at point of claim.
10. §6.2 footnote's historical pattern (covert-vs-overt practice of security states) asserted without citation.
11. English-language/single-culture scope condition for W≈0 never flagged (the one human steganography study cited found W>0 in Chinese).
12. Platform-cycle paragraph is repeated-game claims from a static model; mark explicitly as interpretation beyond the model.

### Missing Caveats
13. **Dependence of the 14 tests** (shared cells/corpora/architecture): add note — uniformity of direction, inferential weight on v3.
14. Pooled-analyst q̂ inherits shared miscalibration; add per-analyst/frontier sensitivity for h\*.
15. Alignment-confound sentence at point of claim in §5 (universal compliance is also RLHF-consistent).
16. "Universal compliance" from AUC is distributional, not per-sender; say "near-universal at the bin level" or report sender shares.

### Minor Language Issues
Unverifiable superlatives in ¶1 (keep one, hedged); "essentially no incumbent model" vs cited Avoyan (specify: the eavesdropper's inference problem is what is new); "no human dataset can deliver" → "no existing human dataset delivers"; Nepal body text asserts government's epistemic state ("reportedly outside the ban's scope"); "never reveal" → "need never reveal"; "same theorem" → "same logic"; cross-sender heterogeneity dressed as strength (one clause); **underconfidence**: v2 failure is a decomposition finding (promote), Opus calm reversal is a risky prediction the theory survived (promote from table note), codedness control is underpowered (own it).

---

## 4. Mathematics, Equations & Notation

### Mathematical Errors
1. **Prop 2: proof proves a decreasing *bound*, not a decreasing *gap*; the stated monotonicity contradicts the paper's own hump-shaped Figure 1** (gap small in hopeless states). Restate as envelope result: gap zero outside censored event, bounded by R·Φ((x̄−θ)/σ), vanishing as θ→∞ — the bound *permits* the hump; the current statement forbids it.
2. **Lemma 1(a): O(1/n) deviation value is wrong** — one message read by n−1 peers can flip the aggregate outcome (O(1) value), and the whole paper depends on messages being first-order influential. Also "dominant" is the wrong solution concept. State for the continuum (trivial) or hypothesize κ > v̄_n with compliance as equilibrium.
3. **Prop 4 proof smuggles its premise**: "censored traffic uninformative beyond the (unidentified) censoring event" is false under rational reading — bunching mass on [x̄, x̄+δ] identifies the censored fraction and hence θ. Keep shrinkage as a maintained assumption (motivated by measured 0.69→0.59) or impose F̄ = truthful conditional density (true camouflage) with care.
4. **Prop 5: "estimate converges to zero" is a non sequitur** — the blinding cost is *unidentified* on path; posterior stays at prior. Needs either small-prior assumption or plug-in estimator framing: "any initial belief that it is small survives all evidence."
5. Corollary 1's "bounded away from zero on {θ<0}" never established and likely false (unbounded below; gap shrinks in hopeless states). Restate on a compact crisis band with an assumed lower bound.

### Notation Inconsistencies
$L_R$ vs $I_R$; δ double-booked (support width vs Dirac); ε vs ϵ; q̂(m,θ) is a random variable written as a function; H's argument changes between model (I_R) and exhibit (h·q̂ — different object; one sentence acknowledging the replacement); continuum vs finite-n never reconciled (n unintroduced); prior G used in eq.(1) but introduced only in Cor. 1/Prop 4; h, h* used before definition; \bar F/\bar{F} mixing.

### Undefined Notation
Φ; μ_i (message symbol); F̄ never characterized beyond support (indistinguishability claim needs F̄ to match the truthful conditional law); R cleaner in statement; M_m; B,C defined but orphaned in formal results; no stray "s"; I_θ/I_A/I_i unused as symbols (consistent).

### Proof Gaps
1. **Prop 1(i): the global-game comparative static is asserted, not proved — and is famously non-monotone** (more informative peer messages can *raise* attack in weak states; the paper's own platform-cycle narrative depends on it). Proof retreats to companion-paper empirics — a referee will strike an experiment cited inside a proof. Options: restate for crisis region with a likelihood-ratio argument; state as assumption-plus-measurement; or cite a result that delivers it.
2. Prop 1(ii) strictness needs a decision-relevance hypothesis.
3. Lemma 1(b) localization: per-message vs n-message bound (R·[1−(1−Φ)^n]); θ-conditional risk vs Bayesian value (one extra line); prose upgrades bound to proportionality.
4. Lemma 1(b) kernel correct but silently selects the truthful m=0 equilibrium — state the selection.
5. Prop 3 algebra verified ✓; the verbal comparative statics inside the proposition have no proof — demote to text or prove the limits.
6. Prop 4(ii): "strictly lowers q̂ on the event" holds in expectation (martingale argument: E[q̂|success]=E[q̂²]/E[q̂]), not realization-by-realization; "successes concentrate on crisis region" asserted.
7. Prop 5 has no learning model for "converges" to bind to (no time index, observables, updating rule); F–L (1993) apt in spirit; cleaner cites: Rothschild 1974, Easley–Kiefer 1988, Sargent/Cho–Kasa; either add a 3-line repeated environment or restate as Remark.
8. "Under Assumption 1 and Lemma 1" — condition on hypotheses, not results.

### LaTeX Math Formatting
Eq refs clean; K∘P notation nonstandard (kernels act right); cases-display defines a kernel as a function (define c as kernel); indicator style; interpretive clause inside Corollary 1; **498 vs 500 cells unexplained** (footnote the 2 dropped cells).

---

## 5. Tables, Figures & Documentation

### Tables
- **Table 1**: add crisis-N per wave (only v3's 181 appears, in text); define the Brier outcome in notes; state placebo test/level and calm-N; explain roster changes across waves (6→4); "(reversed)" breaks checkmark semantics — use ✓† with dagger note; unify p-value convention.
- **Table 2 (prereg)**: no notes block at all — add gate logic, levels, Ns per stage; add registry identifiers/commit hashes + dates; expand caption.
- **Table 3 (decomp)**: "Surveillance" column mixes levels, deltas, and qualitative entries — split or rename; codedness row sits under "Surveillance" though it is explicitly not surveillance; verdict conventions (destroyed/degraded/survives) undefined; N per block; decimal harmonization.

### Figures
- **Fig 1**: define both panels' error bars/bands (method, level, clustering); name the four pooled analysts; flag the (a)=held-out vs (b)=pilot sample switch rationale; **panel (b) two-line x-tick labels collide at print size** — main production defect.
- **Fig 2 (lemma1)**: bootstrap unit and draws unstated; AUC series has no CIs (declare); dotted 0.5 line unexplained; remove editorializing suptitle; **in-figure "p=0.000" must become "p<0.001"**; bin definitions and per-bin N unstated.
- **Fig 3 (dose)**: **uses default matplotlib palette while all others use the paper's purple/gray** — restyle; CI methods unstated; crisis-N per dose unstated; caption quote of mild warning is abridged without saying so; plotted values use join_fraction_valid vs text's join_fraction (reconcile).
- **Fig 4 (survival)**: **remove the in-figure editorializing title**; no uncertainty at all on lines/crossings — add bands or a note; explain crossing dots; 498-vs-500 footnote; name the four analysts in notes.

### Cross-Reference Issues
All figures referenced ✓. Table 2 and Table 3 never cited by number — cite directly. The 50×/100× drift also appears across caption vs abstract.

### Formatting Inconsistencies
Palette (Fig 3); title conventions (editorial suptitles on Figs 2,4 vs none on 1,3); decimal places per metric; p-value conventions incl. in-figure "p=0.000".

---

## 6. Contribution & Referee Assessment

### Part 1 — Central Contribution
Two-sided pricing of visible surveillance with crisis-concentrated blinding via selection, the calibration trap, and self-confirming over-monitoring. Genuinely new relative to Yang 2025 (missingness vs re-encoding), Gehlbach et al. 2026 (no communication stage), Braghieri 2024 (no listener), Edmond/Inostroza–Pavan (vertical vs horizontal). The wedge W and Prop 5 are the most original pieces. **Rating: Significant (conception); execution not yet at the REStud bar** — formal results are near-immediate consequences of an assumed message technology; the empirical world is a simulator.

### Part 2 — Identification and Credibility
Internal validity exemplary (prereg with public failure, held-out, placebos, model-free classifier, cross-family corpus). **The deep problem is the estimand**: a property of the simulator, not a population. The external-anchors paragraph concedes that every *link* is human-documented but every *quantitative claim* (26%, 0.69→0.59, two-thirds) is Llama-only. Vulnerabilities: alignment-training confound (guarded register = RLHF caution; needs a base-model sender arm); single sender family for the dose-response and h\*; LLMs-reading-LLMs (defused partly by TF-IDF); 14/14 overcounts (effectively ~2 independent programs); English-only; ground truth itself LLM-generated. Preregistration rescues internal, not external, credibility.

### Part 3 — Analyses: Required and Suggested
**Required (blockers):**
1. Human evidence on ≥1 link in-pipeline (human readers on the matched corpora, or human senders under incrimination payoffs read by the analyst stack).
2. Dependence-robust inference (joint test per wave; retire "fourteen of fourteen" from abstract/intro).
3. **Vary the incrimination scope x̄** — the model's signature, falsifiable prediction (the blindness should move with what counts as incriminating); severity was varied, scope never.
4. Survival-exhibit robustness: h\* bands under concave/convex/threshold H, per-analyst posteriors, family of G's; state h\* is simulator-internal.
5. Equilibrium closure of the citizen side: characterize the threshold equilibrium under the garbling or restate Prop 1(i) as empirics; address the t5 feedback (anticipated repression should enter join payoffs).

**Suggested:** base/non-RLHF sender arm; sender heterogeneity for the dose curves; real-world text exhibit around a surveillance-salience shock; non-English sender arm; shared-key steganography arm (W>0 attainable? distinguishes equilibrium from capability limit); in-context retraining analog for LLM analysts.

### Part 4 — Literature Positioning
Missing and conspicuous: **Lohmann 1994** (Leipzig cascades — glaring given the opening), **Kuran 1995 book**, **Scott 1985/1990 hidden transcripts** (the named concept for the register), **Hassanpour 2014**, **Sullivan AJPS 2016** + **Davenport** (the H(·) margin and covert-vs-overt footnote), **Bursztyn et al.** misperceived norms, experimental communication-coordination (Blume–Ortmann etc.). Best framing: the dictator's dilemma, microfounded — and Prop 5 should be more prominent.

### Part 5 — Journal Fit and Recommendation
**Desk reject at REStud, with an encouraging letter** — idea is REStud-grade; theory leg and evidence leg are each individually below the bar and do not cross-subsidize. Path back: endogenize the coding strategy (sender–receiver–eavesdropper signaling game where camouflage pooling and W≈0 *emerge*; visibility as signaling), one human arm in-pipeline, dependence-honest inference, x̄-variation. **Best realistic alternatives: APSR/AJPS (where it lands hardest), JEEA, Economic Journal, AEJ: Micro (if theory deepened), J. Public Econ.**

### Part 6 — Questions to the Authors
1. Would a base (non-instruction-tuned) sender produce the guarded register, or is the treatment manipulating RLHF compliance rather than incentives?
2. What is h\*=0.26 a number about? Defend it as an economic quantity; report ranges across H shapes, analysts, G's.
3. Characterize the equilibrium of the communication game under the garbling, or relabel Prop 1(i); why don't join payoffs depend on H?
4. Is W≈0 an equilibrium prediction or a capability limit of one-shot LLM senders with no shared convention? A shared-key arm would distinguish.
5. What is the effective number of independent confirmations among the 14, and what does a joint test give?
6. Lemma 1(b)'s localization is the signature prediction: move x̄ (warning *scope*) and the blindness should move. Why was only severity varied?
7. Analyst Brier is scored against LLM-citizen-generated outcomes; how much of crisis-concentration survives if the θ→outcome mapping is misspecified?

---

## Priority Action Items

**CRITICAL** (could cause desk rejection or major referee objections):
1. Fix the four theory defects: restate Prop 2 as an envelope/localization result (the current claim contradicts the paper's own Figure 1); replace Prop 1(i)'s "proof" (assert for crisis region with a real argument, or restate as assumption-plus-measurement — no experiments cited inside proofs); make Prop 4's shrinkage a maintained assumption (the "derivation" is false under rational bunching inference); fix Lemma 1(a)'s O(1/n) claim and solution concept.
2. Disclose the LLM instrument in the abstract; attach the modeling assumptions to h\*=0.26 at every appearance; change §6.2's "every input measured."
3. Correct the archive-contradicted claims: "best model on every level metric" (false — Opus is 3rd on |ρ|, worst-tier on Brier), "4/6 significant" (it is 3/6), MAE "0.17–0.25" (it is 0.17–0.40), and the 50×/100× cost drift.
4. Retire "fourteen of fourteen" as an independence claim; add a dependence note and a per-wave joint test.
5. Decide the target: for REStud, the required analyses (human arm in-pipeline, endogenized coding game, x̄-scope variation) are blockers; for APSR/AJPS/JEEA, items 1–4 plus the claim-discipline pass may suffice.

**MAJOR** (will be raised by referees):
6. Notation/undefined-symbol sweep: $L_R\to I_R$; δ double-booking; ε/ϵ; introduce G and n; formalize q̂; acknowledge the H(I_R)→h·q̂ replacement; define Φ, μ_i; characterize F̄ or soften "indistinguishable"; 498-vs-500 footnote.
7. Survival-exhibit robustness (H shapes, per-analyst q̂, G family) + uncertainty on Figure 4.
8. Claim-discipline pass per §3 above (Stasi declaratives, "no reader can," "only where," step-function language, prescriptive indicatives, Nepal phrasing) — and promote the three under-claimed results (v2 failure as decomposition finding; Opus calm-state reversal as a successful risky prediction; codedness control as underpowered rather than null).
9. Figure/table production: Fig 3 palette; remove editorial in-figure titles; fix "p=0.000"; Fig 1(b) label collision; CI definitions everywhere; Table 1/2/3 notes (Brier definition, crisis Ns, roster changes, registry IDs, column semantics).
10. Literature: add Lohmann 1994, Kuran 1995, Scott, Hassanpour, Sullivan, Davenport, Bursztyn et al., experimental communication-coordination.

**MINOR** (polish):
11. Agent 1's six critical prose fixes and fifteen minor items; pp/percentage-points convention; calm/crisis terminology defined once; colon-led constructions; citation-referent convention.
