# Pre-Submission Referee Report

**Paper**: Speaking in Code: How Surveillance Suppresses Coordinated Dissent
**Authors**: Khaled Eltokhy (The Graduate Center, CUNY)
**Date**: 2026-06-09
**Review Standard**: JEBO (Journal of Economic Behavior & Organization)

---

## Overall Assessment

The paper embeds LLM agents in a Morris–Shin regime-change game and uses the statelessness of LLM calls to apply surveillance only at the message-writing stage, isolating a "communication poisoning" channel that neither field data nor human labs can implement cleanly — this design is the paper's genuine and substantial strength, and the falsification/placebo apparatus is unusually disciplined. The single most critical issue is that the headline evidence rests on one retired creative-writing finetune (Mistral Small Creative), with cross-model replication reduced to 20 matched cells in 3 of 5 models, and the central mechanism claim (codedness → reduced joining) is never separated from generic message degradation — the paper's own degraded-messages control (−21.4 pp) exceeds the surveillance effect itself. Additionally, Edmond (2013) and Avoyan (2020) — the closest theory and experimental papers, both already in references.bib — are never cited in the text, which a referee will read as not knowing the closest work.

**Preliminary Recommendation**: **Revise before submitting** — not desk-reject material, but submitting now would produce predictable major-revision demands that can be pre-empted.

---

## 1. Spelling, Grammar & Style

### Critical Issues (must fix before submission)

1. **paper.tex, lines 99, 108, 113, 190, 253, 485, 494, 511, 597 | spaced hyphen " - " used as a parenthetical dash** → use em dashes (`---`) or spaced en dashes, consistently | The document mixes three conventions: `---` (line 359, tab_regressions notes), `--` for ranges (correct), and bare " - " as a dash in at least nine prose locations.

2. **paper.tex, line 165 | "it is the evidence that $\mu_S$ contains less public, less action-cueing messages than $\mu_0$"** → "…that the messages in $\mu_S$ are less public and less action-cueing than those in $\mu_0$" | As written, "less … messages" reads as a count-noun quantifier error ("fewer messages").

3. **paper.tex, line 105 | "\PartOneNModels\ models spanning \PartOneNFamilies\ architecture families are embedded in the \citet{morris2003} game"** → "I embed \PartOneNModels\ models … in the \citet{morris2003} game" | The macro renders as the digit "7", so the sentence begins with a numeral; the rewrite also removes an unneeded passive.

4. **tables/tab_prompt_isolation.tex | "$p$" column reports "0.000"** → "$<$0.001" | A p-value is never exactly zero. (The prose correctly uses "<0.001" macros.)

5. **tables/tab_finite_n.tex, caption | Title Case + capital $N$** → sentence case, lowercase $n$ | Clashes with every other caption and the paper's stated convention (line 190); tab_regressions has the same Title Case problem.

6. **Model-name inconsistency across tables** | "Mistral"/"Mistral Small Creative", "Qwen 30B"/"Qwen3 30B", "Trinity"/"Trinity Large", "MiniMax"/"MiniMax M2-Her" vary across tables. Standardize (full name at first table, one consistent short form thereafter).

7. **tab_hypotheses.tex and tab_slider_audit.tex refer to "Part I"/"Part II"** | Never defined anywhere in paper.tex (the design section says "two phases"); a reader cannot map these labels to anything.

8. **Orphaned table files: tab_slider_audit.tex and tab_temperature.tex are not `\input` anywhere** | tab_temperature is superseded; tab_slider_audit still refers to archived information-design conditions. Delete from the submission package or re-link.

9. **paper.tex, line 274 | heading "Decomposition: informational versus coordination components"** → e.g., "Task contrast: private-bet versus coordination decisions" | The paragraph itself disclaims that this is "not a point-identified decomposition." The heading promises what the text retracts.

10. **paper.tex, line 521 | "contaminated-communication interpretation"** → "communication-poisoning interpretation" | The paper's established term (line 257) appears everywhere else.

### Minor Issues

1. paper.tex line 99 | "above the threshold citizens stay, below it they join" → semicolon | Comma splice.
2. paper.tex line 101 | "Field data … confounds" → "confound" (or recast) | data plural.
3. paper.tex line 103 | colon-into-double-question construction forces a re-read; split into two sentences.
4. paper.tex line 105 | "decision rule environment" is an unparseable noun stack → "decision rule".
5. paper.tex line 117 | \citet{larooij2026, larooij2025b} → chronological order.
6. paper.tex line 161 | "My experiment" vs "The experiment" — pick one register.
7. paper.tex line 178 | "what is external is the comparison itself" requires re-reading; recast.
8. paper.tex line 181 (fig:pipeline caption) | "8 evidence domains and 3 latent sliders" → spell out, matching prose elsewhere.
9. paper.tex lines 207/103 | "this paper claims/asks" → "I claim/ask" (paper is otherwise first-person singular).
10. paper.tex line 232 | "Matched-cell comparison … yields" → "A matched-cell comparison" | missing article.
11. paper.tex line 235 | "peer messages carry real information, noisily" → "real, if noisy, information".
12. paper.tex line 253 | "is intentionally treated as" → "I treat … as".
13. paper.tex line 255 | stranded "are aimed at" after a four-item subject list; recast with em dash.
14. paper.tex line 259 | bare appositive comma before "whether" invites misreading; use colon/em dash.
15. paper.tex line 259 | "I interpret mutual legibility as the most consistent reading of the text evidence" → "I read the text evidence as most consistent with a loss of mutual legibility" | current sentence says the opposite of what is meant.
16. paper.tex line 263 | "(to X from Y)" → "(from Y to X)".
17. paper.tex line 270 | "was not collected" → "could not be collected" (matches footnote constraint).
18. paper.tex lines 278/445 | "reject pure writer-reader same-family recognition" — five-noun pile-up; recast.
19. paper.tex lines 207, 289 | "BNE" never formally introduced as acronym at first spelled-out use (line 145).
20. paper.tex line 381 | "The primary belief sample includes the private briefing" → "The primary belief elicitation prompt includes…".
21. paper.tex line 423 | "locate which text surface changes" → "identify".
22. paper.tex line 435 | raw config flag `decision_context=none` leaking into an appendix table; define or cut.
23. paper.tex line 477 | "stable to agent count" → "robust to agent count".
24. paper.tex line 481 | "exact like-for-like" is tautological.
25. paper.tex line 498 | "about $r \geq$ … when rounded" — triple hedging; simplify.
26. Abstract line 80 | appositive "a restricted elicitation…" should be set off with em dashes.
27. tab_comm_estimators caption | "Why the communication estimators differ." — conversational fragment.
28. tab_cross_generator | caption and notes repeat the same two sentences nearly verbatim.
29. tab_msg_features notes | "2268" → "2,268" (separator consistency).
30. tab_surveillance_variants notes | "from two-sample $t$-test" → "from a two-sample $t$-test".
31. tab_parse_errors notes | "5 of 7" → "five of seven"; consider naming GPT-OSS explicitly alongside Trinity.
32. tab_beliefs | "Mean belief" 0.444 vs 0–100 scale elsewhere — one scale.
33. tab_hypotheses notes | trailing space cosmetic issue.
34. paper.tex lines 549–555 | bare "-" in empty header cells → "---".

### Style Patterns to Fix Throughout

1. **Dashes**: replace every spaced " - " used parenthetically with "---"; keep "--" only for ranges/name pairs.
2. **En dashes in compounds**: "country--period" vs "country-period" inconsistent; "writer-reader" should be "writer--reader".
3. **"$z$-score" vs "z-score"**: standardize on "$z$-score" (11 roman instances).
4. **First-person consistency**: "I" in all analytic prose.
5. **Numerals below ten**: spell out in running prose and captions; digit ratios (5/5) fine.
6. **Caption capitalization**: sentence case everywhere (fix tab_finite_n, tab_regressions).
7. **Passive where the agent matters**: "are embedded" → "I embed", etc.
8. **Hedging stacks**: recurring not-X-not-Y-not-Z codas (lines 253, 275, 498); one disclaimer per claim.

No misspellings found; proper nouns verified; no banned filler words; "significant" used only statistically in authorial prose.

---

## 2. Internal Consistency & Cross-Reference Verification

Direction, magnitude, and sample claims for the four headline results (threshold alignment r=+0.80, communication +2.10 pp, surveillance −13.9 pp primary / −11.2 pp equal-weight 5/5 models, stable SOB 31.2 vs 30.9) are consistent across abstract, short abstract, intro, results, conclusion, and tables.

### Critical Inconsistencies (highest-risk near-misses)

1. **[tab:belief_factorial Δ rows ↔ its own cell means]** (paper.tex:385–407) | Δ rows do not reproduce from displayed cell means: Δ belief (excluded) −0.4 vs cells 46.0→45.7 = −0.3; Δ belief (included) −5.7 vs 53.2→47.4 = −5.8; Δ SOB (excluded) −0.1 vs 32.7→32.7 = 0.0; Δ join (included) −11.4 vs 42.2→30.9 = −11.3. Rounding of unrounded values, but a referee will compute these. | MODERATE
2. **[paper.tex:521 ↔ tab:punishment_risk]** | Text asserts differences "$< 0.2$ points" but the table's Mistral-Pure row shows 8.0 vs 8.2 — exactly 0.2. Change to "$\leq$" or report unrounded. | MODERATE
3. **[tab:models ↔ tab:comm_estimators]** | Communication arm totals disagree by one row (Trinity 100 vs 100/99 valid; pooled 2,600 vs 2,599). The 1-row gap (Trinity API failure) is unexplained in tab:models notes. | MODERATE
4. **[§5 coded-metaphor delta]** (paper.tex:261) | 69.8% → 94.1% displayed with Δ = +24.2 pp; endpoints give +24.3. | MINOR–MODERATE
5. **[Abstract ↔ tab:prompt_isolation]** | Abstract calls −11.2 pp the "matched-cell mean Δ"; the table labels −11.2 "Equal-weight avg" and shows a different "Pooled matched" estimate of −13.5 pp that appears nowhere in prose. Align the abstract's wording with §5's "equal-weighted mean". | MODERATE

### Cross-Reference Errors

All `\ref` targets resolve; all 15 `\includegraphics` targets exist; zero dangling references to the removed propaganda treatment.

1. `tab:logistic_params` | input at paper.tex:503, never referenced in text.
2. `tab:bc_statics` | input at paper.tex:574, never referenced in text.
3. `fig:communication` (fig05) | defined, never referenced — the Communication result never points to its own figure.
4. `fig:text_baseline` (fig15) | defined, never referenced; macro `\TextBaselineR` defined but unused.
5. `tab_temperature.tex`, `tab_slider_audit.tex` | dead files, never \input; tab_slider_audit still references archived designs.
6. `tab:hypotheses` notes say "pooled Part I data" — "Part I" never defined in the paper.

### Terminology Drift

1. **Matching key** | Main text: "(model, country, period, θ)"; table notes: "(model, country, period, θ, z, benefit, θ*)". State the full key once, reference it.
2. **"matched-cell" / "matched common-support" / "paired"** | three names for one estimator; pick one.
3. **Primary model naming** | "Mistral" / "Mistral Small" / "Mistral Small Creative" / "the primary model".
4. **Result 2** | "mean *within-country* correlation" qualifier appears only at paper.tex:212; elsewhere the quantity is unqualified. Verify same statistic and harmonize (see also Agent 4, Mathematical Errors #1 — they are NOT the same estimand).
5. **"clean surveillance treatment" vs "clean prompt-isolation rerun"** | both name the same arm; one-line equivalence at first use.

### Minor Inconsistencies

1. Temperature set phrasing (paper.tex:489) implies 15 combos; \TempNCombos = 13 (Mistral ran only 3 temperatures).
2. tab:finite_n Mistral N = 600 unexplained (pure sample is 1,000 rows / 680 unique cells everywhere else).
3. Stale comment in stats_macros.tex:54 ("join_fraction ~ theta" — code regresses J on A(θ)).
4. \BFSurvDeltaBelMsgP stores a threshold for one cell and a point value for another — fragile on regeneration.
5. Bibliography: every in-text key resolves (0 missing). 18 bib entries never cited — including edmond2013, avoyan2020, guriev2019, morris2002, shurchkov2013 (see Agent 6: edmond2013 and avoyan2020 must be cited, not pruned).
6. Unused labels: sec:communication, sec:generator_robustness, sec:identification_appendix, sec:parse_errors, sec:punishment_risk, sec:stakes_appendix (harmless).
7. Stakes-ladder weather placebo moves joining 8–14 pp from an "irrelevant" header; never remarked on — a referee may ask.
8. $\bar{z}_c \sim \mathcal{N}(0, 0.3)$ (paper.tex:581) ambiguous between SD and variance next to $\mathcal{N}(0, 0.05^2)$.

Verified clean: all prompt-isolation numbers, equal-weight average recomputation, comm-estimator cell accounting, tab_models totals, H1–H5 stats, classifier counts, slider values, payoff rows B = θ*/(1−θ*), no-message range, parse-error claims, abstract/conclusion agreement.

---

## 3. Unsupported Claims & Identification Integrity

The body has clearly been through claim-discipline passes; remaining problems concentrate in (a) the title and abstracts, which strip the hedges the body maintains; (b) mechanism language naming publicness/actionability/codedness as the operative channel when the identification map marks the fear-salience alternative "Inconclusive"; (c) missing LLM-specific caveats and an unreported country-level clustering problem.

### Causal Overclaiming (must address)

1. **[Short abstract]** "The mechanism is linguistic: surveilled messages become coded…" | The paper itself (§5:261) disclaims codedness as a randomized mechanism; no mediation links codedness to reduced joining; fear-tone is unseparated. | Fix: "consistent with — though not isolating — a loss of coordination-cueing content."
2. **[Short abstract]** "reduces joining in all prompt-isolation models while stated expectations remain stable" | Drops (i) only 4/5 individually significant, Qwen3 235B −3.0 pp p=0.186 on 20 cells; (ii) stability holds only in the messages-excluded elicitation — SOB falls −8.2 pp (p<0.001) with messages included. | Fix: add both qualifiers.
3. **[§1:108]** "largely non-persuasive" | Rests on the belief-stability result the same paragraph calls "instrument-sensitive"; under the other instrument the channel looks substantially persuasive. | Fix: delete or replace with the prompt-architecture statement.
4. **[§5:257,259]** Header "The mechanism is communication poisoning" + "therefore acts on … public actionability, not on information volume" | Header asserts what the section calls an interpretation; "not information volume" is supported, the specific channel is not. | Fix: "Locating the channel: communication poisoning"; add "whether the operative property is reduced publicness or heightened fear salience is not separately identified."
5. **[§5:261]** "changes the language format … *so that* the same channel becomes less action-guiding" | The format-causes-it claim is not identified; treated messages reduce joining is what's identified. | Fix per report.
6. **[Abstract, final sentence]** "by degrading the publicness and actionability of information" | Names the unidentified mechanism. | Fix: "through the messages themselves, which retain bad news but shift from direct to coded language."
7. **[§5:278]** "the channel operates through message content" | Qwen→Llama −1.9 vs within-Llama −6.0: up to two-thirds could be family-specific. | Fix: "at least part of the effect transmits across model families."
8. **[Conclusion:285]** "join rates fall in all five clean prompt-isolation reruns" | Drops 4/5 significance and 20-cell support caveats. | Fix: restore qualifiers.
9. **[Conclusion:285]** "The same channel that helps citizens coordinate…" | Communication effect is −0.09 pp equal-weighted, negative in 3/7 models. | Fix: "the channel that transmits coordination-relevant information."
10. **[Appendix:511]** "ruling out prose style as the driver" | Two models, N=100/cell, pure only. | Fix: "indicating threshold alignment is not an artifact of one briefing prose format (tested in two models)."
11. **[§5:263]** "reduces joining substantially in every case … confirming" | Primary-model anchor is −4.0 pp at p=0.047. | Fix: report it as marginal; drop "reliably."
12. **[Conclusion:289]** "show that the qualitative pattern is not an artifact of one model family" | §5 itself calls the non-primary rows "imprecise replication." | Fix: "suggest … though imprecise."

### Generalization Issues

1. **[Title]** "How Surveillance Suppresses Coordinated Dissent" — unqualified general claim about (human) dissent that the body explicitly disclaims. | Fix: scope-bearing subtitle, e.g., "…in a Language-Based Global Game with LLM Agents," or at minimum "How Surveillance Can Suppress…".
2. **[Short abstract]** "harder to coordinate around in a crisis" — extends to real crises. | Delete "in a crisis."
3. **[§1:113]** "field data cannot separate the two channels" — universal claim; Penney/Stoycheff measure expression directly. | "field data rarely separates."
4. **[Conclusion:287]** "repression can target…" | "…the experiment isolates a channel by which repression could target…".
5. **[Conclusion:289]** even the limitation overclaims ("evidence about the structure of the game"). | "…evidence about how language-trained agents respond to this information structure."
6. **[§1:103]** "does monitoring change what citizens say…? The answer is yes." | "Within the experiment, the answer is yes."

### Missing Caveats

1. **Training-data contamination** (§3 or Conclusion): Morris–Shin and coded-speech tropes (samizdat, Aesopian language) saturate training corpora; both headline results could partly be retrieval. Name it explicitly.
2. **RLHF/safety-training compliance** (§5): the monitoring warning may trigger alignment-trained caution, observationally equivalent to strategic self-censorship; Trinity's 9–10% content-filter failures are direct evidence safety layers activate. Receiver-side claim unthreatened; sender-as-strategic-actor interpretation is.
3. **Prompt-sensitivity of the surveillance warning**: single fixed phrasing prefixed "IMPORTANT:"; stakes ladder has paraphrase variants, the operative warning has none.
4. **Country-level dependence / unreported clustered SE**: periods nest in countries with persistent means; stats_macros.tex contains a computed country-clustered slope SE of 0.0375 vs HC1 0.0100 — never reported in the paper (\ClusteredSE* macros unused). Report it; state the number of countries; show key p-values survive country-level clustering.
5. **Belief "stability" is a non-rejection, not equivalence**: p=0.587 accepts a null; no MDE or TOST equivalence bound reported.
6. **Model-snapshot population caveat**: results characterize specific hosted snapshots at temperature 0.7; the surveillance contrast (unlike the benchmark) has no temperature/decoding robustness.

### Minor Language Issues

1. Abstract: add "(four individually significant)"; consider quoting the pooled matched estimate (−13.5 pp) rather than the fragile equal-weight mean.
2. "Surveillance Produces a Chilling Effect" — legitimate but imports human-legal terminology.
3. "address" → "respond to/mitigate" for the larooij validation critiques.
4. Keyword "preference falsification" has no corresponding result; drop or connect.
5. Over-hedged paragraphs (§5:270, 275) bury the airtight architectural-isolation result; consolidate.
6. §3:192 "unit of randomization" → "unit of comparison is the matched country–period cell."
7. Appendix figA1: r ∈ [+0.67, +0.73] is below the main-run range; "stable" → "remains strong."

---

## 4. Mathematics, Equations & Notation

Core theory verified correct: Proposition 1, Eqs. 2–4, the variance claim, payoff-sweep mapping B = θ*/(1−θ*), slider formulas, dissent-floor arithmetic, and all cell counts/weights recompute exactly.

### Mathematical Errors

1. **Result 2 (paper.tex:212)** | "reduces the mean within-country correlation from r=+0.80 to r=+0.03" mixes estimands: +0.80 is the raw per-model correlation; +0.03 is computed on country-demeaned data (within_country_pearson applied only to scramble). | Demean both or label each.
2. **tab:belief_factorial** | Displayed deltas contradict displayed cell means (−5.7 vs −5.8; −11.4 vs −11.3; −0.1 vs 0.0). | Recompute from rounded means or note unrounded deltas.
3. **paper.tex:489** | "T ∈ {0.3, 0.5, 0.7, 1.0, 1.2} for three models" implies 15 combos; actual is 13 (Mistral at 3). | "up to five temperatures per model (13 combinations)."
4. **tab:finite_n** | Mistral "N periods" = 600 unexplained; stars computed over θ bins, not the N in the column — units of inference ambiguous. | Define the 600 and the number of bins.
5. **Marginal effect at the mean (paper.tex:501 / tab_regressions)** | "Surveillance reduces P(join) by 38 pp" is a derivative-based MEM on a binary regressor ignoring the θ×Surveillance interaction; discrete-change and matched-cell contrasts are ~13–14 pp. | Report the average discrete-change effect, or flag the MEM convention. **This is the most referee-dangerous math item — the 38 pp will be quoted against the 13.9 pp.**

### Notation Inconsistencies

1. **$\beta$** | logistic steepness (positive = decreasing curve) in tab_logistic_params vs conventional logit coefficients in tab_regressions: +2.15 and −1.745 describe the same comparative static with opposite signs. | Rename steepness (e.g., κ); add cross-table note.
2. **$b_i$** | sender's briefing object (line 163) vs elicited belief in [0,100] (tab_regressions col. 3). | Rename one.
3. **$N$/$n$** | declared convention (line 190) violated by tables where N = agent obs, messages, or periods. | Qualify per table.
4. **$p$** | rewiring probability vs p-values vs fitted probability. | Low risk; consider $p_{\text{rw}}$.
5. **$z$** | signal z-scores vs Fisher-z vs proportion-test z. | Say "Fisher-transformed z" at first use.
6. **pp vs pts** for belief deltas (line 383 "pts" vs line 270 "pp"). | One unit label.
7. **tab_beliefs mean belief 0.444** (fraction) vs 0–100 scale everywhere else. | Report in %.
8. **Correlation subscripts** | $r_{\text{post}}$/$r_{\text{b,d}}$ vs $r_{\text{belief,posterior}}$/$r_{\text{belief,decision}}$; $r(J,A)$ vs $r(J,A(\theta))$. | Harmonize.
9. **Matching key** stated three ways (see Agent 2).

### Undefined Notation

1. **$x_0$** (tab_finite_n) | never defined; Mistral value (−0.11) differs from tab_logistic_params cutoff (−0.32) — reconcile.
2. **$\Lambda(\cdot)$** | first used in tab_regressions note, defined only at line 600. | Define at first use.
3. **$r(J, A(\theta))$** | used in the abstract, defined verbally only in Result 1. | One defining sentence at end of Section 2.
4. **$m$** (message) in $\mu_0(m\mid b_i)$ never introduced.
5. **SOB** never expanded ("second-order belief").
6. **$\Phi$** — add "(standard normal CDF)" at Eq. 3.

### Regression Specification Issues

1. tab_logistic_params and tab_bc_statics \input but never \ref'd (also fig:communication, fig:text_baseline).
2. tab_regressions col. (1) description omits model FE and four θ×treatment interactions; cols. (2)–(3) sample subsets (44,662; 14,990) unexplained in text.
3. tab_hypotheses: H5 should be labeled "Paired t" like H4; H2's p=0.638 with r=0.014 inconsistent with pooled N=1800 — show the scramble-arm N per row.
4. Sign-convention collision between the two fit tables (see Notation #1); add cross-table note.
5. tab_surveillance_variants: baseline arm N never given; matched-vs-unmatched cells unstated.

### LaTeX Math Formatting

1. Text-mode hyphens as minus signs in tab_temperature_expanded, tab_prompt_isolation, tab_comm_estimators, tab_bc_statics. | Wrap negatives in math mode.
2. `$N = 15,000$` renders with a thin-space comma gap; macro renders "15000" elsewhere. | `15{,}000`, add separator to macro.
3. tab_logistic_params caption/notes duplicate the fitted-form sentence verbatim.
4. Hard-coded statistics in tab_msg_features and belief-factorial notes alongside macros — regeneration hazard. | Switch to macros.
5. Dead labels in never-input tab_temperature.tex / tab_slider_audit.tex. | Delete or archive.
6. No missing \left/\right, no `*` multiplication, no un-\text'ed prose found.

---

## 5. Tables, Figures & Documentation

### Tables with Missing or Incomplete Notes

| Table (file) | Missing element | Suggested addition |
|---|---|---|
| tab_models | No API/model identifiers despite paper.tex:579 promising them; no run dates | Add identifier column (OpenRouter slugs) or reword; state collection window |
| tab_main_results | N only for Pure column; no CIs for falsification columns | Add N per treatment; note why scramble/flip lack CIs |
| tab_hypotheses | "Stat" column mixes Pearson r and t-stats unlabeled; no N per row | Label statistic type; add N column |
| tab_comm_estimators | Δ columns lack units; no SEs/p-values; caption not self-contained | Add "(pp)"; SE or p for paired Δ; sample sentence |
| tab_prompt_isolation | p = "0.000"; no SEs on Δ; no CI on r(J,A) | "<0.001"; add SE/CI |
| tab_surveillance_variants | N units undefined; no SEs; baseline mean absent; Llama variants discussed in text but absent | Define N; add baseline row; add Llama rows or note omission |
| tab_temperature_expanded | One-line notes; Mistral coverage mismatch with text | Expand notes; fix text |
| tab_finite_n | Test behind stars undefined; $x_0$ undefined; treatment and θ-bin construction unstated | Define all four |
| tab_regressions | Sample composition per column unstated; clustering level for cols 2–3 ambiguous | State per-column sample and exact cluster variable |
| tab_logistic_params | No N per model/treatment; no significance | Add N column |
| tab_beliefs | 0.444 scale mismatch; no SEs/CIs; N units undefined; posterior computation undefined | Fix scale; define N; add CI; point to formula |
| tab_punishment_risk | No SDs/SEs; N units undefined; only 2/7 models unexplained | Add dispersion; define N; explain model subset |
| tab_classifiers | No train/test N; surveillance-application sample missing | Add both |
| tab_bc_classifier | N=270 units undefined | Define |
| tab_bc_statics | SE only for cutoff; N units undefined | Add SEs/CIs; define |
| tab_cross_generator | Notes duplicate caption; no SEs/CIs; N undefined | Rewrite notes |
| Belief factorial (in paper.tex) | Model never named; test behind Δ p-values undefined | Add "Primary model (Mistral Small Creative)" + test description |
| Slider values | No note that values are deterministic | One-line note |

tab_msg_features, tab_parse_errors, identification map, tab_bc_sweep_mapping have adequately complete notes.

### Figures with Missing or Incomplete Notes

| Figure | Missing element | Suggested addition |
|---|---|---|
| fig01_sigmoid | Band and vertical lines undefined; in-figure r(θ,J) = −0.83 clashes with caption's r(J,A(θ)) = +0.75 | Define band/lines; note the two conventions |
| fig02_cross_model | Vestigial legend entry ("Scramble fails") with no plotted point; axis is \|r\| but caption says r; no CIs; Trinity Comm marker invisible | Fix legend; "absolute value"; add Fisher-z CIs |
| fig03_falsification | $r_A^w$ vs $r_A$ undefined; no N; no CIs | Define both; give N |
| fig05_communication | θ-bins, shading, Δ units (proportions vs pp), CIs all undefined | Full notes |
| fig12_surveillance | Panel A bands undefined; Panel B bars lack CIs/p/N | Define; add N; reference tables |
| fig15_text_baseline | Error bars undefined; slope 7.54 in z-units vs β in θ-units | Define; note units |
| fig16_beliefs | Error bars undefined; Panel A sample/N unstated | Add |
| fig17_second_order_beliefs | Sample, N, binning unstated | Add |
| fig19_nonparametric_beliefs | Model unnamed; error bars undefined; in-figure r = −0.88 unreconciled with text r = +0.80 (different objects) | State all; reconcile |
| fig20_cross_generator | Error bars undefined; 3-decimal legend vs 2-decimal table | Define; harmonize |
| fig_construct_validity | Fragmentary caption; metrics unexplained; empty legend entry | Expand; fix legend |
| figA1/figA2 | Two-word subcaptions; raw monospace box in figA1; bands undefined; model unnamed | Expand |

### Cross-Reference Issues

- tab:bc_statics, tab:logistic_params, fig:communication, fig:text_baseline — all included, never \ref'd.
- tab_slider_audit.tex, tab_temperature.tex — orphaned files on disk.
- paper.tex:579 promises model identifiers in tab:models; the table has none — **broken promise to the reader**.
- Temperature set claim vs table coverage (13 combos).
- Unused figure assets in figures/: diagram_authoritarian_control, diagram_communication, diagram_experimental_design, diagram_game_structure, fig04_r_summary, fig06_agent_threshold, figA3_bandwidth — prune from the submission package.

### Formatting Inconsistencies

- Join rates as proportions in five tables, percentages in four — pick percent (Δs are in pp).
- p-value style: "0.000" vs "<0.001" vs decimals — standardize.
- Stars only in tab_finite_n and tab_regressions; others raw p — acceptable but state it.
- Model naming (see Agent 1 #6) including figure legends ("Mistral-Small" in fig02).
- Cutoff decimals 2 dp vs 3 dp; correlation signed 2 dp vs unsigned 3 dp.
- Notes mechanism: \parbox\footnotesize vs tablenotes \small — one convention.
- tab_bc_classifier Gap (+21.9) vs tab_bc_statics Δ (−21.9): same numbers, flipped sign conventions side by side.
- fig05 in-figure Δ in proportions vs pp everywhere else.
- \resizebox in six tables (variable font size; journals often prohibit) vs natural width elsewhere.

---

## 6. Contribution & Referee Assessment

### Part 1 — Central Contribution

**Claim:** Using LLM agents in a Morris–Shin regime-change game with natural-language briefings, surveillance applied only at the message-writing stage suppresses participation by pushing messages from direct to coded language — degrading publicness/actionability rather than first-order content or receivers' fear.

Two layers: (1) LLM agents reproduce threshold comparative statics — a replication-in-silico of Heinemann–Nagel–Ockenfels (2004)/Szkup–Trevino in the Horton (2023) spirit; on its own insufficient, a crowded genre. (2) The sender-side surveillance isolation is genuinely novel **as a design**: statelessness lets the warning be applied to the sender's call only, holding the receiver's prompt bit-identical — a decomposition neither field data (King et al. 2013; Penney 2016; Stoycheff 2016; Xu 2021) nor human labs can implement, since human subjects cannot have the warning erased at decision time. The bidirectional writer–reader rotation is a clever further check. Closest work — Avoyan (2020) cheap talk in global games and Edmond (2013) information manipulation, **both in references.bib but uncited in the text** — establishes the question; nothing implements this wedge.

**Rating: Incremental, at the high end (bordering Significant).** The design and falsification battery are a real idea. But the headline evidence rests overwhelmingly on one obscure, now-retired creative-writing finetune; cross-model replication is 20 matched cells in 3 of 5 models; and the beliefs-stable/actions-fall wedge is admittedly elicitation-dependent. The evidentiary base is too thin for the strength of the title.

### Part 2 — Identification and Credibility

**Handled well:** direct demand effects on the decision are ruled out by architecture (bit-identical decision prompts; the hold-fixed test showing −20.9 pp when the warning is moved to the decision prompt proves the direct channel exists and is absent in the main design — the paper's best feature). Sender-side prompt-demand partially handled via consequence-free and anonymous placebos (tight nulls in Mistral; Llama placebos have ±8.9 pp CIs, nearly uninformative). The rotation rejects pure same-family recognition, though the 4× asymmetry (−7.5 vs −1.9 pp) is unexplained.

**Where a skeptical seminar audience pushes:**
1. *Coded vs merely vaguer messages.* The degraded-messages control drops joining 21.4 pp — **larger than the surveillance effect**. Codedness is never independently randomized; "loss of publicness" vs "generic informativeness loss plus cautionary tone" is not separated. The surveilled-worse-than-silence fact (31.4% vs 37.1%) suggests active poisoning in Mistral but does not replicate in Llama or Qwen.
2. *RLHF-roleplay objection.* Models trained on samizdat tropes write euphemism when told they're surveilled — expected. The receiver response is the defensible object, but could be an RLHF caution prior rather than higher-order-belief reasoning. The punishment-risk elicitation meant to address fear is ceiling-bound (~8.0/10 in every condition) and cannot carry the weight.
3. *The wedge is fragile.* Messages-included elicitation (arguably the correct information set) moves beliefs sharply (−5.7, −8.2 pts); the decisive pre-decision × messages-excluded cell was never collected (model retired). Abstract and title still trade on the wedge.
4. *Contamination.* Scramble/flip show signal use, not absence of script-following; no Horton-style audit of game/hypothesis recognition.
5. *Generalization.* Seven mid-tier open-weight models, no frontier models; primary is a creative-writing finetune plausibly predisposed to vivid metaphor; pooled estimates dominated by Mistral's rows; primary endpoint retired — headline arm unreproducible by construction.

### Part 3 — Analyses: Required and Suggested

**Required:**
1. **Re-run the surveillance pipeline end-to-end on an accessible model as new primary, with nested grids** (≥500 matched cells); current replication set (20-cell overlaps) is anecdotal and the headline arm is unreproducible.
2. **Codedness-without-surveillance specificity control**: induce indirect/metaphorical style with no monitoring framing; conversely decode surveilled messages back to direct language and feed to receivers. The single biggest gap given the −21.4 pp degraded-messages result.
3. **Complete the belief factorial (pre-decision × messages-excluded) and elicit the actual mechanism object** (e.g., "what fraction of your contacts are speaking freely?").
4. **Contamination and demand audit (Horton-style)**: fresh-call game/hypothesis identification; quantitative analysis of receivers' free-text rationales (surveillance inference vs fundamental reasoning). Near-zero cost given the cached pipeline.
5. **Report headline estimands equal-weighted and reconcile the communication claim** (+2.10 pp pooled paired vs −0.09 pp equal-weight, 3/7 negative): restate as null-on-average with heterogeneity or justify the pooling ex ante.

**Suggested:**
1. Dose–response in surveillance intensity (connects to Edmond 2013 comparative statics; one prompt parameter).
2. Repeated play / code-learning counter-adaptation (euphemism treadmill; genuinely interesting economics).
3. Heterogeneity by θ region (theory predicts concentration near θ*; data exist).
4. Network-position heterogeneity (degree, rewired links; loggable).
5. Anchor magnitudes to human-lab benchmarks (Heinemann et al. 2004; Avoyan 2020).

### Part 4 — Literature Positioning

**Edmond (2013) and Avoyan (2020) sit in references.bib but are never cited in the text — disqualifying in current form** for a paper about communication and its manipulation in a regime-change global game; a referee will assume the author does not know the closest work. (Likely casualties of the recent trim; guriev2019, morris2002, shurchkov2013 and others are likewise orphaned.)

Missing entirely:
- LLM-experiments canon: Aher et al. (2023), Argyle et al. (2023), Mei et al. (2024 PNAS).
- Censorship/surveillance empirics: Chen & Yang (2019 AER), Xu (2021 AJPS), Little (2017), Roberts (2018).
- Second-order beliefs about dissent: Bursztyn–González–Yanagizawa-Drott (2020); Bursztyn–Egorov–Fiorin (2020); Cantoni et al. (2019 QJE) — the paper's central object (others' willingness becoming illegible) is exactly this literature's margin; the omission is glaring.
- Public information in coordination: Cornand & Heinemann; Morris & Shin (2002) (in .bib, uncited).

Better framing: an in-silico test of the Kuran/Edmond conversion margin — surveillance as a tax on the publicness of private information — with Bursztyn-style second-order-belief evidence as the human counterpart the design complements.

### Part 5 — Journal Fit and Recommendation

Substantive fit with JEBO: yes — computational/agent-based work, receptive to homo silicus, and the paper is far above the "LLMs play a game" floor. Fit risks: no human subjects (payoff conditional on accepting LLM populations as informative); reads as AI-measurement wearing political-economy clothes unless the Edmond/Kuran/Bursztyn positioning is built out; single-author single-pipeline with a retired primary model raises replication-policy concerns JEBO takes seriously.

**Preliminary recommendation: Revise before sending to referees.** Not desk reject — prompt isolation and the rotation are genuine methodological contributions and the robustness apparatus is unusually honest. Not yet to referees — predictable major-revision demands are already enumerable.

**Concrete bar:** (a) headline effect on ≥2 accessible models, ≥500 matched cells each; (b) one test separating surveillance-induced codedness from generic vagueness; (c) a belief/publicness elicitation surviving the instrument-sensitivity critique; (d) engage Edmond, Avoyan, Chen & Yang, Xu, Bursztyn et al., and the LLM canon.

**Alternative outlets:** JEDC (computational fit), Journal of Economic Interaction and Coordination (natural home, lower prestige), Experimental Economics (only with a human arm/benchmark), GEB (only with sharper theory), JPubE/QJPS (would want human evidence). A short human-subject companion — humans reading LLM-coded vs direct messages — would open EJ-tier options.

### Part 6 — Questions to the Authors

1. **Primary model selection and survivorship.** Why is the primary model a creative-writing finetune, now retired from the API? A model tuned for vivid prose is plausibly predisposed to produce exactly the metaphorical "coded" messages your mechanism requires. Was it chosen before or after observing treatment effects, and how many models/pipelines were piloted? Report the full set ever run; rerun the complete design on an accessible model with nested grids.
2. **Specificity of the mechanism.** Your degraded-messages control reduces joining by 21.4 pp — half again your surveillance effect. What distinguishes "surveillance degrades publicness" from "any shift toward vaguer messages reduces joining"? Run (i) style-induction without monitoring framing and (ii) a decoding treatment. If (i) reproduces and (ii) reverses the effect, your interpretation survives.
3. **Which belief information set is the right one?** The wedge holds only when peer messages are excluded from the belief prompt; including them — arguably the information set under which the action was taken — moves SOB −8.2 pts, consistent with an ordinary belief channel. On what grounds is messages-excluded "primary," and can you collect the missing factorial cell on the new primary?
4. **Saturated punishment-risk instrument.** Elicited risk is ≈8.0/10 in every condition — a ceiling. How can this rule out fear-salience contagion? Re-elicit with headroom (e.g., P(arrest | join)) and analyze receivers' rationales for fear vs publicness reasoning.
5. **Communication result framing.** "Modestly raises joining" (+2.10 pp pooled) coexists with −0.09 pp equal-weight and 3/7 negative models. Which estimand was specified ex ante? Was any part of the design preregistered?
6. **Contamination and demand audit.** Have you asked the models to identify the game, the experimenter, or the hypothesis (per Horton 2023)? What distinguishes threshold reasoning over the briefing from script-completion, beyond scramble/flip, which only establish signal use?
7. **Unexplained asymmetries.** The rotation gives −7.5 pp one direction and −1.9 pp the other; surveilled messages are worse than silence in Mistral but better than silence in Llama and Qwen3 30B. What model of the mechanism accommodates both, and why are Edmond (2013) and Avoyan (2020) — in your bibliography — never discussed in the text?

---

## Priority Action Items

**CRITICAL** (must fix — could cause desk rejection or major referee objections):

1. **Cite and engage Edmond (2013) and Avoyan (2020)** — the closest theory and experimental papers, already in references.bib but absent from the text; add the Bursztyn second-order-beliefs-about-dissent and Chen & Yang / Xu surveillance-empirics literatures. (Agent 6, Part 4)
2. **Address the mechanism-specificity gap**: the degraded-messages control (−21.4 pp) exceeds the surveillance effect (−13.9 pp); add a codedness-without-surveillance control and/or a decoding treatment, or substantially weaken every "publicness/actionability" mechanism claim (title, both abstracts, §5 header, conclusion). (Agents 3 & 6)
3. **Fix the primary-model fragility story**: the headline arm runs on a retired creative-writing finetune with 20-cell replications in 3/5 models. Either rerun on ≥1 accessible model with nested ≥500-cell grids, or restructure claims around the pooled/equal-weight evidence with full caveats in abstract and conclusion. (Agent 6, Required #1)
4. **Align abstracts with the body's hedges**: short abstract's "the mechanism is linguistic," "all prompt-isolation models" (4/5 significant), and "expectations remain stable" (only messages-excluded) all strip caveats the body maintains; long abstract's "matched-cell mean" mislabels the equal-weight average (−11.2) while the table's "Pooled matched" (−13.5) appears nowhere in prose. (Agents 2 & 3)
5. **Report the country-clustered inference**: the computed country-clustered slope SE (0.0375 vs HC1 0.0100) sits unused in stats_macros.tex; state the number of country clusters and show headline p-values survive country-level clustering. (Agent 3, Missing Caveats #4)

**MAJOR** (should fix — will likely be raised by referees):

6. **Correct or reframe the 38 pp marginal effect** (tab_regressions / paper.tex:501): derivative-based MEM on a binary regressor ignoring the θ×treatment interaction, vs ~13–14 pp discrete-change/matched contrasts — a referee will quote these against each other. (Agent 4, Math #5)
7. **Fix the Result 2 estimand mix-up**: r=+0.80 (raw) vs r=+0.03 (within-country demeaned, applied only to scramble) are different statistics presented as one comparison. (Agent 4, Math #1)
8. **Add the missing LLM-experiment caveats**: training-data contamination, RLHF/safety-compliance as alternative sender mechanism (Trinity's content-filter failures are direct evidence), single-phrasing surveillance warning, model-snapshot scope; plus a Horton-style contamination/demand audit (cheap, given cached pipeline). (Agents 3 & 6)
9. **Resolve the title's unqualified claim**: add a scope-bearing subtitle or "Can Suppress"; fix the six generalization spots (e.g., "in a crisis," "The answer is yes," "field data cannot"). (Agent 3, Generalization)
10. **Cite the four orphaned floats** (tab:logistic_params, tab:bc_statics, fig:communication, fig:text_baseline) or cut them; delete orphaned files tab_temperature.tex and tab_slider_audit.tex; fix the broken "model identifiers are listed in Table tab:models" promise; remove "Part I/II" labels. (Agents 2, 4, 5)
11. **Complete table/figure documentation**: SEs/CIs and N-with-units in tab_comm_estimators, tab_prompt_isolation, tab_surveillance_variants, tab_beliefs, tab_cross_generator, tab_punishment_risk; define error bars/bands in essentially every figure caption; reconcile fig01's in-figure r(θ,J)=−0.83 with the caption's +0.75 convention; remove fig02's vestigial legend entry. (Agent 5)
12. **Reconcile the communication framing**: "modestly raises joining" vs −0.09 pp equal-weight and 3/7 negative models; fix the conclusion's "channel that helps citizens coordinate." (Agents 3 & 6)

**MINOR** (polish):

13. Fix rounding-contradiction cells in the belief factorial and the coded-metaphor delta; "$< 0.2$" → "$\leq 0.2$" in the punishment-risk claim; explain the Trinity 1-row gap and the tab_finite_n N=600. (Agent 2)
14. Notation cleanup: β reuse (steepness vs logit coefficients, opposite sign conventions), b_i collision, N/n convention, pts vs pp, tab_beliefs 0.444 scale, define x_0/Λ/SOB/r(J,A(θ)) at first use, "p = 0.000" → "<0.001", math-mode minus signs. (Agent 4)
15. Style sweep: spaced " - " dashes → em dashes (9+ instances), en dashes in "country–period"/"writer–reader" compounds, "$z$-score" standardization, first-person consistency, sentence-case captions, model-name standardization across tables and figure legends, prune the seven unused figure files, fix the line-165 "less … messages" error and the line-259 inverted-meaning sentence. (Agents 1 & 5)
