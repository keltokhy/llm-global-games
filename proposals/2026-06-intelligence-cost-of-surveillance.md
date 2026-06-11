# The Intelligence Cost of Surveillance

*(Working title history: "The Dictator's Microphone: Surveillance That Blinds" — retitled 2026-06-11.)*

*Proposal draft — June 2026. Companion/JMP candidate following "Speaking in Code."*

## Pitch

Surveillance prices the same message distribution twice. Paper 1 measured the first price: monitored senders write guarded messages, and peers lose the cues that convert bad news into a shared reason to act. This paper measures the second: the regime, listening to those same guarded messages, loses the intelligence it was listening for. Once speech is watched, it stops encoding intentions — Goodhart's law applied to repression. The regime therefore faces a real interior trade-off in how much surveillance to *advertise*: visible monitoring buys silence but spends information; covert monitoring preserves the signal but buys no deterrence. Headline: **the regime that listens loudest hears least.**

## Theoretical framework

**Primitives.** Morris–Shin regime-change game as in paper 1: citizens $i \in [0,1]$, regime strength $\theta$, private signals $x_i = \theta + \varepsilon_i$, join/stay with payoffs $(B, -C, 0)$, regime falls iff $A > \theta$.

**Messaging stage.** Each citizen sends a message to peers before acting. Message form $\varphi_i \in \{\text{direct}, \text{coded}\}$:
- *Direct*: peers observe the sender's signal and willingness-to-act cue; the monitor reads the same content.
- *Coded*: peers observe the signal with extra noise $\eta_p$ (or lose the willingness cue with probability $\lambda$); the monitor reads with noise $\eta_r \geq \eta_p$. The gap $\eta_r - \eta_p$ is the **legibility wedge** — the degree to which the code is steganographic (peers decode, monitor doesn't) vs. jointly destructive (everyone loses). In paper 1 receivers demonstrably lose content, so the wedge sign is an empirical question, not an assumption.

**Surveillance.** The regime's *advertised* monitoring intensity is $\hat m$; senders of intercepted direct subversive messages pay $\kappa$. Citizens best-respond to $\hat m$; rational expectations ties advertised to actual intensity in equilibrium (a signaling extension can relax this).

**Regime's inference problem.** The regime knows $\theta$ but not realized sentiment/intentions; after reading intercepted traffic it chooses a response $\rho \in \{\text{ignore}, \text{concede}, \text{preempt}\}$ whose value depends on the true attack mass. Intelligence value $V(\hat m)$ falls as coding rises; chilling value $S(\hat m)$ (reduction in attack probability via paper 1's channel) rises in $\hat m$.

**Target propositions.**
1. *Equilibrium coding.* Cutoff structure: citizens with sufficiently anti-regime signals code; message informativeness to both peers and monitor is decreasing in $\hat m \cdot \kappa$.
2. *Two prices.* Attack mass falls in $\hat m$ (chilling); the regime's response-error rate rises in $\hat m$ (blinding).
3. *Interior optimum.* Regime survival is non-monotone in $\hat m$ whenever the response margin matters (crises, concession-effective regimes); the corner of maximal advertised surveillance is optimal only when intelligence is worthless. Comparative static: regimes facing volatile fundamentals should *whisper* their threats.
4. *Goodhart microfoundation of Yang (2025).* The measured informativeness of intercepted traffic declines as advertised monitoring expands, and the decline is steepest in weak-regime states — exactly when subversive content would have been written. Yang gets this from training-data missingness; here it is equilibrium obfuscation of messages that *are* written.
5. *(Extension) Visibility wedge.* A regime that could decouple believed from actual monitoring would intercept covertly and advertise nothing; reputation dynamics close the gap and generate cycles of crackdown and quiet.

**Theory anchors.** Braghieri (AER 2024) — monitored speech loses informativeness (model + experiment template). Spector (RIO 2022) — cheap talk under an eavesdropping antitrust authority (the only existing eavesdropper-cheap-talk model in econ; porting it to political economy is open). Blume–Board (ECMA 2013) — differential language competence as the formal hook for the legibility wedge. Shadmehr–Bernhardt (QJPS 2017) — the lone citizen-to-citizen communication model in regime change. Edmond (ReStud 2013); Egorov–Guriev–Sonin (APSR 2009); Wintrobe; Lorentzen (2013, 2014); Tirole "Digital Dystopia" (AER 2021); Dragu–Lupu (IO 2021).

**Must-cite-and-differentiate (the committee will check).**
- **Eddie Yang, "The Limits of AI for Authoritarian Control" (AJPS 2025).** Closest paper. His channel: self-censorship as *missingness* corrupts censorship-AI training data; ML simulation on Weibo, no equilibrium model, no citizen coordination game, coded language explicitly flagged as unmodeled. This paper: obfuscation of written messages, equilibrium object, both sides priced in one model.
- **Gehlbach–Luo–Shirikov–Vorobyev, "Is There Really a Dictator's Dilemma?" (AJPS 2026).** Formalizes popularity opacity; no communication stage, no message-form margin.
- **Andirin–Neggers–Shadmehr–Shapiro (NBER 30167, rev. 2025).** Preventive repression makes protest less predictable — adjacent unpredictability result, different instrument.
- **AI-safety bridge:** Baker et al. (OpenAI, 2025) — optimizing against chain-of-thought monitors produces *obfuscated* misbehavior; they propose a "monitorability tax." Motwani et al. (NeurIPS 2024), Mathew et al. (2025) — steganographic collusion emerges under oversight pressure; paraphrasing defenses fail. Same game, monitor's side. Citing this buys a second audience and a policy frame: autocrats and AI labs face the same trade-off.

## Experimental design

**Phase A — regime-analyst pilot (near-zero cost; archived data only).**
Inputs already exist: `CLEAN_RUN/message_banks/baseline_surveillance_matched.parquet` (matched μ₀/μ_S message pairs), `direct_coded_pairs.parquet`, and agent-level logs containing each sender's actual JOIN/STAY decision. Analyst LLMs (several open models; include the nested Llama 3.3 70B for continuity) receive batches of intercepted messages and perform:
1. Estimate regime-fall probability / recover $\theta$ (score: MSE, calibration against realized $\theta$).
2. Predict the cell's join fraction (score against realized $J$).
3. Classify each sender's intent (will this sender JOIN?) — score against the sender's actual recorded decision.
4. Targeting task: rank the 25 senders in a cell by subversiveness; score precision@k against actual joiners. This is the repression-targeting margin.

Predictions: all four degrade on μ_S relative to μ₀; degradation largest at low $\theta$. The targeting task is the headline — it converts "blinding" into the quantity autocrats actually care about.

**Phase B — visibility and adaptation arms (new sender-side runs).**
- *Covert arm*: by construction, covert interception of unwarned senders ≡ scoring baseline messages — the design collapses cleanly, no new generation needed.
- *Overt arms*: existing surveillance warning, plus severity gradations (addresses paper 1's single-phrasing limitation directly).
- *Decoder-regime arm*: analysts receive an in-domain codebook (examples of decoded coded messages) — an upper bound on regime adaptation, and the static stand-in for the arms race.
- *Legibility wedge measurement*: same coded corpus scored by peer-role receivers (paper 1 machinery) and regime-role analysts; report $\Delta_{\text{peer}}$ vs. $\Delta_{\text{regime}}$.

**Phase C — equilibrium closure (the money figure).**
Plug measured chilling (paper 1) and measured blinding (Phase A/B) into the regime's response problem: regime preempts/concedes on predicted $A$; compute survival across $\theta$ under each advertised-surveillance level. Target exhibit: **regime survival as an inverted-U in advertised surveillance.**

**Validation per mid-2026 norms — zero-budget version.** The regime-side claim does not need human subjects at all, and this is a feature, not a workaround: the modern monitor *is* a machine reader (Yang 2025; Beraja et al. AI-tocracy), so LLM/ML analysts are the object of interest, not a proxy for humans. Two moves make the blinding result model-free:
- *Classifier and econometrician baselines*: alongside LLM analysts, fit a supervised classifier (and a simple regression on text features) to predict $\theta$ / sender intent from μ₀ messages, then evaluate on μ_S (and vice versa). If even an in-domain-trained statistical reader loses accuracy on surveilled traffic, the result cannot be an artifact of any one analyst model.
- *Analyst-model × sender-model factorial* as the multiverse; preregistration mirroring `CLEAN_RUN/preregistration.md`; seed/temperature manifests as in paper 1.

The only human-validity question attaches to the sender side (do LLM senders self-censor the way humans do) — which paper 1 already defends, and which can be additionally anchored *descriptively and for free* by comparing the feature shifts (direct terms ↓, metaphor ↑, deniable phrasing ↑) against documented real-world coded-dissent corpora (Chinese homophone evasion, algospeak, Aesopian language).

## Risks and fallbacks

1. *Read as "Yang with extra steps."* Mitigate by leading with the equilibrium model and the two-sidedness; Yang has neither. The survival inverted-U is a result he cannot produce.
2. *Analyst task at ceiling/floor.* Pilot calibrates difficulty via batch size (messages per inference) and signal-to-noise of the cell. If single messages are too easy, score sparse intercepts (realistic anyway).
3. *Mechanical equilibrium closure.* Ground the response rule in the model section; report robustness over response-rule families.
4. *Wedge could be ≈ 0 (joint destruction).* Still a finding: "coded dissent protects senders by blinding everyone" — reframes the contribution toward the communication tax. Either sign publishes.

## Milestones

1. **Week 1–2:** Phase A pilot on archived banks (one weekend of analyst inference + scoring). Go/no-go signal on the blinding effect.
2. **Week 2–4:** Two-page model note (Propositions 1–3 sketched; check the cutoff equilibrium is clean under the Blume–Board legibility structure).
3. **Month 2:** Phase B arms + prereg; theory full draft.
4. **Month 3–4:** Phase C closure, classifier/econometrician baselines, working paper.

Fast path: because ~80% of infrastructure is reused, this can be a circulating working paper by early fall 2026.
