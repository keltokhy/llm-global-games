# Theory Note: The Intelligence Cost of Surveillance

*v1 — June 2026. Advisor-ready sketch; propositions stated with proof strategy, not proofs.*

## 1. One-sentence claim

Surveillance deters dissent by changing how citizens speak, but the same speech change degrades the regime's own intelligence, so surveillance buys quiet at the price of blindness precisely in crisis states.

The key phrase is **actionable intelligence**, not "public opinion." The regime may still know people are angry — the pilot shows aggregate temperature-reading survives. What it loses is the harder object: who is willing to act, whether enough others are willing, and whether today's weak-state cell is the one where coordination tips.

## 2. Motivation

The usual view treats surveillance as a monotone authoritarian technology: more monitoring, more control. This paper argues the technology is two-sided. Public surveillance changes citizen speech endogenously. Citizens do not simply stop talking; they substitute away from direct, incriminating, coordination-relevant messages into coded, deniable, low-legibility messages. That reduces citizen coordination — and it damages the regime's signal.

> The regime listens harder and hears less.

The platform-cycle story frames the introduction: new channels create mutual legibility before the state learns to monitor them; monitoring restores quiet but destroys actionable information; the next low-monitoring channel reopens common knowledge and can produce sudden bursts. Nepal 2025 is the modern anecdote, worded cautiously: Reuters/AP report that a social-media ban helped trigger the September 2025 youth protests and that Sushila Karki was appointed interim PM after the unrest; claims that Discord deliberation shaped the leader selection should be pinned to the best available sourcing before print.

## 3. Timing

The game's timeline (matching the experiment's structure):

- **t = 0.** The regime publicly commits to (or has a reputation for) surveillance level $s \in [0,1]$ — *advertised monitorability*, chosen ex ante, before $\theta$ is realized. Hidden interception is free and irrelevant to behavior; only believed monitoring chills. Rational expectations ties advertised to believed monitoring; a signaling extension (Section 12) relaxes this.
- **t = 1.** Nature draws regime strength $\theta$; citizen $i$ observes private willingness/signal $x_i$ (increasing in perceived regime weakness and grievance).
- **t = 2.** Citizens exchange pre-play messages under $s$ (style choice below). The regime intercepts.
- **t = 3.** The regime reads intercepted messages and chooses a response $q \in \{\text{ignore}, \text{preempt}, \text{concede}\}$.
- **t = 4.** Citizens choose $a_i \in \{0,1\}$; the uprising succeeds iff $A = \sum_i a_i \ge \tau(\theta)$, with $\tau'(\theta) > 0$ (low $\theta$ = weak regime).

## 4. Message technology and the coding decision

Citizen $i$ chooses message style $m_i \in \{\text{direct}, \text{coded}\}$. Direct messages are highly informative about $(x_i, \text{intent})$ but incriminating: if intercepted (probability tied to $s$) and subversive, the sender pays $\kappa > 0$. Coded messages are safer but garble content — to *all* readers.

**Assumption A1 (incrimination).** Punishment attaches to intercepted direct messages whose content is subversive (anti-regime, action-cueing). Pro-regime or neutral direct content carries no risk.

**Lemma 1 (cutoff coding) — to prove.** Given A1, the coding decision is a cutoff in the private signal: there exists $\hat{x}(s)$, increasing in $s\kappa$, such that citizen $i$ codes iff $x_i$ is more anti-regime than $\hat{x}(s)$. Only citizens with something incriminating to say pay the coding cost.

This is the load-bearing structural result, and it is what converts the pilot's empirical pattern from an assumption into a theorem downstream: because coding is selected on $x_i$, and low-$\theta$ states generate more low-$x_i$ citizens, **the message distribution diverges across surveillance regimes only in weak states**. In strong states almost no one has anything to hide, so surveilled and free message distributions nearly coincide — and the regime loses nothing. (The pilot found exactly this, including the mild *reversal* at $\theta \ge 0$: uniformly guarded calm-state traffic is trivially classifiable.)

So the crisis-state condition the data demands,
$$\frac{\partial^2 d}{\partial s\, \partial(-\theta)} < 0$$
(direct speech falls faster in $s$ when the regime is weak), should be **derived from Lemma 1**, not assumed. State it as a corollary.

**Empirical update (2026-06-11, text-divergence test on 24,306 matched messages).** The data refine Lemma 1's object. Surface *style* divergence across arms is nearly global — an arm-classifier on text features reaches AUC 0.93–0.97 in *every* sender-signal and θ bin — with a significant but modest selection gradient (divergence falling in sender z, ρ = −0.79, p = 0.02; in θ, ρ = −0.98, p < 0.001). Yet *analyst performance* degrades only in crisis states. Reconciliation: under a warning, essentially everyone shifts to a guarded register (compliance is near-universal), but the shift is **content-destroying only where incriminating content existed**. A guarded version of "nothing is happening, I'm staying home" carries the same likelihood ratio as the direct version; a guarded version of "I'll be in the square tonight" does not. The cutoff therefore lives in the **informativeness loss**, not (only) in the style choice. This suggests stating Lemma 1 in two parts: (1a) style adjustment weakly increasing in $s$ for all types (possibly near-uniform); (1b) the likelihood-ratio degradation induced by any common guarded register is concentrated on low-$x_i$ senders — which is mechanical, and delivers Prop 2 with *weaker* assumptions than a coding-choice cutoff. The model may not need the discrete style choice at all.

## 5. Two readers and the legibility wedge

Both citizens and the regime read the same transformed speech. Define informativeness of the message vector $M$ about three objects:

- $I_\theta(s,\theta)$: regime weakness (the FALL task),
- $I_A(s,\theta)$: total mobilization (the JOIN_PERCENT task),
- $I_i(s,\theta)$: individual join decisions (the per-sender task).

Let $L_C(s,\theta)$ be citizen-side legibility and $L_R(s,\theta)$ regime-side legibility, and define the **wedge**
$$W(s,\theta) = L_C(s,\theta) - L_R(s,\theta).$$

Coding aims to be steganographic ($W > 0$: peers decode, the monitor doesn't — Blume–Board language competence is the natural formalization). Whether it succeeds is empirical. Paper 1 establishes $\partial L_C / \partial s < 0$ (receivers lose; joining falls). The pilot establishes $\partial L_R / \partial s < 0$ concentrated at low $\theta$. Both readers lose: in the data, equilibrium coding looks closer to *mutually assured confusion* than to successful steganography. The model should permit both regimes of $W$ and let the experiment locate it — this is a quantity, not an axiom.

The distinction from "censorship reduces speech" models: messages still exist, volumes barely change. What collapses is their **likelihood ratios**.

## 6. Citizen side: the deterrence benefit

Citizen $i$ joins when
$$\Pr\!\left(A_{-i} + 1 \ge \tau(\theta) \mid x_i, M\right) B - c - r_i \ge 0.$$

Lower $L_C$ means receivers can no longer read others' willingness from messages, strategic uncertainty rises, and the uprising probability
$$U(s,\theta) = \Pr(A \ge \tau(\theta))$$
falls: $\partial U / \partial s < 0$. This is paper 1's measured channel and the regime's benefit from surveillance.

## 7. Regime side: the intelligence cost

The regime's response value depends on intelligence quality. Let $H(I_R)$ be the probability the regime successfully heads off an otherwise-successful uprising (preempts the right cells, arrests the right organizers, concedes when it must), $H' > 0$. Reduced-form survival:
$$S(s,\theta) = 1 - U(s,\theta)\,\bigl[1 - H(I_R(s,\theta))\bigr].$$

$$\frac{\partial S}{\partial s} = \underbrace{-\frac{\partial U}{\partial s}\,[1 - H(I_R)]}_{\text{chilling benefit } (+)} \; + \; \underbrace{U\, H'(I_R)\,\frac{\partial I_R}{\partial s}}_{\text{blinding cost } (-)}.$$

That is the whole model. The regime's problem is $\max_s \mathbb{E}_\theta[S(s,\theta)]$, and the two terms price the same instrument.

*Modeling flag:* this $S$ is deliberately reduced-form. The microfounded version needs the t=3 response to feed back into $U$ (preemption lowers participation), which risks double-counting; cleanest is to define success as {coordination clears $\tau$} ∩ {regime fails to preempt}, with the two conditionally independent given $(\theta, M)$. Settle this before writing the proofs.

## 8. Main propositions

**Proposition 1 (two-price surveillance).** Under A1 and Lemma 1, increasing $s$ weakly reduces both citizen-side coordination ($U$) and regime-side intelligence ($I_R$). Surveillance is simultaneously a monitoring technology and a data-destruction technology. *Proof strategy:* coding cutoff moves with $s$; both readers' likelihood ratios degrade on the coded region.

**Proposition 2 (crisis blindness).** Because coding is selected on $x_i$ (Lemma 1), the intelligence cost is concentrated in weak-regime states:
$$\left|\frac{\partial I_R}{\partial s}\right| \text{ is increasing in } (-\theta),$$
and $\partial I_R/\partial s \approx 0$ for $\theta$ high. *This is the pilot's exact pattern* — Brier deltas of −0.05 to −0.14 at $\theta<0$ (all five analysts, p ≤ 0.001), nothing or mild reversal at $\theta \ge 0$. *Proof strategy:* selection — the mass of citizens past the coding cutoff is increasing in $(-\theta)$; in the limit $\theta \to \infty$ no one codes and the distributions coincide.

**Proposition 3 (interior surveillance).** If marginal deterrence is initially high but bounded, and marginal intelligence loss rises in crisis states (Prop 2), regime survival is non-monotone in advertised surveillance: the regime can over-monitor. *Conditions to pin down:* relative curvature of $U$ and $H \circ I_R$ in $s$; the corner $s=1$ is optimal only when $H' \approx 0$ (intelligence worthless — e.g., no preemption technology).

**Empirical update (2026-06-11, dose-response on the matched 500-cell grid).** Both margins are step-like in $s$, not smooth. Chilling (mean join fraction): none 0.471 → mild 0.408 → full 0.391 → severe 0.376 — monotone but strongly concave, two-thirds of total chilling at the mildest warning ("routine monitoring," no consequence language). Crisis blinding (ΔBrier, 3 analysts): mild already delivers ~75–85% of the full-dose effect (e.g., DeepSeek −0.082 mild vs −0.112 full vs −0.110 severe; all p<0.0001 at every dose). Consistent with the Lemma 1 text finding: citizens treat *any* monitoring cue as a switch into the guarded register — a salience/trigger response (Kuran-style discrete public persona), not marginal deterrence. Implications: (i) the economically meaningful margin is **extensive** — visibly monitor or don't; both the benefit and the cost arrive almost entirely at $s > 0$; (ii) *conditional on monitoring*, escalating severity buys additional silence (−6.3 → −9.5 pp) at little additional blinding, so within the monitoring regime more severity weakly dominates for the regime; (iii) Prop 3 should therefore be restated at the extensive margin — $S(\text{monitor}) \lessgtr S(\text{silent})$ as a function of crisis exposure and response value $H'$ — with the smooth interior version as a special case the data do not favor. Policy reading: even the softest monitoring notice imposes nearly the full coordination cost on citizens *and* nearly the full intelligence cost on the state.

**Proposition 4 (suppressed mean, fattened tail).** Surveillance lowers the rate of mobilization attempts but raises the regime's surprise conditional on an attempt — lower mean unrest, fatter tail of large, poorly anticipated explosions. *This needs the most work:* requires a distribution over $\theta$ (or repeated play with stored grievance) and a formal surprise object (e.g., regime's posterior on $\{A \ge \tau\}$ at t=3 vs. the realized outcome). It is the bridge to the platform-cycle introduction; acceptable to state as a corollary of Props 1–2 plus a grievance-accumulation assumption in v1.

## 9. Platform-cycle extension

Platforms $p = 1, \dots, P$ with monitoring $s_p(t)$; new platforms start at $s_{\text{new}}(0) \approx 0$ and the regime learns: $\dot{s}_p(t) > 0$ after adoption. Citizens choose platforms on reach vs. safety. Each new low-monitoring platform restores direct speech and mutual legibility, spiking the hazard of mass action; monitoring then restores quiet while actionable intelligence decays; stored discontent re-becomes legible on the next channel.

The cycle: new channel → mobilization wave → regime learns and advertises monitoring → speech codes → quiet plus decaying intelligence → next channel → stored discontent ignites. **Surveillance controls ignition, not fuel.** Testable timing implication: revolution hazard clusters in the window between citizen adoption of a channel and state monitoring capability on it (cassettes '79, fax '89, Facebook '11, Telegram '19–'20, Discord '25).

Keep this as a stylized extension (one subsection), not a second model.

## 10. Experimental mapping

| Model object | Experimental counterpart |
|---|---|
| $s = 0$ vs $s = 1$ | baseline vs surveillance-warning message arms (matched cells, identical $\theta$, senders, signals) |
| $I_\theta$ | analyst FALL estimate scored against $\theta$ / coup outcome (Brier, AUC, $\rho$) |
| $I_A$ | JOIN_PERCENT vs realized join share (MAE) |
| $I_i$ | per-sender verdicts vs actual decisions (accuracy, AUC, targeting) |
| $L_C$ | paper 1's receiver-side joining response |
| $W$ | $\Delta_{\text{peer}}$ vs $\Delta_{\text{regime}}$ on the same coded corpus |
| Lemma 1 selection | message-feature divergence across arms by $\theta$ bin |

Crisis-state prediction: $I_R(1,\theta) < I_R(0,\theta)$ especially for $\theta < 0$ — confirmed in the pilot for $I_\theta$ (5/5 analysts) and $I_i$ (4/5), with $I_A$ unaffected. The headline is not "analysts are worse on average"; it is **analysts are worse exactly where the regime needs intelligence most** — and average metrics would never reveal it.

## 11. Literature positioning

- **Myatt–Wallace (ReStud 2012)**: coordination-game language for signal precision vs. clarity — the result here degrades the *clarity* of social signals, not the quantity of speech.
- **Yang (AJPS 2025)**: closest neighbor — AI censorship accuracy falls with repression, worst in crises. His channel is missingness in training data; here the monitored population endogenously *re-writes* speech, so the regime's intelligence problem is produced by its own surveillance in equilibrium, and the citizen-coordination side is priced in the same model.
- **Gehlbach–Luo–Shirikov–Vorobyev (AJPS 2026)**, Egorov–Guriev–Sonin, Wintrobe: the dictator's-dilemma lineage — no communication stage, no message-form margin.
- **Braghieri (AER 2024)**: monitored speech loses informativeness (social-image setting) — the model+experiment template.
- **Spector (2022)**: the only eavesdropper-cheap-talk model in economics (IO); ported here to political economy.
- **Shadmehr–Bernhardt (QJPS 2017)**: the one citizen-to-citizen communication model in regime change.
- **AI-safety bridge (intro/discussion only)**: OpenAI's CoT-monitoring results — penalizing visible "bad thoughts" produces models that hide intent; monitorability is fragile. Same abstract mechanism: monitoring pressure selects for less monitorable communication.
- Kuran (1989, 1991) and Chwe for preference falsification, surprise, and common knowledge — the Prop 4 / platform-cycle frame.

## 12. Open theory questions (ordered by importance)

1. Prove Lemma 1 cleanly (cutoff coding under A1) — everything hangs on it.
2. Derive Prop 2 from Lemma 1 by selection; verify the limit cases.
3. Settle the t=3 feedback structure (preemption vs. double-counting) before Prop 3.
4. Formalize Prop 4's surprise object; decide static-with-$\theta$-distribution vs. light dynamics.
5. The visibility wedge: if actual and advertised $s$ can differ, the regime wants covert-max/advertised-low; what does the signaling equilibrium look like? (Extension or follow-up paper.)
6. Where does $W > 0$ (true steganography) survive in equilibrium? Connects to the decoder-arm experiment and the Blume–Board structure.

## 13. The first two pages

Do not start with equations. Paragraph 1: new channels made dissent mutually visible. Paragraph 2: regimes monitored them; visible mobilization fell — and speech changed. Paragraph 3: the claim — monitored citizens still communicate, but in a form that destroys the regime's actionable intelligence. Paragraph 4: pilot headline — analyst models reading surveilled messages are *especially* worse in weak-regime cells, where correct inference matters most. Paragraph 5: model preview — surveillance enters twice. Then equations.

Abstract, first two sentences:

> Surveillance deters dissent, but the deterrence is purchased with the regime's own intelligence. In a matched communication experiment, analyst models reading surveilled messages are not simply noisier; they become least accurate in weak-regime states, exactly when regimes most need to identify mobilization.
