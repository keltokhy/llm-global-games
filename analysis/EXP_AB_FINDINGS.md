# Experiments A & B: positively identifying the surveillance mechanism

Run June 2026 on Llama 3.3 70B receivers, nested 500-cell grid (10 countries x
50 periods, n=25), messages rewritten by Qwen3 30B and replayed through the
fixed-message rig with the decision prompt held surveillance-free
(`--decision-context none`). Inference is matched-cell, country-clustered
(G=10) with an exact wild-cluster bootstrap. These are NOT yet integrated into
the paper.

## Experiment B — does inferred repression risk drive the effect? (Issue 3)

Four arms replayed on the same cells:

| Arm | Join % | vs baseline | vs surveillance |
|-----|-------:|------------:|----------------:|
| baseline (comm, verbatim)    | 45.6 | --             | |
| surveillance (verbatim)      | 37.4 | **-8.16** (p=.002) | -- |
| risk-stripped                | 45.8 | +0.28 (p=.31)  | **+8.44** (p=.002) |
| risk-only                    | 41.6 | **-3.99** (p=.002) | +4.17 (p=.002) |

Manipulation check (share of messages with the cue, full 500 cells):
risk/fear terms — baseline 65.5%, surv 81.3%, **risk-stripped 9.5%**,
**risk-only 100%**; actionability and weakness preserved/sharpened throughout.

**Findings.**
1. The replay reproduces the live nested surveillance effect almost exactly
   (-8.16 pp vs the -8.0 pp live estimate), validating the rig.
2. **risk-only:** appending one explicit punishment-risk clause to otherwise
   normal baseline messages suppresses joining by ~4 pp — about HALF the
   surveillance effect, clustered-significant. So inferred repression risk
   transmitted through message content IS a real, non-negligible channel.
   This *nuances / partly corrects* the manuscript's current claim that the
   risk channel is "disfavored": it is not the whole story, but it is not
   negligible either.
3. **risk-stripped:** removing the cautious/hedged framing (and restating
   weakness + readiness plainly) restores joining fully to baseline. None of
   the effect survives once the guarded framing is gone.

**Caveat.** risk-stripped is not a pure risk removal: the rewrite also sharpened
actionability (78.6%->95.1%) and stated weakness more plainly (40.6%->55.5%),
so it bundles "remove caution" with "make actionability legible." And risk-only
cannot separate personal deterrence from signaling that *others* are deterred.
Net reading: the surveillance effect travels with the guarded framing, which
carries both an explicit-risk component (~half, isolated by risk-only) and a
loss-of-legible-actionability component (the rest) — consistent with the
paper's actionability/legibility thesis, while giving the risk channel more
weight than the current draft does.

## Experiment A — willingness x style 2x2 (Issue 1 headline)

Each baseline message rewritten into four variants holding the factual
regime-strength assessment fixed; each replayed on the same 500 cells.

| Arm | Join % | vs baseline |
|-----|-------:|------------:|
| baseline (comm)              | 45.6 | -- |
| willingness+direct (w1d)     | 53.0 | +7.47 (p=.002) |
| willingness+coded  (w1c)     | 48.7 | +3.17 (p=.002) |
| no-willingness+direct (w0d)  | 45.1 | -0.48 (p=.09)  |
| no-willingness+coded  (w0c)  | 43.2 | -2.34 (p=.002) |

Manipulation check (full logs): willingness cue present in 99.5%/51.8% of the
w1 arms vs 2.9%/3.2% of the w0 arms; coded markers in 85-91% of coded arms vs
15-19% of direct arms; factual weakness preserved within the direct arms
(dictionary collapses for coded arms because metaphor replaces explicit terms,
confirmed by inspection).

**Willingness contrast (holds factual content AND style fixed):**
- direct: w1d - w0d = **+7.95 pp** (p=.002)
- coded:  w1c - w0c = **+5.50 pp** (p=.002)

**Style contrast (holds willingness fixed):**
- willingness present: w1d - w1c = +4.30 pp (p=.002)
- willingness absent:  w0d - w0c = +1.86 pp (p=.002)

**Findings.**
1. **The willingness-to-act cue is positively identified as the dominant
   lever.** Adding an explicit "people are ready to act" cue, with factual
   content and style held fixed, raises joining by +5.5 to +8.0 pp -- about the
   size of the entire surveillance effect (-8 pp). This is the higher-order
   "others are willing to act" channel the abstract previously could only
   "point to"; it is now directly measured.
2. **Coded form matters, but about half as much** (+1.9 to +4.3 pp), and partly
   independently of willingness. So coded style is a real but secondary
   suppressor, consistent with the codedness-control result.
3. The confound runs conservatively: the no-willingness rewrites state weakness
   slightly MORE (which should raise joining), yet willingness arms still join
   more -- so +7.95/+5.50 pp are lower bounds on the willingness effect.

## Joint reading

The two experiments triangulate the mechanism the manuscript argues for:
- A shows that *adding* legible willingness-to-act content raises joining by
  ~6-8 pp (the size of the surveillance effect), and that coded form is a
  smaller, separable suppressor.
- B shows that the surveillance effect *travels with* the guarded framing:
  explicit risk salience accounts for ~half (risk-only -4 pp), and removing the
  guarded framing while restoring legible actionability erases all of it.

Together they support reframing the headline from "points to a loss of public,
action-cueing value" to "the loss of legible willingness-to-act content is the
dominant identified channel, with inferred risk a secondary contributor." The
two manipulations are not on identical footing, so this is triangulation, not
an exact additive decomposition. Both are Llama 3.3 70B only.
