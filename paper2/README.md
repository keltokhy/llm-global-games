# Paper 2: The Auditability Frontier in Strategic Multi-Agent Communication

This folder starts the second paper as code rather than prose.

The paper asks one direct question:

```text
Can monitoring keep communication useful without letting agents move strategic
information into a channel the auditor cannot see?
```

That is the whole point. Communication can help agents coordinate. Monitoring is
supposed to make that communication legible. But if receivers can still use a
hidden part of the message, then monitoring fails operationally.

The current code has two layers:

```text
audit_game.py / ppo_agents.py       discrete-message MVP and protocol checks
continuous_audit_game.py            continuous monitored channel, the main paper spine
```

The continuous version is the sharper AI/game-theory experiment. Agents emit
messages in `R^d`. The auditor observes only a lower-dimensional projection
`z = A m`. The receiver may observe the full message, only the auditor-visible
component, or no message. This makes hidden signaling literal: a message can be
decomposed into an auditor-visible row-space component and an auditor-hidden
null-space component.

The figures should be read with four plain quantities:

```text
messages help          payoff with real messages minus payoff with shuffled messages
hidden use             how much hidden message content changes joining
offline probe reads    post-hoc probe accuracy using only z
live monitor reads     accuracy of the monitor that agents are trained to evade
```

A good monitoring design keeps `messages help` high, pushes `hidden use` down,
and keeps strategic content visible under monitoring pressure.

## Core environment

Nature draws regime strength:

```text
theta ~ Uniform(theta_low, theta_high)
```

The default range is `[-0.2, 1.8]`. This matters. Since success is determined
by whether the share joining exceeds `theta`, values below zero are almost
surely weak and values above one cannot be defeated even if everyone joins. That
prevents the degenerate policy where everyone always joins.

Each agent privately observes a noisy signal:

```text
x_i = theta + epsilon_i
epsilon_i ~ Normal(0, sigma^2)
```

Each agent has a payoff type:

```text
tau_i = (B_i, C_i)
```

where `B_i` is the benefit from joining a successful attack and `C_i` is the cost
of joining a failed attack. The code uses the type-specific cutoff:

```text
cutoff_i = B_i / (B_i + C_i)
```

Lower signals mean weaker regimes. A signal far below the cutoff is
action-favorable; a signal far above the cutoff is action-unfavorable.

## Continuous policies

The main sender policy is a Gaussian neural policy:

```text
pi_sender(m_i | x_i, tau_i)
```

where `m_i` is a continuous vector in `R^d`. The auditor observes:

```text
z_i = A m_i
```

where `A` has fewer rows than columns. This creates a literal monitored
subspace and an auditor-hidden null space.

The receiver policy is a learned binary policy:

```text
pi_receiver(a_i | x_i, tau_i, peer_message_summary)
```

where `a_i = 1` means join and `a_i = 0` means stay out.

The attack succeeds when:

```text
mean_i(a_i) > theta
```

The payoff is:

```text
B_i   if agent joins and attack succeeds
-C_i  if agent joins and attack fails
0     if agent stays out
```

## Communication regimes

The continuous runner has six frontier arms:

```text
no_comm               receiver observes no communication
visible_bottleneck    receiver observes only the auditor-visible projection
asymmetric_free       receiver observes the full message, auditor sees only z
public_monitored      visible receiver observation plus live monitor penalty
private_monitored     full receiver observation plus live monitor penalty
monitored_penalty     full receiver observation plus live monitor and hidden-influence penalties
```

The main identification contrast is `public_monitored` versus
`private_monitored`. Both arms face the same monitor trying to decode strategic
content from `z = A m`; only `private_monitored` lets the receiver see the
monitor-hidden null space. If private monitored agents keep message value high,
lower live monitor accuracy, and raise hidden use, the hidden channel is
load-bearing.

The older discrete validation suite supports:

```text
none      no useful messages
learned   sender chooses arbitrary discrete message tokens
semantic  message token is forced to equal weak / ambiguous / strong content
```

That suite is now a validation layer, not the main paper spine.

## Main metric

The first metric is whether messages actually help:

```text
messages_help = welfare(normal messages) - welfare(shuffled messages)
```

The second metric is whether the help comes from a hidden channel:

```text
hidden_use = change in join probability when hidden message content is removed
```

The third metric is whether an auditor can read the strategic content:

```text
offline_probe_reads = accuracy of a simple post-hoc probe using only z = A m
live_monitor_reads  = accuracy of the active monitor during training
```

The academic checks add three harder tests:

```text
distribution shift   re-evaluate the same trained policy under changed signal and regime primitives
cross-play           pair one seed's sender with another seed's receiver
finite auditor       estimate how many labeled examples a monitor needs to read z
```

The paper's expected tradeoff is simple: unconstrained communication should help
more, but it may rely on hidden content. Bottlenecks and penalties should reduce
hidden use, possibly at some cost to coordination.

## Run: full Paper 2 bundle

This is the paper-facing command. It trains the main PPO frontier, runs
cross-play / distribution-shift / finite-auditor checks, reruns a compact
REINFORCE diagnostic, writes Tufte-style static figures, and generates the
LaTeX manuscript.

Smoke test:

```bash
uv run python -m paper2.run_paper2 --fast --device cpu
```

M3 Ultra run:

```bash
uv run python -m paper2.run_paper2 \
  --device mps \
  --seeds 0 1 2 3 \
  --algorithm-seeds 0 1 2 \
  --updates 120 \
  --algorithm-updates 120 \
  --batch-size 6144 \
  --minibatch-size 6144 \
  --ppo-epochs 3 \
  --eval-episodes 10000 \
  --robustness-episodes 5000 \
  --auditor-episodes 6000
```

Outputs go to:

```text
paper2/output/paper2_full/
paper2/auditability_frontier.tex
paper2/auditability_frontier.pdf
```

The paper figures are exported as `.pdf`, `.png`, and `.svg`:

```text
fig1_auditability_frontier
fig2_control_metrics
fig3_crossplay_loss
fig4_distribution_shift
fig5_auditor_sample_curve
fig6_decision_rule
fig7_algorithm_diagnostic
```

Use `--plot-only` after a completed run to regenerate figures, tables, and the
manuscript from saved CSVs.

## Run: legacy continuous auditability frontier

Smoke test:

```bash
uv run python -m paper2.run_auditability_frontier --fast --device cpu
```

M3 Ultra run:

```bash
uv run python -m paper2.run_auditability_frontier \
  --device mps \
  --seeds 0 1 2 3 4 \
  --updates 180 \
  --batch-size 8192 \
  --minibatch-size 8192 \
  --ppo-epochs 3 \
  --eval-episodes 12000 \
  --robustness-episodes 8000 \
  --auditor-episodes 8000
```

Outputs go to:

```text
paper2/output/auditability_frontier/
```

Key outputs:

```text
continuous_results.csv
continuous_summary.csv
continuous_history.csv
continuous_join_curves.csv
robustness_results.csv
crossplay_results.csv
crossplay_summary.csv
auditor_sample_curve.csv
research_design.md
tables/main_results.md
tables/main_results.tex
tables/robustness_summary.md
tables/robustness_summary.tex
figures/continuous_frontier.png
figures/hidden_channel_tradeoff.png
figures/continuous_join_curves.png
figures/control_diagnostics.png
figures/robustness_shift.png
figures/crossplay_penalty.png
figures/auditor_sample_curve.png
figures/figure_notes.md
```

The runner exports matching `.svg` files for paper and slide workflows.

Use `--skip-academic-checks` for quick training-only runs. Use `--plot-only`
after a completed run to regenerate figures and tables from the saved CSVs.

## Run: discrete validation suite

Full protocol-validation suite:

```bash
uv run python -m paper2.run_full_experiments
```

Full suite plus the initial learned-agent baselines:

```bash
uv run python -m paper2.run_full_experiments --learned --learned-trainer ppo
```

On Apple Silicon, use MPS for larger PPO sweeps:

```bash
uv run python -m paper2.run_full_experiments --learned --device mps
```

Small legacy MVP run:

```bash
uv run python -m paper2.run_mvp --steps 800 --seeds 0 1 2
```

Outputs go to:

```text
paper2/output/full/
```

The first outputs are:

```text
protocol_results.csv
protocol_summary.csv
protocol_join_curves.csv
protocol_message_dictionary.csv
crossplay_summary.csv
figures/protocol_frontier.png
figures/protocol_join_curves.png
figures/crossplay_private_code.png
figures/sweep_sigma_welfare.png
figures/sweep_message_weight_strategic_opacity_gap.png
```

## What this is for

This is now a full empirical scaffold for Paper 2. The protocol suite validates
the metrics with known communication codes. The continuous runner is the paper's
main evidence: learned strategic agents, a monitored communication channel,
explicit hidden-message interventions, robustness checks, cross-play checks, and
finite-sample auditor probes.

The next substantive upgrade is not natural language. It is population training
and stronger equilibrium selection: larger seed populations, partner randomization
during training, and adversarial monitors that adapt while agents learn.
