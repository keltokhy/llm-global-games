"""Write the Paper 2 manuscript from experiment outputs."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from paper2.paper_figures import ARM_LABELS, ARM_ORDER, SHIFT_LABELS, aggregate_metric, arm_label


METRICS = [
    "welfare",
    "message_value",
    "hidden_causal_influence",
    "auditor_content_balanced_accuracy",
    "live_monitor_content_accuracy",
    "crossplay_welfare_gap",
]


def write_manuscript(outdir: Path, *, tex_path: Path = Path("paper2/auditability_frontier.tex")) -> Path:
    outdir = Path(outdir)
    tex_path = Path(tex_path)
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    bib_path = tex_path.with_name("paper2_references.bib")
    write_bibliography(bib_path)

    results = pd.read_csv(outdir / "continuous_results.csv")
    crossplay = _read_optional(outdir / "crossplay_summary.csv")
    robustness = _read_optional(outdir / "robustness_results.csv")
    auditor = _read_optional(outdir / "auditor_sample_curve.csv")
    algorithm = _read_optional(outdir / "algorithm_results.csv")
    summary = _summary_with_crossplay(results, crossplay)
    (outdir / "tables").mkdir(exist_ok=True)
    summary.to_csv(outdir / "tables" / "paper_summary.csv", index=False)
    summary_display = _format_summary_for_tex(summary)
    (outdir / "tables" / "paper_summary.tex").write_text(
        summary_display.to_latex(index=False, escape=True, float_format="%.3f")
    )

    numbers = _numbers(results, crossplay, robustness, auditor, algorithm)
    figdir = _relpath(outdir / "figures", tex_path.parent)
    tabledir = _relpath(outdir / "tables", tex_path.parent)
    bib_stem = bib_path.with_suffix("").name

    tex = rf"""\documentclass[11pt]{{article}}

\usepackage[margin=1in]{{geometry}}
\usepackage{{amsmath, amssymb, booktabs, graphicx, hyperref, natbib, caption}}
\usepackage{{setspace}}
\usepackage{{microtype}}

\hypersetup{{colorlinks=true, linkcolor=black, citecolor=black, urlcolor=black}}
\setstretch{{1.08}}
\graphicspath{{{{{figdir}/}}}}

\title{{The Auditability Frontier in Strategic Multi-Agent Communication}}
\author{{Khaled Eltokhy}}
\date{{\today}}

\begin{{document}}
\maketitle

\begin{{abstract}}
Strategic AI agents can learn to communicate in ways that improve coordination, but monitoring becomes a control problem only when agents have an incentive to avoid being read. This paper studies that problem in a continuous-message global game. Agents privately observe noisy signals and payoff types, send learned continuous messages, and choose whether to join a collective action. A monitor observes only a lower-dimensional projection of each message. In active-monitor treatments, a live adversarial monitor predicts each sender's strategic content from that projection, and agents are penalized when the monitor assigns high probability to the truth. I define an auditability frontier that compares the payoff value of communication, the causal influence of monitor-hidden message components, and the live monitor's ability to recover strategic content. Across trained neural agents, communication raises payoff by {numbers['best_message_value']} payoff points relative to shuffled messages. Under active monitoring, private communication preserves message value ({numbers['private_monitored_value']}) while raising hidden influence to {numbers['private_monitored_hidden']} and reducing the live monitor's balanced accuracy (chance one third) from {numbers['public_live_monitor']} in the public monitored design to {numbers['private_live_monitor']}. Adding a hidden-influence penalty cuts hidden influence to {numbers['penalty_hidden']}. Cross-play and distribution-shift tests show that some high-performing protocols are brittle private codes rather than portable public languages. The result is a measurement framework for controllable strategic communication: monitoring should be evaluated by whether strategic content remains visible under pressure and whether hidden channels are behaviorally irrelevant.
\end{{abstract}}

\section{{Introduction}}

Large language models and learned agents increasingly interact with other agents in settings where messages are strategic. They may coordinate, persuade, collude, bluff, or reveal private information. A natural response is monitoring: let agents communicate, but inspect the communication channel. The premise of this paper is that monitoring is not the same as control. If a receiver can condition its action on message components outside the monitor's observation, the system may look legible while the strategically relevant information has moved somewhere else.

This paper turns that concern into a small but sharp experiment. I take a global-game coordination problem, replace prompted language agents with trained neural agents, and give them a continuous communication channel. The monitor sees only a projection of each message. The receiver may see the same projection, the full message, or no message. Because the channel is continuous, every message decomposes exactly into a monitor-visible component and a monitor-hidden null-space component. This lets us ask a direct question: can agents learn useful strategic communication while keeping the hidden part causally irrelevant for receiver behavior?

The answer is not simply yes or no. The main empirical object is an \emph{{auditability frontier}}. On one axis is message value: the payoff gain from real messages relative to shuffled messages. On the second is hidden causal influence: the change in receiver join probabilities when the hidden message component is removed. On the third is live monitor readability: whether an adversarial monitor trained during learning can recover strategic content from the visible projection. A good control design is not the one with the most communication or the easiest post-hoc probe accuracy in isolation. It is the design that delivers useful communication while keeping hidden causal influence low and strategic content visible enough for monitoring to matter.

The paper makes three contributions. First, it gives an operational definition of auditability for strategic learned communication. Auditability is measured by interventions on the hidden channel, not by visual inspection or post-hoc interpretability alone. Second, it turns monitoring into an incentive problem by adding a live adversarial monitor that agents are rewarded to evade. This makes concealment a strategic object rather than a measurement artifact. Third, it adds stress tests that matter for AI systems: cross-play between independently trained senders and receivers, distribution shifts in the underlying game, finite-sample auditor calibration, and an on-policy learning-rule diagnostic.

\section{{Related Literature}}

The economic environment is motivated by global games, where agents coordinate under private information about an underlying state \citep{{morris1998unique}}. The communication problem is connected to strategic information transmission \citep{{crawford1982strategic}}: senders may reveal information, but the meaning of a message depends on incentives and equilibrium selection. The machine-learning side connects to emergent communication in multi-agent reinforcement learning \citep{{foerster2016learning, lazaridou2017multi, lowe2017multi}} and to policy-gradient methods such as REINFORCE and PPO \citep{{williams1992simple, schulman2017proximal}}. The safety and governance motivation is close to work on specification, monitoring, and unintended channels in learned systems \citep{{amodei2016concrete, leike2018scalable}}. The distinct object here is not whether agents communicate, but whether the behaviorally relevant part of communication is visible to a monitor.

\section{{Environment}}

There are $N$ agents. Nature draws regime strength
\[
    \theta \sim U[\underline{{\theta}}, \bar{{\theta}}].
\]
Agent $i$ observes a private signal
\[
    x_i = \theta + \epsilon_i, \qquad \epsilon_i \sim \mathcal{{N}}(0,\sigma^2),
\]
and a payoff type $\tau_i=(B_i,C_i)$. Each agent chooses $a_i \in \{{0,1\}}$, where $1$ means join. The collective action succeeds when
\[
    \frac{{1}}{{N}}\sum_i a_i > \theta.
\]
If agent $i$ joins and the action succeeds, payoff is $B_i$; if it joins and fails, payoff is $-C_i$; if it stays out, payoff is zero.

Before acting, each agent emits a continuous message $m_i \in \mathbb{{R}}^d$. A monitor observes
\[
    z_i = A m_i,
\]
where $A \in \mathbb{{R}}^{{k \times d}}$ has orthonormal rows and $k<d$. This defines a visible component $m_i^V=A^\top A m_i$ and a hidden component $m_i^H=m_i-m_i^V$. The decomposition is exact: $m_i=m_i^V+m_i^H$ and $A m_i^H=0$.

The sender is a Gaussian neural policy $\pi_s(m_i\mid x_i,\tau_i)$. The receiver is a binary neural policy $\pi_r(a_i\mid x_i,\tau_i,\bar{{m}}_{{-i}})$, where $\bar{{m}}_{{-i}}$ is the mean message observed from peers. The receiver's observation varies by treatment. In the no-communication treatment, it observes zeros. In the public-only treatment, it observes only the monitor-visible message. In the private-allowed treatments, it observes the full message.

\section{{Control Designs}}

The experiment compares six designs:
\begin{{enumerate}}
    \item \textbf{{No Messages}}: receivers observe no communication.
    \item \textbf{{Public Only}}: receivers observe exactly the monitor-visible projection.
    \item \textbf{{Private Allowed}}: receivers observe the full message while the monitor sees only $z_i$.
    \item \textbf{{Public Monitored}}: receivers observe only the monitor-visible projection while a live monitor tries to decode strategic content from $z_i$ and agents are penalized when it succeeds.
    \item \textbf{{Private Monitored}}: receivers observe the full message under the same live monitor penalty. This is the main concealment treatment: agents can preserve coordination by moving information into the monitor-hidden null space.
    \item \textbf{{Monitored + Penalty}}: receivers observe the full message under active monitoring, but training also penalizes hidden causal influence.
\end{{enumerate}}

The first three designs define the passive-monitoring contrast. The two active-monitoring designs ask whether surveillance pressure induces strategic concealment. The final design asks whether reward shaping can preserve communication while making hidden channels less behaviorally relevant.

\section{{Metrics}}

The first estimand is message value:
\[
    \Delta_M = W(m) - W(\operatorname{{shuffle}}(m)),
\]
where $W$ is mean payoff and the shuffled counterfactual breaks the association between messages and the world while preserving the marginal message distribution.

The second estimand is hidden causal influence:
\[
    H = \mathbb{{E}}\left[ \left| \Pr(a_i=1\mid m_i^V+m_i^H) - \Pr(a_i=1\mid m_i^V) \right| \right].
\]
This is the central auditability statistic. It asks whether the monitor-hidden part of the message changes the receiver's action probability.

The third estimand is finite-auditor readability. A logistic probe tries to recover the sender's strategic content class from only $z_i$. The reported score is balanced accuracy, so chance is one third. This is deliberately a weak auditor; the point is not to prove optimal decoding, but to test whether the visible channel contains accessible strategic information.

The fourth estimand exists only in active-monitor treatments. Let $c_i$ denote the sender's strategic content class and let $q_\phi(c_i \mid z_i)$ be the live monitor's predicted probability of the true class. I report the live monitor's balanced accuracy, comparable to the offline probe with chance one third, and the true-class probability. These are not post-hoc diagnostics. They enter the agents' objective during training.

The stress tests are cross-play and distribution shift. Cross-play pairs one seed's sender with another seed's receiver. A large self-play to cross-play loss means the protocol is not portable. Distribution shifts re-evaluate trained policies under cleaner signals, noisier signals, knife-edge regimes, and harder regimes.

\section{{Training}}

The main agents are trained with PPO using shared sender and receiver networks. Rewards equal realized strategic payoffs minus treatment-specific communication penalties. In active-monitor treatments, a separate neural monitor $q_\phi(c_i \mid z_i)$ is trained by cross-entropy to decode strategic content from the monitored projection. Agents receive
\[
    r_i = u_i - \kappa \lVert m_i\rVert^2 - \eta \lVert m_i^H\rVert^2 - \lambda q_\phi(c_i \mid z_i),
\]
with terms omitted in treatments where the corresponding coefficient is zero. The monitor and agents are updated alternately, so the monitor is part of the training environment. The monitored-plus-penalty treatment also subtracts a penalty proportional to the contemporaneous difference between full-message and visible-message receiver probabilities. For algorithm diagnostics, I train an unclipped REINFORCE-style policy-gradient variant with the same value baselines and rewards. This diagnostic is not the paper's main identification claim; it checks whether the measured frontier is an artifact of PPO's clipped objective.

\section{{Results}}

Figure~\ref{{fig:frontier}} plots the main frontier. The no-message baseline has zero message value by construction. Communication designs move upward when messages improve payoff and rightward when receivers rely on hidden message content. The passive private channel has hidden influence of {numbers['private_hidden']}. The active-monitor comparison is sharper: Public Monitored gains {numbers['public_monitored_value']} payoff points with hidden influence of {numbers['public_monitored_hidden']}, while Private Monitored gains {numbers['private_monitored_value']} with hidden influence of {numbers['private_monitored_hidden']}. At the same time, the live monitor's balanced accuracy falls from {numbers['public_live_monitor']} to {numbers['private_live_monitor']}. That is the frontier: under surveillance pressure, private communication can preserve value while the in-the-loop monitor loses its grip on strategic content.

\begin{{figure}}[t]
    \centering
    \includegraphics[width=0.86\linewidth]{{fig1_auditability_frontier.pdf}}
    \caption{{The auditability frontier. Message value is the payoff gain from real messages over shuffled messages. Hidden causal influence is the change in receiver join probabilities when hidden message components are removed.}}
    \label{{fig:frontier}}
\end{{figure}}

Figure~\ref{{fig:metrics}} decomposes the frontier into four quantities. The public-only design is the clean monitor-aligned benchmark: what the receiver sees is what the monitor sees. The private-allowed design tests passive hidden use. Public Monitored and Private Monitored isolate the effect of active monitoring by holding the monitor penalty fixed while changing whether the receiver can see the hidden subspace. Monitored + Penalty shows that direct control can reduce hidden influence to {numbers['penalty_hidden']} while keeping the live monitor's balanced accuracy at {numbers['penalty_live_monitor']}.

A caveat sharpens the interpretation. A post-hoc logistic probe recovers strategic content from the visible projection about equally well under Public Monitored ({numbers['public_monitored_offline']}) and Private Monitored ({numbers['private_monitored_offline']}). The visible channel is therefore no less readable when the receiver gains access to the hidden subspace; what changes is the \emph{{live}} monitor, which is trained one step at a time against an adapting sender. The drop in the live monitor's balanced accuracy from {numbers['public_live_monitor']} to {numbers['private_live_monitor']}, near the one-third chance floor, thus measures the limits of an in-the-loop adversary, not evidence that strategic content has left the monitored projection. The model-free signature of a hidden channel is instead behavioral: removing the hidden component shifts receiver join probabilities by {numbers['private_monitored_hidden']} under Private Monitored, against {numbers['public_monitored_hidden']} under Public Monitored.

\begin{{figure}}[t]
    \centering
    \includegraphics[width=\linewidth]{{fig2_control_metrics.pdf}}
    \caption{{Control diagnostics by design. High message value means communication is useful. Hidden use measures receiver dependence on monitor-hidden content. Offline probe and live monitor are both balanced accuracy (chance one third); the offline probe is post-hoc readability from $z_i$, while the live monitor is the adversarial monitor trained inside the loop.}}
    \label{{fig:metrics}}
\end{{figure}}

Cross-play exposes a different failure mode. A protocol can look effective in self-play but fail when paired with another independently trained partner. Figure~\ref{{fig:crossplay}} reports self-play payoff minus cross-play payoff. The failure is not selective: every communication design loses heavily and drops below the no-communication payoff of {numbers['nocomm_welfare']}, with the largest loss of {numbers['worst_crossplay_loss']} for {numbers['worst_crossplay_arm']}. Foreign messages are worse than no messages, which is the signature of seed-specific private codes rather than a portable public language. The one partial exception is Monitored + Penalty, the most transferable design, with a cross-play loss of {numbers['penalty_crossplay_gap']} and cross-play payoff of {numbers['penalty_crossplay_welfare']}. Penalizing hidden influence appears to pull agents toward a less idiosyncratic protocol, although it remains far from fully portable.

\begin{{figure}}[t]
    \centering
    \includegraphics[width=0.86\linewidth]{{fig3_crossplay_loss.pdf}}
    \caption{{Cross-play loss. Large losses indicate protocols that work with the original partner but do not transfer to independently trained agents.}}
    \label{{fig:crossplay}}
\end{{figure}}

Figure~\ref{{fig:shift}} gives the distribution-shift check. The largest payoff loss across shifts is {numbers['worst_shift_loss']} for {numbers['worst_shift_arm']} under {numbers['worst_shift_name']}. This matters because a monitor that is only calibrated on the training distribution may miss brittle protocols that fail when the strategic environment changes.

\begin{{figure}}[t]
    \centering
    \includegraphics[width=0.86\linewidth]{{fig4_distribution_shift.pdf}}
    \caption{{Worst distribution-shift payoff loss by design. Each dot reports the largest loss relative to the training distribution across four evaluation shifts.}}
    \label{{fig:shift}}
\end{{figure}}

Figure~\ref{{fig:auditor}} turns monitor readability into a finite-sample object. The auditor observes only $z_i$. The curve asks how many labeled examples are needed before a simple probe can recover strategic content. The best design reaches {numbers['best_auditor_accuracy']} balanced accuracy at the largest sample size. This complements hidden causal influence and the live monitor: a protocol can be readable after the fact, exploitable by the receiver through hidden components, or strategically adapted against the monitor. The three quantities answer different questions.

\begin{{figure}}[t]
    \centering
    \includegraphics[width=0.86\linewidth]{{fig5_auditor_sample_curve.pdf}}
    \caption{{Finite-auditor sample complexity. Balanced accuracy is measured from the monitored projection only; chance is one third.}}
    \label{{fig:auditor}}
\end{{figure}}

Finally, Figure~\ref{{fig:algorithm}} shows the algorithm diagnostic. PPO and REINFORCE are not expected to select identical protocols. The useful result is whether the auditability metrics continue to separate public, private, and controlled channels across learning rules.

\begin{{figure}}[t]
    \centering
    \includegraphics[width=\linewidth]{{fig7_algorithm_diagnostic.pdf}}
    \caption{{Algorithm diagnostic. The same strategic designs are trained with PPO and an unclipped REINFORCE-style policy gradient.}}
    \label{{fig:algorithm}}
\end{{figure}}

\section{{Discussion}}

The key lesson is that monitoring must be evaluated as an incentive problem, not just as an interpretation problem. A passive probe can say whether a representation is readable after training. It cannot say what agents will do when being read is costly. The active monitor closes part of that gap. When only the visible projection is available to the receiver, agents can coordinate only through the same channel the monitor sees. When the receiver can see the full message, agents preserve communication value while developing behavioral dependence on the monitor-hidden subspace: hidden causal influence rises from {numbers['public_monitored_hidden']} to {numbers['private_monitored_hidden']}.

The claim should be stated carefully. The live monitor reads private-channel agents less accurately, but a post-hoc probe recovers content from the visible projection about equally well in both monitored designs ({numbers['public_monitored_offline']} versus {numbers['private_monitored_offline']}). The honest reading is therefore narrower than ``monitoring is evaded.'' What the experiment establishes is that an in-the-loop monitor is weaker than a forensic auditor, and that the robust, model-free signature of a hidden channel is behavioral rather than representational: receiver actions depend on components the monitor does not see, even when those components do not reduce the visible channel's readability. Whether a substantially stronger live monitor, trained more aggressively against the sender, closes the accuracy gap is the natural next experiment.

The second lesson is causal. Probe accuracy alone is not enough. If the auditor can decode the public part of a message but the receiver's action still changes when hidden components are removed, the system is not operationally auditable. Conversely, a strict public bottleneck is auditable by construction but may leave value on the table. The design problem is to move the frontier: preserve enough communication to coordinate while making hidden components behaviorally irrelevant under active monitoring.

The cross-play results add an equilibrium-selection interpretation. A high self-play payoff is not necessarily a stable language. It may be a private code between co-trained networks. In game-theoretic terms, independent seeds can select different communication equilibria. Population training, partner randomization, and coarse-correlated-equilibrium objectives are natural next extensions because they target that selection problem directly.

This paper deliberately starts with a stylized strategic game rather than natural language. That is the point of the design. The continuous channel gives exact interventions on monitored and hidden subspaces. Once the measurement object is clean, the same logic can be extended to richer agents: language models fine-tuned with RL, multi-round games, stronger off-policy actor-critic baselines, and institution-specific monitoring constraints.

\section{{Conclusion}}

Strategic agents can learn communication that helps them coordinate. The hard question is whether the resulting protocol remains controllable when monitoring changes incentives. This paper proposes an auditability frontier for that question. The frontier measures the payoff value of communication, hidden causal influence, and live monitor readability, then stress-tests protocols through cross-play, distribution shift, and finite-sample auditor calibration. The result is a practical research program: do not ask only whether agents communicate or whether a monitor can decode messages after the fact. Ask what agents do when being decoded is costly, and whether the parts the monitor cannot see are causally irrelevant for strategic action.

\clearpage
\section*{{Main Result Table}}
\begin{{center}}
\resizebox{{\linewidth}}{{!}}{{\input{{{tabledir}/paper_summary.tex}}}}
\end{{center}}

\bibliographystyle{{apalike}}
\bibliography{{{bib_stem}}}

\end{{document}}
"""
    tex_path.write_text(tex)
    return tex_path


def write_bibliography(path: Path) -> None:
    path.write_text(
        r"""@article{morris1998unique,
  title={Unique Equilibrium in a Model of Self-Fulfilling Currency Attacks},
  author={Morris, Stephen and Shin, Hyun Song},
  journal={American Economic Review},
  volume={88},
  number={3},
  pages={587--597},
  year={1998}
}

@article{crawford1982strategic,
  title={Strategic Information Transmission},
  author={Crawford, Vincent P. and Sobel, Joel},
  journal={Econometrica},
  volume={50},
  number={6},
  pages={1431--1451},
  year={1982}
}

@inproceedings{foerster2016learning,
  title={Learning to Communicate with Deep Multi-Agent Reinforcement Learning},
  author={Foerster, Jakob N. and Assael, Ioannis Alexandros and de Freitas, Nando and Whiteson, Shimon},
  booktitle={Advances in Neural Information Processing Systems},
  year={2016}
}

@inproceedings{lazaridou2017multi,
  title={Multi-Agent Cooperation and the Emergence of (Natural) Language},
  author={Lazaridou, Angeliki and Peysakhovich, Alexander and Baroni, Marco},
  booktitle={International Conference on Learning Representations},
  year={2017}
}

@inproceedings{lowe2017multi,
  title={Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments},
  author={Lowe, Ryan and Wu, Yi and Tamar, Aviv and Harb, Jean and Abbeel, Pieter and Mordatch, Igor},
  booktitle={Advances in Neural Information Processing Systems},
  year={2017}
}

@article{williams1992simple,
  title={Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning},
  author={Williams, Ronald J.},
  journal={Machine Learning},
  volume={8},
  pages={229--256},
  year={1992}
}

@article{schulman2017proximal,
  title={Proximal Policy Optimization Algorithms},
  author={Schulman, John and Wolski, Filip and Dhariwal, Prafulla and Radford, Alec and Klimov, Oleg},
  journal={arXiv preprint arXiv:1707.06347},
  year={2017}
}

@inproceedings{amodei2016concrete,
  title={Concrete Problems in AI Safety},
  author={Amodei, Dario and Olah, Chris and Steinhardt, Jacob and Christiano, Paul and Schulman, John and Man{\'e}, Dan},
  booktitle={arXiv preprint arXiv:1606.06565},
  year={2016}
}

@article{leike2018scalable,
  title={Scalable Agent Alignment via Reward Modeling: A Research Direction},
  author={Leike, Jan and Krueger, David and Everitt, Tom and Martic, Miljan and Maini, Vishal and Legg, Shane},
  journal={arXiv preprint arXiv:1811.07871},
  year={2018}
}
"""
        + "\n"
    )


def _read_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _summary_with_crossplay(results: pd.DataFrame, crossplay: pd.DataFrame) -> pd.DataFrame:
    summary = aggregate_metric(
        results,
        [
            "welfare",
            "message_value",
            "hidden_causal_influence",
            "auditor_content_balanced_accuracy",
            "live_monitor_content_accuracy",
            "live_monitor_content_balanced_accuracy",
            "probe_strategic_opacity_gap",
        ],
    )
    if not crossplay.empty and "crossplay_welfare_gap" in crossplay:
        summary = summary.merge(crossplay[["arm", "crossplay_welfare_gap"]], on="arm", how="left")
    else:
        summary["crossplay_welfare_gap"] = float("nan")
    return summary


def _format_summary_for_tex(summary: pd.DataFrame) -> pd.DataFrame:
    display = pd.DataFrame(
        {
            "Design": summary["arm"].map(arm_label),
            "Payoff": summary["welfare_mean"],
            "Message value": summary["message_value_mean"],
            "Hidden use": summary["hidden_causal_influence_mean"],
            "Offline probe": summary["auditor_content_balanced_accuracy_mean"],
            "Live monitor": summary["live_monitor_content_balanced_accuracy_mean"],
            "Cross-play loss": summary["crossplay_welfare_gap"],
        }
    )
    return display


def _numbers(
    results: pd.DataFrame,
    crossplay: pd.DataFrame,
    robustness: pd.DataFrame,
    auditor: pd.DataFrame,
    algorithm: pd.DataFrame,
) -> dict[str, str]:
    summary = aggregate_metric(
        results,
        [
            "welfare",
            "message_value",
            "hidden_causal_influence",
            "auditor_content_balanced_accuracy",
            "live_monitor_content_accuracy",
            "live_monitor_content_balanced_accuracy",
        ],
    )
    rows = {str(row["arm"]): row for _, row in summary.iterrows()}
    useful = summary[summary["arm"] != "no_comm"].copy()
    best_message = useful.loc[useful["message_value_mean"].idxmax()] if not useful.empty else summary.iloc[0]
    passive_private = rows.get("asymmetric_free", best_message)
    public_monitored = rows.get("public_monitored", best_message)
    private_monitored = rows.get("private_monitored", best_message)
    monitored_penalty = rows.get("monitored_penalty", best_message)
    private_hidden = float(passive_private["hidden_causal_influence_mean"])

    def _metric(row: pd.Series, column: str) -> float:
        value = row.get(column, float("nan"))
        return float(value) if pd.notna(value) else float("nan")

    public_live = _metric(public_monitored, "live_monitor_content_balanced_accuracy_mean")
    private_live = _metric(private_monitored, "live_monitor_content_balanced_accuracy_mean")
    penalty_live = _metric(monitored_penalty, "live_monitor_content_balanced_accuracy_mean")
    monitor_drop = public_live - private_live if np_finite(public_live) and np_finite(private_live) else float("nan")

    public_offline = _metric(public_monitored, "auditor_content_balanced_accuracy_mean")
    private_offline = _metric(private_monitored, "auditor_content_balanced_accuracy_mean")
    nocomm_welfare = _metric(rows.get("no_comm", best_message), "welfare_mean")

    penalty_cross_gap = "not available"
    penalty_cross_welfare = "not available"
    if not crossplay.empty and "arm" in crossplay:
        prow = crossplay[crossplay["arm"] == "monitored_penalty"]
        if not prow.empty:
            penalty_cross_gap = f"{float(prow['crossplay_welfare_gap'].iloc[0]):.3f}"
            if "cross_play_welfare" in prow.columns:
                penalty_cross_welfare = f"{float(prow['cross_play_welfare'].iloc[0]):.3f}"

    worst_cross_arm = "not available"
    worst_cross_loss = "not available"
    if not crossplay.empty and "crossplay_welfare_gap" in crossplay:
        valid = crossplay.dropna(subset=["crossplay_welfare_gap"])
        if not valid.empty:
            row = valid.loc[valid["crossplay_welfare_gap"].idxmax()]
            worst_cross_arm = arm_label(str(row["arm"]))
            worst_cross_loss = f"{float(row['crossplay_welfare_gap']):.3f}"

    worst_shift_arm = "not available"
    worst_shift_name = "not available"
    worst_shift_loss = "not available"
    if not robustness.empty:
        shift = robustness.groupby(["arm", "shift"], sort=False)["welfare"].mean().reset_index()
        train = shift[shift["shift"] == "train_distribution"][["arm", "welfare"]].rename(columns={"welfare": "train_welfare"})
        shift = shift.merge(train, on="arm", how="left")
        shift = shift[shift["shift"] != "train_distribution"].copy()
        if not shift.empty:
            shift["loss"] = shift["train_welfare"] - shift["welfare"]
            row = shift.loc[shift["loss"].idxmax()]
            worst_shift_arm = arm_label(str(row["arm"]))
            worst_shift_name = SHIFT_LABELS.get(str(row["shift"]), str(row["shift"])).lower()
            worst_shift_loss = f"{float(row['loss']):.3f}"

    best_auditor_accuracy = "not available"
    if not auditor.empty and {"train_samples", "auditor_content_balanced_accuracy"}.issubset(auditor.columns):
        max_n = int(auditor["train_samples"].max())
        final = auditor[auditor["train_samples"] == max_n].groupby("arm")["auditor_content_balanced_accuracy"].mean()
        if not final.empty:
            best_auditor_accuracy = _format_percent(float(final.max()))

    return {
        "best_message_value": f"{float(best_message['message_value_mean']):.3f}",
        "private_hidden": _format_percent(private_hidden),
        "public_monitored_value": f"{float(public_monitored['message_value_mean']):.3f}",
        "private_monitored_value": f"{float(private_monitored['message_value_mean']):.3f}",
        "private_monitored_hidden": _format_percent(float(private_monitored["hidden_causal_influence_mean"])),
        "public_monitored_hidden": _format_percent(float(public_monitored["hidden_causal_influence_mean"])),
        "penalty_hidden": _format_percent(float(monitored_penalty["hidden_causal_influence_mean"])),
        "public_live_monitor": _format_percent(public_live),
        "private_live_monitor": _format_percent(private_live),
        "penalty_live_monitor": _format_percent(penalty_live),
        "monitor_drop": _format_percent(monitor_drop),
        "worst_crossplay_arm": worst_cross_arm,
        "worst_crossplay_loss": worst_cross_loss,
        "worst_shift_arm": worst_shift_arm,
        "worst_shift_name": worst_shift_name,
        "worst_shift_loss": worst_shift_loss,
        "best_auditor_accuracy": best_auditor_accuracy,
        "public_monitored_offline": _format_percent(public_offline),
        "private_monitored_offline": _format_percent(private_offline),
        "nocomm_welfare": f"{nocomm_welfare:.3f}" if np_finite(nocomm_welfare) else "not available",
        "penalty_crossplay_gap": penalty_cross_gap,
        "penalty_crossplay_welfare": penalty_cross_welfare,
    }


def np_finite(value: float) -> bool:
    return pd.notna(value) and value == value and value not in {float("inf"), float("-inf")}


def _format_percent(value: float) -> str:
    if not np_finite(value):
        return "not available"
    return f"{value:.1%}".replace("%", r"\%")


def _relpath(path: Path, start: Path) -> str:
    return os.path.relpath(path.resolve(), start.resolve()).replace(os.sep, "/")
