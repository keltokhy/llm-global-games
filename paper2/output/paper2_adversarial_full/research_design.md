# Paper 2 Research Design

Question: can trained strategic agents communicate useful private information while remaining auditable when monitoring is itself part of the incentive environment?

Main estimands:

- Message value: payoff with real messages minus payoff after messages are shuffled.
- Hidden causal influence: change in receiver join probability when monitor-hidden message components are removed.
- Offline auditor readability: finite-sample balanced accuracy for recovering strategic content from the monitored projection.
- Live monitor readability: accuracy of the adversarial monitor that trains against the agents in active-monitor treatments.
- Cross-play loss: self-play payoff minus payoff when senders and receivers come from different seeds.

The paper's identification logic is the comparison between Public Monitored and Private Monitored. Both arms face the same live monitor penalty, but only Private Monitored gives the receiver access to the monitor-hidden null space. If hidden use rises and live monitor accuracy falls in Private Monitored, concealment is load-bearing rather than a passive measurement artifact.

Public Only and Private Allowed are passive baselines. Monitored + Penalty asks whether reward design can preserve message value while pushing hidden causal influence back down.

The algorithm diagnostic compares PPO against a REINFORCE-style policy gradient. It is a robustness diagnostic for the training procedure, not the main theoretical claim. The next algorithmic extension is an off-policy actor-critic baseline such as SAC/DDPG adapted to the continuous sender channel.
