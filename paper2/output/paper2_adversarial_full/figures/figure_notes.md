# Figure Notes

The paper figures are designed around one claim: communication can raise strategic payoff while creating hidden receiver dependence that the auditor cannot see.

- No Messages: payoff 0.397, message value 0.000, hidden use 0.0%, offline probe 33.3%, live monitor not trained.
- Public Only: payoff 0.537, message value 0.261, hidden use 0.0%, offline probe 95.1%, live monitor not trained.
- Private Allowed: payoff 0.535, message value 0.268, hidden use 8.1%, offline probe 94.4%, live monitor not trained.
- Public Monitored: payoff 0.537, message value 0.259, hidden use 0.0%, offline probe 84.1%, live monitor 77.0%.
- Private Monitored: payoff 0.531, message value 0.267, hidden use 13.8%, offline probe 84.0%, live monitor 52.4%.
- Monitored + Penalty: payoff 0.531, message value 0.268, hidden use 1.6%, offline probe 80.2%, live monitor 61.7%.
- Largest cross-play loss: Private Allowed, 0.438 payoff points.
- Largest shift loss: Public Monitored under Harder regimes, 0.115 payoff points.
- Best finite auditor at 5,000 examples: Public Only, 93.4%.
- The algorithm diagnostic is not the main identification claim; it checks whether the measured frontier is specific to PPO.
