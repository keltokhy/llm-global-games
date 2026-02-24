OVERWRITE INCIDENT BACKUP
=========================
A prior Claude session ran 200-period belief-elicitation experiments
without --append, overwriting the original 600-period pure/comm data.

These files preserve the 200-period overwrite data.

Files:
  experiment_pure_summary.csv          200 rows (5 countries x 40 periods)
  experiment_comm_summary.csv          200 rows (5 countries x 40 periods)
  experiment_pure_log.json             200-period LLM trace log (3.4M)
  experiment_comm_log.json             200-period LLM trace log (5.2M)

Duplicates (byte-identical to above, kept for provenance):
  experiment_pure_beliefs_log.json     = experiment_pure_log.json
  experiment_pure_beliefs_summary.csv  = experiment_pure_summary.csv
  experiment_surveillance_beliefs_log.json  = experiment_comm_log.json
  experiment_surveillance_beliefs_summary.csv = experiment_comm_summary.csv

The original 600-period data was restored from:
  /Users/khaled/GitHub/Global-Games-and-Coups/agent_based_simulation.zip

Date of incident: ~Feb 18, 2025
Date of recovery: Feb 23, 2026
