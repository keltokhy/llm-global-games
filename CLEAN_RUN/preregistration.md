# CLEAN_RUN Pre-registration

Manifest: `/Users/khaled/GitHub/llm-global-games/CLEAN_RUN/plans/main.yaml`
Generated: `2026-05-07T17:59:16.427778+00:00`
Git commit: `85bdabcab879325e413ffac6bd5081a5281e393f`

## Arms

| arm_id | claim | model | expected_rows | primary_outcomes |
|---|---|---|---:|---|
| `signal_pure_qwen30` | language_signal | `qwen/qwen3-30b-a3b-instruct-2507` | 1000 | join_fraction_valid |
| `signal_scramble_qwen30` | language_signal | `qwen/qwen3-30b-a3b-instruct-2507` | 1000 | join_fraction_valid |
| `signal_flip_qwen30` | language_signal | `qwen/qwen3-30b-a3b-instruct-2507` | 1000 | join_fraction_valid |
| `comm_baseline_qwen30` | sender_side_surveillance | `qwen/qwen3-30b-a3b-instruct-2507` | 1000 | join_fraction_valid, message_sent |
| `surv_sender_only_qwen30` | sender_side_surveillance | `qwen/qwen3-30b-a3b-instruct-2507` | 1000 | join_fraction_valid, message_sent |
| `signal_pure_gemma4` | language_signal | `google/gemma-4-26b-a4b-it-20260403` | 1000 | join_fraction_valid |
| `signal_scramble_gemma4` | language_signal | `google/gemma-4-26b-a4b-it-20260403` | 1000 | join_fraction_valid |
| `signal_flip_gemma4` | language_signal | `google/gemma-4-26b-a4b-it-20260403` | 1000 | join_fraction_valid |
| `comm_baseline_gemma4` | sender_side_surveillance | `google/gemma-4-26b-a4b-it-20260403` | 1000 | join_fraction_valid, message_sent |
| `surv_sender_only_gemma4` | sender_side_surveillance | `google/gemma-4-26b-a4b-it-20260403` | 1000 | join_fraction_valid, message_sent |
| `signal_pure_deepseek` | language_signal | `deepseek/deepseek-v4-flash-20260423` | 1000 | join_fraction_valid |
| `signal_scramble_deepseek` | language_signal | `deepseek/deepseek-v4-flash-20260423` | 1000 | join_fraction_valid |
| `signal_flip_deepseek` | language_signal | `deepseek/deepseek-v4-flash-20260423` | 1000 | join_fraction_valid |
| `comm_baseline_deepseek` | sender_side_surveillance | `deepseek/deepseek-v4-flash-20260423` | 1000 | join_fraction_valid, message_sent |
| `surv_sender_only_deepseek` | sender_side_surveillance | `deepseek/deepseek-v4-flash-20260423` | 1000 | join_fraction_valid, message_sent |
| `signal_pure_llama4` | language_signal | `meta-llama/llama-4-maverick-17b-128e-instruct` | 1000 | join_fraction_valid |
| `signal_scramble_llama4` | language_signal | `meta-llama/llama-4-maverick-17b-128e-instruct` | 1000 | join_fraction_valid |
| `signal_flip_llama4` | language_signal | `meta-llama/llama-4-maverick-17b-128e-instruct` | 1000 | join_fraction_valid |
| `comm_baseline_llama4` | sender_side_surveillance | `meta-llama/llama-4-maverick-17b-128e-instruct` | 1000 | join_fraction_valid, message_sent |
| `surv_sender_only_llama4` | sender_side_surveillance | `meta-llama/llama-4-maverick-17b-128e-instruct` | 1000 | join_fraction_valid, message_sent |
| `signal_pure_mistral` | language_signal | `mistralai/mistral-small-2603` | 1000 | join_fraction_valid |
| `signal_scramble_mistral` | language_signal | `mistralai/mistral-small-2603` | 1000 | join_fraction_valid |
| `signal_flip_mistral` | language_signal | `mistralai/mistral-small-2603` | 1000 | join_fraction_valid |
| `comm_baseline_mistral` | sender_side_surveillance | `mistralai/mistral-small-2603` | 1000 | join_fraction_valid, message_sent |
| `surv_sender_only_mistral` | sender_side_surveillance | `mistralai/mistral-small-2603` | 1000 | join_fraction_valid, message_sent |
| `signal_pure_glm51` | language_signal | `z-ai/glm-5.1-20260406` | 1000 | join_fraction_valid |
| `signal_scramble_glm51` | language_signal | `z-ai/glm-5.1-20260406` | 1000 | join_fraction_valid |
| `signal_flip_glm51` | language_signal | `z-ai/glm-5.1-20260406` | 1000 | join_fraction_valid |
| `comm_baseline_glm51` | sender_side_surveillance | `z-ai/glm-5.1-20260406` | 1000 | join_fraction_valid, message_sent |
| `surv_sender_only_glm51` | sender_side_surveillance | `z-ai/glm-5.1-20260406` | 1000 | join_fraction_valid, message_sent |
| `no_peer_messages_deepseek` | message_value_control | `deepseek/deepseek-v4-flash-20260423` | 1000 | join_fraction_valid, message_sent |
| `degraded_messages_deepseek` | generic_message_degradation | `deepseek/deepseek-v4-flash-20260423` | 1000 | join_fraction_valid, message_sent |
| `monitored_for_research_deepseek` | observation_placebo | `deepseek/deepseek-v4-flash-20260423` | 1000 | join_fraction_valid, message_sent |
| `anonymous_aggregation_deepseek` | observation_placebo | `deepseek/deepseek-v4-flash-20260423` | 1000 | join_fraction_valid, message_sent |
| `receiver_warning_deepseek` | direct_receiver_warning | `deepseek/deepseek-v4-flash-20260423` | 1000 | join_fraction_valid |
| `no_peer_messages_llama4` | message_value_control | `meta-llama/llama-4-maverick-17b-128e-instruct` | 1000 | join_fraction_valid, message_sent |
| `degraded_messages_llama4` | generic_message_degradation | `meta-llama/llama-4-maverick-17b-128e-instruct` | 1000 | join_fraction_valid, message_sent |
| `monitored_for_research_llama4` | observation_placebo | `meta-llama/llama-4-maverick-17b-128e-instruct` | 1000 | join_fraction_valid, message_sent |
| `anonymous_aggregation_llama4` | observation_placebo | `meta-llama/llama-4-maverick-17b-128e-instruct` | 1000 | join_fraction_valid, message_sent |
| `receiver_warning_llama4` | direct_receiver_warning | `meta-llama/llama-4-maverick-17b-128e-instruct` | 1000 | join_fraction_valid |
| `no_peer_messages_qwen30` | message_value_control | `qwen/qwen3-30b-a3b-instruct-2507` | 1000 | join_fraction_valid, message_sent |
| `degraded_messages_qwen30` | generic_message_degradation | `qwen/qwen3-30b-a3b-instruct-2507` | 1000 | join_fraction_valid, message_sent |
| `monitored_for_research_qwen30` | observation_placebo | `qwen/qwen3-30b-a3b-instruct-2507` | 1000 | join_fraction_valid, message_sent |
| `anonymous_aggregation_qwen30` | observation_placebo | `qwen/qwen3-30b-a3b-instruct-2507` | 1000 | join_fraction_valid, message_sent |
| `receiver_warning_qwen30` | direct_receiver_warning | `qwen/qwen3-30b-a3b-instruct-2507` | 1000 | join_fraction_valid |
| `direct_replay_deepseek` | direct_coded_mechanism | `deepseek/deepseek-v4-flash-20260423` | 1000 | join_fraction_valid, belief_pre_success, belief_pre_join_share, belief_pre_shared_understanding, belief_pre_others_expect_join |
| `coded_replay_deepseek` | direct_coded_mechanism | `deepseek/deepseek-v4-flash-20260423` | 1000 | join_fraction_valid, belief_pre_success, belief_pre_join_share, belief_pre_shared_understanding, belief_pre_others_expect_join |
| `direct_replay_llama4` | direct_coded_mechanism | `meta-llama/llama-4-maverick-17b-128e-instruct` | 1000 | join_fraction_valid, belief_pre_success, belief_pre_join_share, belief_pre_shared_understanding, belief_pre_others_expect_join |
| `coded_replay_llama4` | direct_coded_mechanism | `meta-llama/llama-4-maverick-17b-128e-instruct` | 1000 | join_fraction_valid, belief_pre_success, belief_pre_join_share, belief_pre_shared_understanding, belief_pre_others_expect_join |
| `direct_replay_qwen30` | direct_coded_mechanism | `qwen/qwen3-30b-a3b-instruct-2507` | 1000 | join_fraction_valid, belief_pre_success, belief_pre_join_share, belief_pre_shared_understanding, belief_pre_others_expect_join |
| `coded_replay_qwen30` | direct_coded_mechanism | `qwen/qwen3-30b-a3b-instruct-2507` | 1000 | join_fraction_valid, belief_pre_success, belief_pre_join_share, belief_pre_shared_understanding, belief_pre_others_expect_join |
| `prebelief_comm_messages_excluded_deepseek` | pre_decision_belief_mechanism | `deepseek/deepseek-v4-flash-20260423` | 1000 | join_fraction_valid, belief_pre_success, belief_pre_join_share, belief_pre_shared_understanding, belief_pre_others_expect_join |
| `prebelief_comm_messages_included_deepseek` | pre_decision_belief_mechanism | `deepseek/deepseek-v4-flash-20260423` | 1000 | join_fraction_valid, belief_pre_success, belief_pre_join_share, belief_pre_shared_understanding, belief_pre_others_expect_join |
| `prebelief_surv_messages_excluded_deepseek` | pre_decision_belief_mechanism | `deepseek/deepseek-v4-flash-20260423` | 1000 | join_fraction_valid, belief_pre_success, belief_pre_join_share, belief_pre_shared_understanding, belief_pre_others_expect_join |
| `prebelief_surv_messages_included_deepseek` | pre_decision_belief_mechanism | `deepseek/deepseek-v4-flash-20260423` | 1000 | join_fraction_valid, belief_pre_success, belief_pre_join_share, belief_pre_shared_understanding, belief_pre_others_expect_join |
| `prebelief_comm_messages_excluded_llama4` | pre_decision_belief_mechanism | `meta-llama/llama-4-maverick-17b-128e-instruct` | 1000 | join_fraction_valid, belief_pre_success, belief_pre_join_share, belief_pre_shared_understanding, belief_pre_others_expect_join |
| `prebelief_comm_messages_included_llama4` | pre_decision_belief_mechanism | `meta-llama/llama-4-maverick-17b-128e-instruct` | 1000 | join_fraction_valid, belief_pre_success, belief_pre_join_share, belief_pre_shared_understanding, belief_pre_others_expect_join |
| `prebelief_surv_messages_excluded_llama4` | pre_decision_belief_mechanism | `meta-llama/llama-4-maverick-17b-128e-instruct` | 1000 | join_fraction_valid, belief_pre_success, belief_pre_join_share, belief_pre_shared_understanding, belief_pre_others_expect_join |
| `prebelief_surv_messages_included_llama4` | pre_decision_belief_mechanism | `meta-llama/llama-4-maverick-17b-128e-instruct` | 1000 | join_fraction_valid, belief_pre_success, belief_pre_join_share, belief_pre_shared_understanding, belief_pre_others_expect_join |
| `coord_with_baseline_messages_llama4` | cross_task_decomposition | `meta-llama/llama-4-maverick-17b-128e-instruct` | 1000 | join_fraction_valid |
| `coord_with_surveillance_messages_llama4` | cross_task_decomposition | `meta-llama/llama-4-maverick-17b-128e-instruct` | 1000 | join_fraction_valid |
| `bet_with_baseline_messages_llama4` | cross_task_decomposition | `meta-llama/llama-4-maverick-17b-128e-instruct` | 1000 | join_fraction_valid |
| `bet_with_surveillance_messages_llama4` | cross_task_decomposition | `meta-llama/llama-4-maverick-17b-128e-instruct` | 1000 | join_fraction_valid |
| `coord_with_baseline_messages_mistral` | cross_task_decomposition | `mistralai/mistral-small-2603` | 1000 | join_fraction_valid |
| `coord_with_surveillance_messages_mistral` | cross_task_decomposition | `mistralai/mistral-small-2603` | 1000 | join_fraction_valid |
| `bet_with_baseline_messages_mistral` | cross_task_decomposition | `mistralai/mistral-small-2603` | 1000 | join_fraction_valid |
| `bet_with_surveillance_messages_mistral` | cross_task_decomposition | `mistralai/mistral-small-2603` | 1000 | join_fraction_valid |
| `xmodel_llama4_writes_qwen30_reads_baseline` | cross_model_generalization | `qwen/qwen3-30b-a3b-instruct-2507` | 1000 | join_fraction_valid, message_sent |
| `xmodel_llama4_writes_qwen30_reads_surv` | cross_model_generalization | `qwen/qwen3-30b-a3b-instruct-2507` | 1000 | join_fraction_valid, message_sent |
| `xmodel_qwen30_writes_llama4_reads_baseline` | cross_model_generalization | `meta-llama/llama-4-maverick-17b-128e-instruct` | 1000 | join_fraction_valid, message_sent |
| `xmodel_qwen30_writes_llama4_reads_surv` | cross_model_generalization | `meta-llama/llama-4-maverick-17b-128e-instruct` | 1000 | join_fraction_valid, message_sent |
| `xmodel_mistral_writes_deepseek_reads_baseline` | cross_model_generalization | `deepseek/deepseek-v4-flash-20260423` | 1000 | join_fraction_valid, message_sent |
| `xmodel_mistral_writes_deepseek_reads_surv` | cross_model_generalization | `deepseek/deepseek-v4-flash-20260423` | 1000 | join_fraction_valid, message_sent |
| `xmodel_deepseek_writes_mistral_reads_baseline` | cross_model_generalization | `mistralai/mistral-small-2603` | 1000 | join_fraction_valid, message_sent |
| `xmodel_deepseek_writes_mistral_reads_surv` | cross_model_generalization | `mistralai/mistral-small-2603` | 1000 | join_fraction_valid, message_sent |

## Exclusion Rules

- `signal_pure_qwen30`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
- `signal_scramble_qwen30`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
- `signal_flip_qwen30`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
- `comm_baseline_qwen30`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
- `surv_sender_only_qwen30`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
- `signal_pure_gemma4`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
- `signal_scramble_gemma4`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
- `signal_flip_gemma4`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
- `comm_baseline_gemma4`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
- `surv_sender_only_gemma4`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
- `signal_pure_deepseek`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
- `signal_scramble_deepseek`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
- `signal_flip_deepseek`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
- `comm_baseline_deepseek`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
- `surv_sender_only_deepseek`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
- `signal_pure_llama4`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
- `signal_scramble_llama4`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
- `signal_flip_llama4`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
- `comm_baseline_llama4`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
- `surv_sender_only_llama4`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
- `signal_pure_mistral`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
- `signal_scramble_mistral`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
- `signal_flip_mistral`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
- `comm_baseline_mistral`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
- `surv_sender_only_mistral`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
- `signal_pure_glm51`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
- `signal_scramble_glm51`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
- `signal_flip_glm51`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
- `comm_baseline_glm51`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
- `surv_sender_only_glm51`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
- `no_peer_messages_deepseek`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
- `degraded_messages_deepseek`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
- `monitored_for_research_deepseek`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
- `anonymous_aggregation_deepseek`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
- `receiver_warning_deepseek`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
- `no_peer_messages_llama4`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
- `degraded_messages_llama4`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
- `monitored_for_research_llama4`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
- `anonymous_aggregation_llama4`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
- `receiver_warning_llama4`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
- `no_peer_messages_qwen30`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
- `degraded_messages_qwen30`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
- `monitored_for_research_qwen30`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
- `anonymous_aggregation_qwen30`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
- `receiver_warning_qwen30`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
- `direct_replay_deepseek`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
- `coded_replay_deepseek`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
- `direct_replay_llama4`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
- `coded_replay_llama4`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
- `direct_replay_qwen30`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
- `coded_replay_qwen30`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
- `prebelief_comm_messages_excluded_deepseek`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
- `prebelief_comm_messages_included_deepseek`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
- `prebelief_surv_messages_excluded_deepseek`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
- `prebelief_surv_messages_included_deepseek`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
- `prebelief_comm_messages_excluded_llama4`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
- `prebelief_comm_messages_included_llama4`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
- `prebelief_surv_messages_excluded_llama4`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
- `prebelief_surv_messages_included_llama4`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
- `coord_with_baseline_messages_llama4`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
- `coord_with_surveillance_messages_llama4`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
- `bet_with_baseline_messages_llama4`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
- `bet_with_surveillance_messages_llama4`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
- `coord_with_baseline_messages_mistral`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
- `coord_with_surveillance_messages_mistral`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
- `bet_with_baseline_messages_mistral`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
- `bet_with_surveillance_messages_mistral`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
- `xmodel_llama4_writes_qwen30_reads_baseline`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
- `xmodel_llama4_writes_qwen30_reads_surv`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
- `xmodel_qwen30_writes_llama4_reads_baseline`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
- `xmodel_qwen30_writes_llama4_reads_surv`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
- `xmodel_mistral_writes_deepseek_reads_baseline`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
- `xmodel_mistral_writes_deepseek_reads_surv`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
- `xmodel_deepseek_writes_mistral_reads_baseline`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
- `xmodel_deepseek_writes_mistral_reads_surv`: drop API-error and unparseable decisions from join_fraction_valid; report parse and API error rates by arm; trigger JOIN/STAY/drop missing-response sensitivity if arm error rates differ by more than 1 percentage point
