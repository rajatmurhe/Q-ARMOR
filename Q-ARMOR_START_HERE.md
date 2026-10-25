# Q-ARMOR — Start Here

Read [PROJECT_STATUS.md](PROJECT_STATUS.md) and [DELIVERY_README.md](DELIVERY_README.md) for the recovered working copy and current verification. The active local project is `/Users/rajatmurhe/Projects/Q-ARMOR`, outside Desktop sync.

Verification before data restoration on 23 October 2026 passed (`results/verification_20261023/verification.json`): 37 unit tests, frozen source/model/notebook checks, saved-study integrity and metric recomputation, original-architecture training feedback, real-model authenticated TCP smoke, and memory/durable crypto benchmarks. New smoke inputs are explicitly synthetic, not CICIoT2023 quality measurements.

Historical nine-run bounded-sample results are in [the study report](results/remaining_study_v2/REPORT.md). No overall macro-F1 gain is claimed. The raw archive and all three exact CSV splits have been restored and fingerprint-verified; repeated checkpoints and later XSS outputs were not committed. Raw-packet feature parity/live capture remain unverified. Optional PQC/alternate datasets remain future work.

Run `python -m tools.verify_project --output results/verification_new` or the focused `python -m tools.run_deployment_demo --synthetic --output results/demo_new` from your activated Python 3.11 environment. For dataset restoration and final packaging, use the delivery instructions.

Current restoration checks: `results/restored_verification_20261023`. Real validation replays: `results/restored_validation_replay_20261023` and `results/restored_validation_tcp_20261023`; both delivered 121 predicted alerts from 128 original validation records. These are deployment integration checks, not a new accuracy experiment.
