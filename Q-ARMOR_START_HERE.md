# Q-ARMOR — Start Here

The remaining offline study is complete. Read the [completed experiment report](results/remaining_study_v2/REPORT.md) first, then the [implementation report](RESEARCH_IMPLEMENTATION_REPORT.md) and [runbook](RESEARCH_RUNBOOK.md).

Nine ten-epoch runs and six paired calibration probes used the bounded 34-class CICIoT2023 archive sample. Baseline macro-F1 was 0.53952 ± 0.00765; the calibrated policy chose NORMAL and matched it. The original-parameter reference actuated reweighting but produced no overall macro-F1 gain (0.53864 ± 0.01619). All 29 saved unit tests and result integrity checks passed.

Raw results, model checkpoints, plots, frozen protocols and source snapshots are in `results/remaining_study_v2/`. The original predefined split is absent; do not describe these sample results as full-dataset evaluation.

The deployment bundle and agent/server feature-stream inference path are now verified. Read [DEPLOYMENT_RUNBOOK.md](DEPLOYMENT_RUNBOOK.md) to run them. Raw packet-to-feature parity and live capture remain outstanding. Authenticated transport requires provisioned, pinned Ed25519 identities. Optional post-quantum integration is not implemented.
