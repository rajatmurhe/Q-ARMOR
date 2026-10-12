# Q-ARMOR review package

Read Q-ARMOR_START_HERE.md, then results/remaining_study_v2/REPORT.md and DEPLOYMENT_RUNBOOK.md.

This repository contains the updated source, original training notebook, research report, epoch histories, three-seed metrics, figures, paired calibration evidence, security benchmarks, tests and frozen inference deployment bundle. The initial import came from the verified review package; README.md and DEPLOYMENT_RUNBOOK.md include checkout setup instructions.

Mandatory priorities 1-7 in the teacher plan were implemented/evaluated on the bounded CICIoT2023 archive sample. The game/training loop actuated reweighting in the original-parameter reference; the measured configuration selected NORMAL under declared costs. No overall macro-F1 improvement is claimed. Some costs remain assumptions and co-evolution probabilities are diagnostics.

Optional second dataset and PQC KEM remain future work. Raw packet extraction parity, interface capture and full-dataset evaluation remain outstanding. The working deployment consumes compatible extracted feature records.

Large dataset CSVs/CSV.zip, virtual environments, private identities and repeated training checkpoints are omitted. The fixed seed-42 inference checkpoint is included in deployment/archive_sample_seed42/. Dataset fingerprints and training checkpoint/result metadata are retained. Re-training requires the matching dataset, and replay demonstration requires a compatible feature CSV.

Dependency pins are in results/remaining_study_v2/pip_freeze.txt; requirements-mac.txt is also included. Create a local Python 3.11 environment for your machine. Historical paths to /tmp environments describe the development machine. Use the local environment setup in README.md for this checkout. Identity keys must be provisioned locally; no private keys are included.
