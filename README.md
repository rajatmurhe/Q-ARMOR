# Q-ARMOR / QuantumGuard-IoT

IoT intrusion detection research with a Stackelberg validation-feedback training loop and authenticated classical telemetry transport.

## Completed local study

The remaining offline experiments are complete: **nine runs of ten epochs**, six paired validation-only calibration probes, and three seeds per method. Each epoch uses all 34,000 balanced training rows across 34 classes. The source is a reproducible 99,514-row sample of the supplied CICIoT2023 archive.

| Method | Test macro-F1, mean ± sample SD |
|---|---:|
| Baseline | 0.53952 ± 0.00765 |
| Empirically calibrated Stackelberg | 0.53952 ± 0.00765 |
| Original-parameter Stackelberg reference | 0.53864 ± 0.01619 |

The calibrated policy chose NORMAL throughout under the declared resource/latency costs. The original-parameter reference made 23 weight updates followed by training; SQL injection and XSS F1 improved, while other classes worsened. **The study does not demonstrate an overall macro-F1 improvement.** The reference arm was declared before the new test evaluations and is not an empirically calibrated recommendation.

- [Completed experiment report](results/remaining_study_v2/REPORT.md): metrics, all-class results, calibration, probability diagnostics, plots and limitations.
- [Implementation report](RESEARCH_IMPLEMENTATION_REPORT.md): implementation and security evidence, plus historical records.
- [Research runbook](RESEARCH_RUNBOOK.md): reproduction commands, data preparation and security setup.
- [Integrity checks](results/remaining_study_v2/integrity_checks.json): all nine runs, frozen configuration, matched initial epochs and next-epoch weights.

The saved unit suite has **29 passing tests**. Committed experiment evidence includes epoch histories, confusion matrices, protocols, fingerprints, dependency versions and plots under `results/remaining_study_v2/`. The fixed inference checkpoint is in `deployment/archive_sample_seed42/`; repeated training checkpoints and preprocessing pickle files are kept locally. Earlier two-epoch/1% experiments remain local historical wiring evidence and are superseded by this study.

## Scope and remaining deployment work

These results concern a bounded archive sample, not full-dataset performance. Seventeen classes require row-level splits and may share capture characteristics across splits; no exact raw feature-row train/holdout duplicates were found. The test sample has prior exploratory usage and is not a fresh external holdout.

Replay protection, signed ephemeral X25519/Ed25519 handshake, HKDF, AES-GCM and durable counter state are implemented, with measured processing benchmarks in `results/benchmarks_m1_current/` and `results/benchmarks_m1_current_durable/`. This is classical cryptography; optional post-quantum integration remains future work.

The trained model is now packaged with frozen preprocessing, and the agent/server inference and encrypted-alert path is verified on feature records. See [Deployment runbook](DEPLOYMENT_RUNBOOK.md). Raw packet extraction parity, interface capture and full-dataset/live-network evaluation remain outstanding.

## Deployment continuation — 12 October 2026

A fixed seed-42 baseline checkpoint now has a checksummed deployment bundle. The import-safe agent consumes CSV/JSONL feature records, suppresses the actual benign label (`BenignTraffic`), sends authenticated alerts and verifies receiver acknowledgement. A real-model replay and a separate agent/server loopback TCP run each processed 128 validation records, delivered 121 predicted alerts and suppressed seven benign predictions. Nineteen focused deployment/security tests passed. These are integration checks, not additional model-quality measurements. [Commands and evidence](DEPLOYMENT_RUNBOOK.md).

## Setup

Use Python 3.11 and run these commands from the repository root:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The included deployment bundle can consume compatible feature CSV or JSONL records. Follow [DEPLOYMENT_RUNBOOK.md](DEPLOYMENT_RUNBOOK.md) for identity provisioning and agent/server commands. The dataset CSVs and raw archive are not committed. Reproducing training requires the matching data described in `data/CICIoT23_archive_sample_v1/dataset_manifest.json`.

`requirements-mac.txt` includes optional Mac notebook and dashboard dependencies. The recorded development environment is in `results/remaining_study_v2/pip_freeze.txt`.

Some historical logs and manifests contain absolute paths from the development machine. Those paths record provenance; use the repository paths and local environment above when running a fresh checkout.
