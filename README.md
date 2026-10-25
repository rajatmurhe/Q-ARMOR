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

## Recovery and fresh verification — 21 October 2026

The active local working copy is `/Users/rajatmurhe/Projects/Q-ARMOR`, outside Desktop/iCloud eviction. On this Mac, `.venv` links to `/Users/rajatmurhe/.venvs/qarmor`; recipients should create their own environment. The earlier Desktop recovery remains historical.

The deleted project was restored from GitHub commit `70f9176d6a24907899adcc52890c50d7c02f4ead`. The deployment checkpoint and preprocessing JSON match their original fingerprints. Saved research scores above are historical measurements, not new runs. The original raw archive and exact split CSVs were restored on 23 October and match all saved fingerprints. They are included in the delivery ZIP. Repeated training checkpoints and later XSS intervention outputs remain unavailable.

The project supports a runnable checkpoint/secure-network demonstration. The explicit synthetic mode remains available for integration checks:

```sh
source .venv/bin/activate
python -m tools.check_environment
python -m tools.verify_project --output results/verification_new
# Or only the real checkpoint and authenticated TCP integration:
python -m tools.run_deployment_demo --synthetic --output results/demo_new
```

`--synthetic` is explicit. The generated schema inputs, delivered alerts and verification records are marked synthetic. They test actual checkpoint inference, signatures, encryption, replay state and receiver acknowledgement, not model accuracy, true attacks or packet-feature parity. No packet-capture privileges are needed. Supply `--features /path/to/features.csv` to demonstrate compatible real feature records instead. Without either input, a missing original validation CSV fails clearly.

Recover the **same** sample when original files become available:

```sh
python -m tools.restore_data --archive /absolute/path/to/CSV.zip
# Only for a Kaggle ZIP whose class-folder prefix is data/:
python -m tools.restore_data --mirror-archive /path/to/kaggle-download.zip
# Or restore previously saved exact split CSVs:
python -m tools.restore_data --splits /absolute/path/to/CICIoT23_archive_sample_v1
```

The original-archive mode checks raw-archive and generated-split SHA-256 hashes against the frozen study manifest. Explicit mirror mode accepts a different ZIP container only after all three generated split files match their original fingerprints. Mismatched data is rejected; existing mismatched CSVs are never overwritten. It cannot reconstruct deleted training checkpoints or the original Kaggle predefined split. If downloading again, obtain the correct archive through the [official CICIoT2023 publisher](https://www.unb.ca/cic/datasets/iotdataset-2023.html), then verify it locally.

Create a new delivery ZIP with all available source, untracked data/results, checkpoint and Git history:

```sh
python -m tools.package_project --output ../Q-ARMOR_Project_Delivery.zip
```

The packager reads the ZIP back and verifies every included file. It includes a SHA-256 manifest and produces an archive checksum. Machine-specific virtual environments, caches, private identities and counter databases are deliberately excluded; their paths are listed in the manifest. Setup instructions and recorded dependency versions replace the virtual environment for portability. See [DELIVERY_README.md](DELIVERY_README.md) and [current verification](results/verification_20261021/verification.json) for exact evidence and remaining blockers.
