# Q-ARMOR research implementation report — current status

## Deployment continuation — 12 October 2026

The model artifact gap is resolved by a fixed seed-42 baseline deployment bundle with training medians, scaler, feature order, labels and integrity fingerprints. The agent and server now run inference and authenticated alert delivery from compatible feature CSV/JSONL, including benign suppression and final receiver acknowledgement. Actual checkpoint replay and separate loopback TCP processes each verified 128 records and 121 delivered predictions. Nineteen focused deployment/security tests passed. See [DEPLOYMENT_RUNBOOK.md](DEPLOYMENT_RUNBOOK.md) for commands and saved evidence.

Earlier statements below about absent deployment artifacts describe the prior implementation. Raw packet extraction parity and full-dataset evaluation remain outstanding.

## Completed continuation — 12 October 2026

The remaining **offline archive-sample study is complete**. The authoritative numerical record is the [completed study report](results/remaining_study_v2/REPORT.md). Historical entries below retain earlier limitations and incomplete runs; those completion statements are superseded by this section.

### Delivered

- Genuine measured validation F1 → Stackelberg solve → bounded class weights → next-epoch training, with actual used weights logged.
- Nine ten-epoch runs: baseline, calibrated game and an original-parameter reference, each at seeds 42–44; full 34,000-row balanced training budget each epoch.
- Six paired calibration probes across seeds 101–103 after five warmup epochs. Calibration and audit use disjoint validation halves; test scores do not fit the game configuration.
- Matched model and optimizer states for paired probes; NORMAL and reweighting both estimate before-to-after weakness reduction, avoiding an unequal comparison against assumed ordinary learning.
- Accuracy, macro/weighted F1, macro precision/recall, seven preselected class F1s, all-class tables, confusion matrices, mean/sample SD, paired differences, learning curves, and strategy probability/utility traces.
- A predeclared original-parameter reference arm demonstrating 23 actual weight updates followed by training. The calibrated policy legitimately selected NORMAL throughout under remaining assumed costs.
- 29 passing unit tests in `results/remaining_study_v2/unit_tests.txt`; completed result integrity checks in `results/remaining_study_v2/integrity_checks.json` confirm nine complete runs, frozen configuration, matched first epochs and controller-to-next-epoch weights.
- Authenticated replay-protected transport, signed handshake, persistent counters and measured local cryptographic/telemetry overhead. These security components have independent evidence; live packet-to-alert deployment is not claimed.

### Main findings

| Method | Accuracy mean ± SD | Macro-F1 mean ± SD |
|---|---:|---:|
| Baseline | 0.56083 ± 0.00789 | 0.53952 ± 0.00765 |
| Calibrated Stackelberg | 0.56083 ± 0.00789 | 0.53952 ± 0.00765 |
| Original-parameter reference | 0.55621 ± 0.01170 | 0.53864 ± 0.01619 |

Original-parameter reweighting improved SQL injection F1 (0.25439 → 0.32521) and XSS F1 (0.09423 → 0.16423), but worsened other classes. No aggregate macro-F1 benefit or statistical significance is established. Strategy probabilities evolve and remain normalized; the pure Stackelberg recommendation controls training, while probability traces are diagnostic.

Calibration estimates are limited to three warmup states and two classes selected by validation weakness. Resource/latency costs and payoff weights remain prototype assumptions. Validation-only sensitivity shows that those costs materially affect action selection; no test-based cost tuning was performed.

### Evidence boundary and remaining resources

The supplied raw archive yielded 99,514 rows across 34 classes. Seventeen classes use row-level splits; capture correlation remains possible despite zero exact raw feature-row matches between train and holdouts. The test sample was used in earlier exploratory work. Results therefore do not establish full-dataset or external-dataset generalization.

The main offline priorities are completed for this sample. Full-dataset confirmation needs the intended split and an assessed compute budget. Live deployment needs the missing packet sniffer, feature extractor, custom layers and compatible model/scaler/encoder/feature-order artifacts. A second dataset and post-quantum KEM remain optional future work.

## Historical implementation and experiment records

The dated material below is retained for provenance. References to partial calibration, 27 tests or the two-epoch study describe earlier stages, not the current completion status.

# Q-ARMOR research implementation report — 9 October 2026

## Outcome and evidence boundaries

The repository now has one maintainable epoch/validation/Stackelberg/next-epoch implementation, train-only preprocessing, a reproducible ablation runner, validation-only calibration probes, authenticated telemetry transport, security tests and measured local processing benchmarks. The original predefined CICIoT2023 split files are absent. A bounded 34-class sample was prepared from the supplied raw `CSV.zip`, and low-budget exploratory training was completed on that sample; the full-scale study and reliable performance claims remain outstanding. Reproduction commands and evidence limits are in [RESEARCH_RUNBOOK.md](RESEARCH_RUNBOOK.md).

## Audit and implementation checklist

- Inspected README, start instructions, live agent/server, classical/legacy crypto, attacker/defender/payoff/game/controller, main notebook, original notebook backup and existing bridge/co-evolution tests.
- Searched project directories (excluding virtual environment and caches) for CSVs and configuration; found no original split CSVs. QARMOR_DATA_DIR was unset and original Kaggle paths absent.
- Inspected the original 43-cell backup and saved `Training_Report/final-cdl-fl-work (1)_PRE_RESEARCH_20261009.ipynb` before notebook changes.
- Moved duplicated training logic into `game_theory/ids_training.py`, and the original CNN/LSTM/attention builder into `tools/ids_model.py`. The notebook and standalone runner call these modules.
- Replaced preprocessing leakage with training-only imputation, label encoding, constant-column selection, mutual information and scaling in `tools/data_pipeline.py`. Preserve original holdout rows/splits; training resampling happens once.
- Removed fabricated feature-importance scores and feature-list overwrites from active notebook execution. Historical training/federated variants remain as inactive markdown plus complete backups; stale outputs were cleared.
- Added bounded mean-normalized class-weight scaling, explicit no-op/unimplemented action logging, validation-deterioration rollback, input validation and stable exponentiated replicator updates.
- Added fair baseline/proposed seed runner, all-class metrics, seven pre-established candidate-class metrics, individual and aggregate tables, checkpointing, confusion matrices, manifests, histories and plots.
- Added paired validation-only reweighting probes and synthetic payoff sensitivity diagnostics. Final empirically calibrated settings remain blocked by missing data.
- Added AES-GCM authenticated counter envelopes and atomic SQLite sender/replay state; signed ephemeral X25519/Ed25519 handshake; bounded framed socket telemetry; pinned identity configuration and a local demo-key provisioning helper.
- Updated live agent/server to use authenticated transport. Legacy classical/HybridCrypto APIs remain for compatibility and are explicitly outside the new replay/authentication guarantees.
- Added crypto/telemetry benchmark tools with repeated timings, raw data, medians/p95, throughput, overhead and platform/resource metadata.

## Important findings

### Frozen original strategies

Original `default_strategy_matrices` derives weights from class counts. When hybrid balancing gives equal counts, normalized minority importance is uniform, and every defender/attacker strategy row becomes uniform. The strategy utility dot products are therefore equal for **any** measured class-performance vector. Uniform probabilities correctly remain fixed under the original replicator update. Reusing the latest probability state or changing entropy clipping alone cannot resolve this lack of strategy distinction.

The replacement computes expected utilities for four real defender actions and every non-benign attack class using the exact action evaluator used by Stackelberg solve. F1 changes affect residual risks and utilities. Stable shifted exponentials avoid division by a tiny shifted mean; exploration mixture preserves a true probability floor. Equal expected utilities can still legitimately produce fixed distributions. Probability diagnostics record normalization, changes and utility vectors without introducing artificial motion.

`results/component_tests` contains **synthetic state-transition evidence only**: probability/utility/F1 plots, histories, original balanced-row symmetry assertion and parameter sensitivity. These are component diagnostics, not model research results.

### Default game may not reweight

On a synthetic uniform-severity grid of 101 F1 states, original payoff settings chose NORMAL for 49 states and TARGETED_DEFENSE for 52; CLASS_REWEIGHTING and STRICT_THRESHOLD were selected zero times. This grid is not proof about every possible heterogeneous severity/state, but it exposes why the proposed default can have no actuator effect. Threshold and targeted defences are modelled by the game but unimplemented as training actuators. Recommendations remain logged honestly with no applied action.

Reweighting unit tests use an explicitly labelled configurable scenario with higher targeted-defence resource cost. The real solver still evaluates all actions and chooses the maximum utility. This exercises the actuator and feedback wiring; it is **not calibration, a recommended production setting, or evidence of real IDS improvement**. Existing game assumptions were not silently changed to force the desired result.

### Environment/filesystem issues

The original `.venv` import failed with filesystem `TimeoutError: [Errno 60]`. Several original source/cached files also intermittently failed reads. Verification uses a fresh isolated Python 3.11 ARM64 environment at `/tmp/qarmor-research-venv` and separate `/tmp/qarmor-pycache`.

Attacker/payoff sources had been inspected in full before later read failures; they were rematerialized atomically from the inspected APIs. The original risk formula is preserved. The payoff now subtracts the already severity-weighted residual attacker utility from severity, correcting the previous double application of severity for nonuniform severity values; defaults with severity=1 are unchanged. A regression test covers this correction, and payoff parameters are validated. The original server was inspected before replacement. Original README became unreadable during the update and was preserved by rename as `README_HISTORICAL.md`; the new README points to current evidence and explicitly identifies old metrics/PQC assertions as unverified historical claims.

Model_Keras and Model_&_files contain confusion matrices but no deployable model/scaler/encoder/feature artifacts, so live inference/packet-to-alert benchmarking is also blocked.

Existing `tools/test_coevolution` runs successfully. `tools/test_34class_bridge` and `game_theory/test_game` were attempted but their source reads failed with errno 60; do not interpret those attempts as passed. New tests independently cover game action evaluation, API dimensions, controller wiring and strategy updates. `tools/check_environment.py` now bounds dependency imports and smoke-tests the authenticated handshake/channel instead of treating an unsigned legacy exchange as authentication.

## Training and evaluation design

The same compiled model/optimizer persists between epochs. Every fit uses exactly one epoch; full original validation predictions produce precision, recall and F1 for all label IDs, including zero-prediction classes. Those measured values enter solve through the controller. Returned effective weights are used on the next epoch. Each epoch saves actual used weights, recommendation, anticipated attack, selected target, all action evaluations, utilities, distributions, metrics, applied action and scale-change reason.

Weight scales remain in [1, max_scale] and are normalized with ordinary base weights to mean one. If the prior increase is followed by material target or aggregate validation-F1 deterioration, the scale is restored and further increase withheld for that decision. This does not establish causality or roll back the model. Unsupported actions and saturated weight changes are explicit no-ops.

The baseline/proposed runner resets model and optimizer for each method/seed, reuses per-seed train-only preprocessing/resampling, matches sample orders/subsampling and training budgets, selects checkpoints using validation macro-F1, and evaluates final test once. Outputs include accuracy, macro/weighted-F1, macro-precision/recall, all-class metrics, seven candidate class F1s, confusion matrices and mean/sample-std/count. Test data cannot enter adaptive control or calibration.

The seven-class list reuses a pre-existing **synthetic wiring-test candidate list**; it is not a measured weakest-seven ranking. No test-derived class selection was introduced. The full-budget three-seed evaluation on the intended original split remains unexecuted until those split files and an assessed compute budget are available; a low-budget three-seed run on the archive-derived sample is documented in the local continuation below.

Calibration probes branch from identical warmup model/optimizer states, compare ordinary and scaled class weights for one matched epoch, and record validation target/macro-F1 and false-alarm effects. A bounded candidate effectiveness mapping is emitted separately; negative effects remain in raw records. No final parameter setting is presented as optimal or experimentally justified without data. Other action costs/effects remain relative prototype assumptions. Alternative datasets and post-quantum KEM integration remain optional/unimplemented.

## Security scope

The canonical signed handshake binds identities, protocol, fresh challenges and both ephemeral keys. Pinned Ed25519 peer signatures are verified before session acceptance; HKDF binds shared X25519 secret to transcript hash. Direction/session identities derive distinct AES keys. Envelope AAD authenticates session, peers and monotonic counter. Tag failures do not update trusted replay state. Exact replays and stale counters are rejected. SQLite counter reservations and authenticated receive commits are atomic; durable state and concurrency are tested.

Live connections always establish fresh ephemeral sessions; memory-backed sessions are never resumed after restart. Persistent direct channels require shared durable state and protection against counter-state rollback. Fresh key/session rotation is required after state loss, before counter budget exhaustion and according to traffic-volume policy. The implementation is a tested research transport, not a production security certification. Provisioning, rate limiting, multi-device trust registry, deployment audit and rollback-resistant state storage remain external requirements. X25519/Ed25519 are classical, not post-quantum. See the [cryptography AES-GCM requirements](https://cryptography.io/en/stable/hazmat/primitives/aead/) and [X25519 KDF guidance](https://cryptography.io/en/stable/hazmat/primitives/asymmetric/x25519/).

## Validation records and measured benchmarks

The final verification summary and benchmark table are generated below from saved outputs. Failed early attempts remain in `results/research` for audit. Only successful final runs are reported as passed. No benchmark timings are IDS accuracy/F1 results.
<!-- GENERATED EVIDENCE -->

- **Unit tests:** 27 passed; see `results/research/unit_tests.txt`. Includes negative crypto cases, SQLite concurrency/persistence, model-spy feedback, normalization, matched baseline budgets/sampling, aggregation and train-only leakage tests.
- **Existing co-evolution test:** passed. Existing game and 34-class scripts remain unreadable with errno 60; attempts are preserved and not claimed passed.
- **Compilation/notebook:** changed modules compiled; all active notebook cells parsed, import ordering checked and exported cell27 matched. Full notebook execution remains blocked by missing original CSVs.
- **synthetic_smoke:** passed; synthetic only. Small Keras integration pipeline.
- **synthetic_reweight_fixture:** passed; synthetic only. Small Keras integration pipeline.
- **synthetic_original_model:** passed; synthetic only. Original architecture, three actual Keras fit/predict/feedback epochs.
- **synthetic_original_reweight_fixture:** passed; synthetic only. Original architecture, three actual Keras fit/predict/feedback epochs.
- **Environment audit:** core dependencies and authenticated smoke pass; optional live-app dependencies scapy, streamlit and seaborn are absent from the isolated verification environment, so the full environment check correctly exits with failure. Metal is not enabled in the temporary environment. See `results/research/environment_check.txt`.
- **Full-budget original-split ablation/calibration:** not completed. A bounded archive-sample three-seed exploratory run and a partial one-seed calibration are documented in the local continuation below; they are not final evidence.

### Measured local ARM64 processing timings

Python 3.11.4, cryptography 46.0.7; in-memory 1,000 repetitions, durable SQLite 500 repetitions; 25-message warmup. Values below are microseconds. Raw samples, wire overhead, throughput and resource metadata are in the benchmark JSON/CSV files.

| Workload | State | Median µs | p95 µs |
|---|---|---:|---:|
| AES envelope 64 bytes roundtrip | memory | 42.08 | 113.60 |
| AES envelope 1024 bytes roundtrip | memory | 61.46 | 78.06 |
| AES envelope 16384 bytes roundtrip | memory | 355.58 | 412.66 |
| telemetry_json | plain JSON | 6.54 | 7.04 |
| telemetry_authenticated | memory | 49.25 | 55.25 |
| authenticated_handshake | fresh handshake | 2074.33 | 2348.06 |
| AES envelope 1024 bytes roundtrip | durable SQLite | 965.46 | 1211.87 |
| telemetry_json | plain JSON | 9.35 | 17.17 |
| telemetry_authenticated | durable SQLite | 1394.83 | 2094.34 |

These measurements include local envelope serialization, crypto, replay checks and (durable mode) state commits. They exclude network delivery, model inference and disk alert logging. Different runs are subject to host scheduling and load.

### Remaining work requiring original resources

1. Restore the original train/validation/test CSVs and assess RAM/runtime with the documented real-data pilot.
2. Run validation-only paired calibration/sensitivity probes, review default non-actuated recommendations and freeze empirically justified parameters.
3. Execute the actual three-seed baseline/proposed study and produce real aggregate/per-class/confusion results; a method that never actuates reweighting must be reported as such.
4. Restore deployed model/scaler/encoder/features and provision trusted identities before live IDS/network benchmarking.
5. Restore unreadable legacy test files and run those exact scripts; optional PQ KEM and alternate datasets remain unimplemented.


## Local continuation — 12 October 2026

This section supersedes the earlier “no real CICIoT2023 experiments” statements above for the bounded archive sample only. The supplied `CSV.zip` is the raw 34-class archive, not a predefined train/validation/test release. `tools/prepare_archive_splits.py` processed it in chunks without extracting or modifying it and produced `data/CICIoT23_archive_sample_v1/`: 66,342 train, 16,586 validation and 16,586 test rows (99,514 total). The archive SHA-256 and class/source assignment are recorded in `dataset_manifest.json`. Whole capture files are held out when a class has at least three source CSVs; 17 classes with one or two source files use deterministic row-level splits and may have capture overlap. This is a bounded development sample, not the full dataset or the original split.

### Sample experiments

A two-epoch single-seed pilot at 1% of the balanced training rows per epoch exercised the complete measured validation-F1 → Stackelberg solve → class-reweighting → next-epoch training path. Epoch 1 recommended `CLASS_REWEIGHTING` for `DDoS-RSTFINFLOOD` at scale 1.1481, and epoch 2 used updated weights. Its test accuracy was 0.0302 and macro-F1 0.0020; these near-chance results show wiring only.

A separate three-seed exploratory run (42, 43, 44), also two epochs at 1% training fraction and balanced target 1,000, completed all six baseline/proposed runs. The validation-calibrated game selected `NORMAL` in every proposed epoch, so baseline and proposed were identical: mean accuracy 0.05085 (sample SD 0.01860), macro-F1 0.01063 (SD 0.00818), weighted-F1 0.01089 (SD 0.00838). All seven preselected weak-class F1 values were zero. This is not evidence of useful IDS accuracy or an improvement from the game; the model was severely undertrained and the calibrated policy made no intervention. Outputs are in `results/archive_sample_v1_exploratory_3seed/`.

The higher-budget validation-only calibration was interrupted after paired probes for seed 42 because runtime was excessive on the available 8 GB M1 Air. At scale 1.15 the target F1 change was 0 and macro-F1 changed by −0.000068; at scale 1.5 target F1 rose by 0.115868 and macro-F1 by 0.000823. These two one-seed observations produced only a provisional candidate in `results/archive_sample_v1_calibration_measured/`; the candidate is not final, generalizable, or independently validated. It was not used to make test-driven changes.

### Security measurements and verification

On macOS 14.6.1 / arm64, Python 3.11.16 and cryptography 46.0.7, 1,000-sample in-memory benchmarks measured authenticated 112-byte telemetry at 44.292 µs median / 50.185 µs p95, and a fresh signed handshake at 2.051 ms / 2.223 ms. X25519 exchange was 264.083 µs median and HKDF-SHA256 9.708 µs. With durable SQLite FULL counter state, authenticated telemetry measured 949.647 µs / 1.097 ms (500 samples). These are local processing measurements, not network or IDS inference timings. Raw samples and overhead are in `results/benchmarks_m1_current/` and `results/benchmarks_m1_current_durable/`. The current complete test suite passed: 27 tests, including replay rejection, tampering, handshake validation and persistence.

### Remaining integration boundary

This workspace has no `Model_Keras/` deployment artifacts and only `Agent/ids_agent.py` under `Agent/`; its referenced `Agent.packet_sniffer`, `Agent.feature_extractor` and `Agent.custom_layers` modules are absent. Therefore the authenticated socket transport is testable and the server is implemented, but packet capture → feature extraction → trained inference → alert delivery cannot currently run end to end. A trained model, matching scaler/label encoder/feature order, and those packet modules must be restored before claiming a live IDS deployment.

The complete full-budget CICIoT study and multi-seed calibration remain unrun. Current best evidence is the bounded, undertrained exploratory run above; no production or publication-level model performance claim is supported. Current source/document hashes are recorded in `results/research/current_working_manifest.json`; the original recovery manifest is preserved separately.
