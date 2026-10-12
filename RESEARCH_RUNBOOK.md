# Q-ARMOR research execution

## Deployment continuation — 12 October 2026

The model artifact gap is resolved by a fixed seed-42 baseline deployment bundle with training medians, scaler, feature order, labels and integrity fingerprints. The agent and server now run inference and authenticated alert delivery from compatible feature CSV/JSONL, including benign suppression and final receiver acknowledgement. Actual checkpoint replay and separate loopback TCP processes each verified 128 records and 121 delivered predictions. Nineteen focused deployment/security tests passed. See [DEPLOYMENT_RUNBOOK.md](DEPLOYMENT_RUNBOOK.md) for commands and saved evidence.

Earlier statements below about absent deployment artifacts describe the prior implementation. Raw packet extraction parity and full-dataset evaluation remain outstanding.

The active implementation preserves provided split files when present. This workspace contains the raw class-folder `CSV.zip`, not the original predefined split CSVs. A bounded sample was prepared from that archive and low-budget exploratory training was run; full-scale results are not claimed. Synthetic tests, archive-sample model metrics and local security microbenchmarks are separate evidence categories. See the current execution record at the end of this runbook.

## Environment and tests

Use Python 3.11 and `requirements-mac.txt`. The original `.venv` produced filesystem read timeouts during this audit. Verification used `/tmp/qarmor-research-venv`, with `PYTHONPYCACHEPREFIX=/tmp/qarmor-pycache` to avoid unreadable cached files. The temporary environment can disappear; reproduce with a healthy local virtual environment. CPU smoke tests do not verify Metal performance.

```sh
python -m unittest discover -s tests -v
python -m tools.test_34class_bridge
python -m tools.test_coevolution
python -m game_theory.test_game
python tools/check_environment.py
python -m tools.game_diagnostics
python -m tools.smoke_training
python -m tools.smoke_training --original-model --output results/synthetic_original_model
python -m tools.smoke_training --reweight-fixture --output results/synthetic_reweight_fixture
python -m tools.smoke_training --original-model --reweight-fixture --output results/synthetic_original_reweight_fixture
```

The reweighting smoke tests use a **unit-test payoff scenario** with a higher TARGETED_DEFENSE resource cost. All actions are evaluated; no solver recommendation is mocked. This scenario exercises the reweighting actuator. It is **not** measured game calibration and must not be used to claim real experimental improvement.

## Original data and fair ablation

Set a directory containing exactly the existing split layout:

```text
CICIOT23/
  train/train.csv
  validation/validation.csv
  test/test.csv
```

The `CSV.zip` supplied in this workspace is the raw class-folder archive, not the predefined split above. A bounded 34-class development sample is available at `data/CICIoT23_archive_sample_v1/`, generated with `tools/prepare_archive_splits.py` using a fixed seed and per-class caps of 2,000 train, 500 validation and 500 test rows. The actual sample contains 99,514 rows (66,342 train, 16,586 validation and 16,586 test). The sampler reads ZIP members in chunks and leaves the archive untouched. Where a class has at least three source CSVs, whole source files are assigned to separate splits; the 17 classes with only one or two source files use row-level assignment and can have capture overlap. `dataset_manifest.json` records the archive SHA-256, assignment and actual per-class row counts. This is a development sample, **not** the original predefined split or a full-scale final result. In particular, `Uploading_Attack` and `Recon-PingSweep` have fewer available rows than the requested caps. Do not replace the original-split study with this sample without labeling it clearly.

On this 8 GB Mac, start with the bounded sample and a small epoch/batch budget. The original full-split workflow loads all three CSVs in memory and balances toward 300,000 rows per class, which is beyond a safe first run on this machine. The separate local sample supports a 3-seed development ablation without extracting the full archive.

```sh
export QARMOR_DATA_DIR=/absolute/path/to/CICIOT23
python -m tools.run_ablation --check-data
# First assess RAM and runtime with this reduced TRAINING budget:
python -m tools.run_ablation --seeds 42 --epochs 2 --minority-target 1000 --majority-target 1000 --batch-fraction 0.01 --output results/real_pilot
# Full comparable baseline/proposed budget (three seeds):
python -m tools.run_ablation --seeds 42 43 44 --epochs 10 --batch-fraction 0.10 --output results/ablation
```

Without the setting, original `/kaggle/input/CICIOT23` paths remain supported. No automatic download or replacement occurs. Validation/test rows are preserved, including duplicates. Imputation, constant-feature removal, mutual information, label encoding and scaling fit training only. Unknown holdout labels fail explicitly. CICIoT features must be numeric; incompatible schemas fail instead of inventing an encoding. SMOTE/undersampling use training only, with singleton minorities rejected. The 300,000-per-class default matches the last original balancing variant; 34 classes at 25 float32 features can exceed a gigabyte before temporary copies, neighbours, original CSV frames and model memory. The pilot changes training balancing/budget and is not the final ablation.

Both methods use the extracted original CNN/LSTM/attention architecture, a fresh seeded initialization and Adam optimizer, the same preprocessing and balanced arrays per seed, identical seeded per-epoch row orders/subsamples, and the same number of one-epoch `fit` calls. The baseline keeps mean-normalized ordinary class weights fixed. Proposed updates only following a genuine CLASS_REWEIGHTING recommendation. Both select the best checkpoint using validation macro-F1. Final test predictions occur once after checkpoint restoration and never enter game decisions or tuning.

Each seed/method writes config, epoch history, best weights, test metrics, per-class metrics, confusion matrix CSV/plot, validation F1 plots and (proposed) game probability/utility plots. Root tables contain individual metrics and mean/sample-standard-deviation/count, with incomplete runs recorded. Do not compare partial groups as a completed three-seed study. Seeds and requested deterministic operations are recorded; exact cross-device reproducibility, especially on Metal, is not guaranteed.

The seven named classes reuse the existing `tools/test_34class_bridge.py` list: Backdoor_Malware, BrowserHijacking, CommandInjection, SqlInjection, Uploading_Attack, VulnerabilityScan, XSS. That list was used in a synthetic wiring test. It is a pre-established candidate weak-class set, **not evidence that these are the seven weakest real classes**. Report this basis, or pre-register a replacement based on training/validation before final test evaluation.

## Feedback-loop semantics

`game_theory/ids_training.py:train_ids` trains the same compiled model one epoch, predicts the full validation split, calculates precision/recall/F1 for every output ID with `zero_division=0`, passes the measured class-name mapping into `StackelbergIDSController.step`, and applies returned weights on the next epoch. The controller calls the real `StackelbergSecurityGame.solve`. The research training configuration restricts its available actions to `NORMAL` and `CLASS_REWEIGHTING`, because those are the only actions implemented in the IDS loop; the standalone game still supports all four theoretical actions. Co-evolution utilities use the same restricted action evaluator and ordering as the training recommendation. Strategy distributions are diagnostic; the actual actuator follows the pure Stackelberg recommendation, not a sampled action.

Scales start at 1 and increase by `scale_step * (1 - target F1)`, capped at `max_scale` (default 2.5). Effective weights are base-weight times class scale, normalized to mean one. Following target-F1 or mean-validation-class-F1 deterioration beyond `deterioration_tolerance` (default .01), the previous scale increase is rolled back and further increase is suppressed for that decision. This is a conservative validation heuristic, not a causal attribution or automatic model rollback. No automatic unbounded scaling or test-based correction exists. Saturated scales log an unchanged reason and no applied action. Unsupported threshold/targeted actions remain recommendations with `applied_action=null`.

The standalone theoretical game still supports four actions. For the actual training study, the controller is explicitly restricted to `NORMAL` and `CLASS_REWEIGHTING`; it never claims to actuate an unsupported IDS action. The resource/latency/false-positive costs and effectiveness values still require validation-only calibration. Do not alter costs solely to force a desired result.

## Validation-only calibration and sensitivity

```sh
python -m tools.calibrate_game --seeds 42 43 44 --scales 1.15 1.5 2.0 --output results/calibration
python -m tools.game_diagnostics
# Only after reviewing measured validation probes and freezing a configuration:
python -m tools.run_ablation --game-config results/calibration/candidate_game_config.json --output results/calibrated_ablation
```

The current calibration tool never reads the test CSV. It uses a fixed stratified half of validation (seed 2026) to select the weakest non-benign class after the requested warmup epochs. The other half audits the effect without fitting the candidate. Every control/reweighted branch starts with identical model **and optimizer state**, matched samples and random seeds, and one complete training epoch. The control branch may be reused across scales because its starting state and training sequence are identical. A requested scale of 1.15 means an actual first-step multiplier of `1 + 0.15 * (1 - target_F1)`, matching the controller's first update.

Both NORMAL and CLASS_REWEIGHTING effectiveness now estimate before-to-after target-weakness reduction; NORMAL includes ordinary learning. Signed changes are averaged before bounding the final estimate to [0, .95]. Negative observations remain in the records. A constant effect is fitted because three seeds do not identify a reliable two-parameter weakness curve. False-positive costs use observed benign false-alarm rates. Resource/latency costs and utility weights remain prototype assumptions. `--candidate-scale` is fixed before probes; other scales supply sensitivity evidence. Audit scores and test scores never select the candidate.

`--device cpu --threads 4` avoids Metal execution overhead measured for this architecture on the M1 Air. `--resume` reuses completed seed/probe records when arguments and input fingerprints match; an interrupted warmup is rerun from its fixed seed rather than claiming weights alone restore random-generator state. Ablation `--resume` skips complete method/seed runs; an interrupted method restarts from its initialization. Existing completed results are not overwritten with incompatible configurations.

Defender security benefit is `severity - residual_attacker_utility`: attacker residual risk already includes severity, so severity must not multiply that risk again.

`--game-config` accepts `payoff` weights, per-action `resource_cost`, `latency_cost`, `false_positive_cost`, and `effectiveness` pairs `[intercept, weakness_slope]`. No settings have been selected as empirically optimal without original data. Defaults (.15 step, 2.5 cap, .5 replicator learning rate) are bounded prototype settings. Component sensitivity explores resource penalty weights; the paired probes explore scaling effects using actual validation predictions when data becomes available.

## Authenticated live telemetry

Live Agent/Server use `crypto/secure_transport.py`, not the legacy unsigned exchange. Provision Ed25519 identity private keys and pin the expected peer public keys out of band. Verify public-key fingerprints through a trusted administrative channel; do not accept a public key from the handshake as its own trust anchor. The helper below creates **local demonstration identities**, not a key-distribution service:

```sh
python -m tools.provision_demo_identities --directory /absolute/private/qarmor-demo-keys
# Server terminal:
export QARMOR_IDENTITY=server QARMOR_PEER_IDENTITY=agent
export QARMOR_SIGNING_KEY=/absolute/private/qarmor-demo-keys/server.private.pem
export QARMOR_TRUSTED_PEER_KEY=/absolute/private/qarmor-demo-keys/agent.public.pem
python -m Server.server
# Agent terminal:
export QARMOR_IDENTITY=agent QARMOR_PEER_IDENTITY=server
export QARMOR_SIGNING_KEY=/absolute/private/qarmor-demo-keys/agent.private.pem
export QARMOR_TRUSTED_PEER_KEY=/absolute/private/qarmor-demo-keys/server.public.pem
python -m Agent.ids_agent
```

The agent additionally needs its existing model/scaler/encoder/feature artifacts and packet-capture permissions. Ordinary tests use local socket pairs and do not load the live agent or capture packets. The current server configuration pins one peer identity. Supporting many device identities requires a provisioned trust registry, admission controls and concurrency work; this change does not claim those features.

The three-message handshake binds protocol context, initiator/responder identities, fresh challenges and both ephemeral X25519 keys in a canonical JSON transcript. The responder and initiator sign role-separated transcripts with Ed25519. Each verifies its pinned expected peer before accepting a derived session. HKDF-SHA256 binds the session key to the transcript hash; direction-specific keys bind session and sender/receiver. Fresh challenges and one-use state reject old transcript responses/finishes. The initial offer is unauthenticated until the signed finish arrives; resource-exhaustion/rate-limiting is outside this prototype.

AES-GCM AAD includes canonical protocol, session, sender, receiver and counter. Nonces are the 96-bit counter encoding under a direction/session-derived key. Counters are allocated atomically before encryption (crashes may leave gaps). Decryption verifies the tag before advancing trusted receiver state; duplicate and lower counters are rejected. Delivery is strictly ordered: out-of-order old packets are intentionally rejected. A 32-bit message budget forces session rotation before counter exhaustion; applications should rotate earlier for volume/security policy.

Default live transport creates a fresh ephemeral handshake on every connection and **never resumes a memory-backed session after restart**. Repeated handshake channel access shares counter state. Persistent/resumable direct SecureChannel use requires the same protected SQLite CounterStore path for every writer of that key/session and durable receiver state. Do not duplicate memory stores, fork active sessions, reuse a key/session with reset counters, delete state or restore old counter backups. SQLite commits are atomic, but this is not rollback-resistant storage against an attacker with filesystem access. Use restrictive file permissions and a fresh authenticated session after state loss. Private keys need access controls, rotation and secure provisioning. No production-grade or post-quantum security claim is made.

`crypto/pqc_crypto.py:HybridCrypto` is actually classical X25519/Ed25519; its unsigned legacy exchange remains for compatibility only. `ClassicalCrypto` retains its legacy AES/RSA API. Neither legacy API gains replay protection by implication; use SecureChannel and authenticated_handshake.

The nonce and authentication-tag requirements follow the [cryptography AES-GCM documentation](https://cryptography.io/en/stable/hazmat/primitives/aead/); [X25519 documentation](https://cryptography.io/en/stable/hazmat/primitives/asymmetric/x25519/) specifies deriving keys with a KDF.

## Benchmarks

```sh
python -m tools.benchmark_security --iterations 1000 --output results/benchmarks
python -m tools.benchmark_security --iterations 500 --store /absolute/private/counters.sqlite --output results/benchmarks_durable
```

Outputs include raw repeated timings, JSON, CSV and telemetry latency plot. AES envelopes use 64/1024/16384-byte payloads, a 25-message warmup, encryption/decryption/roundtrip median and p95, throughput and wire overhead. The telemetry workload matches the existing agent metrics JSON; plain JSON processing is compared with authenticated encryption, replay validation, decryption and parsing. Durable mode includes SQLite FULL commits. Handshake timing includes ephemeral generation, signatures, verification and HKDF, excluding provisioned identity generation. CPU/RSS and dependency/platform information are saved. These are measured local processing benchmarks, not live network/IDS inference/packet-to-alert latency or Metal model throughput.

The current benchmark also times X25519 shared-secret exchange and HKDF-SHA256 separately, and compares AES-GCM operations with no AAD against counter-bound AAD. The AEAD-only rows exclude AAD construction; the end-to-end telemetry rows include counter reservation, canonical envelope handling, authenticated decryption/replay validation and JSON parsing. Wire overhead compares a nonce-plus-tag ciphertext with the full counter/context envelope. Keep CPU-only cryptographic latency distinct from network latency and IDS inference.


## Historical exploratory execution record — 12 October 2026

The raw class-folder archive was successfully sampled with `tools/prepare_archive_splits.py`; see `data/CICIoT23_archive_sample_v1/dataset_manifest.json` for the archive fingerprint, exact rows, class counts and split source assignments. The output contains 99,514 rows across 34 classes. Seventeen low-source-count classes are split by rows, so capture overlap is possible for them. Treat all results from this sample as exploratory.

Completed outputs:

- `results/archive_sample_v1_pilot/`: one seed, two epochs, 1% training fraction. The reweighting feedback path actuated in epoch 1 and the next epoch used the resulting weights. Test accuracy 0.0302 / macro-F1 0.0020; wiring evidence only.
- `results/archive_sample_v1_calibration_measured/`: partial higher-budget paired calibration, seed 42 only; two scales completed before runtime interruption. Candidate config is provisional and must not be described as calibrated.
- `results/archive_sample_v1_exploratory_3seed/`: 3 seeds × baseline/proposed completed at two epochs and 1% training fraction. Proposed selected NORMAL throughout, making the methods identical. Mean accuracy 0.05085 (SD 0.01860), macro-F1 0.01063 (SD 0.00818); all seven candidate-class F1 values were zero.
- `results/benchmarks_m1_current/` and `results/benchmarks_m1_current_durable/`: current in-memory and durable SQLite security timings on the M1 Air. The full test suite passed 27/27.

Do not use the sample run as the final three-seed study: the full training budget was intentionally reduced, the sample is not the original split, and several classes can have row-level split overlap. A full-scale experiment still requires a reviewed resource budget and the intended dataset split.

Live packet-to-alert operation is not ready to run from this checkout. `Agent/ids_agent.py` references missing packet sniffer/feature extractor/custom layer modules, and no `Model_Keras/` model/scaler/encoder/feature-order artifacts are present. The authenticated transport and replay protections have independent tests; restore compatible live inference components before claiming end-to-end IDS operation.


## Stronger local study protocol (v2)

`tools/run_remaining_study.py` records and executes a fixed protocol in `results/remaining_study_v2/`: calibration seeds 101–103, five full warmup epochs and paired probes at maximum first-step scales 1.15 and 1.5; a candidate fitted at the preselected 1.15 scale; then baseline/proposed seeds 42–44 with ten full epochs each. All runs use 34,000 balanced training rows (1,000 per class), batch size 256 and the unchanged original architecture on CPU. The second validation half supplies ablation controller feedback and checkpoint selection. The test split has prior exploratory usage, which is disclosed. No new test-driven tuning is allowed.

The study saves its source snapshot, dependency versions, protocol, dataset fingerprints, full epoch histories, model checkpoints, calibration audit, runtime logs and all-class metrics. `tools/report_remaining_study.py` only produces its final report after all stages complete. It also reports a declared resource/latency-cost sensitivity grid on baseline validation states, without changing the frozen policy or retraining to select a preferred result.


## Completed v2 study and reproduction

The stronger study and its predeclared original-parameter reference arm are complete: nine ten-epoch runs and six paired calibration probes. See [REPORT.md](results/remaining_study_v2/REPORT.md) for the authoritative results. The reference arm was declared before any new main-study test result and reuses the matched baseline runs. It is an uncalibrated mechanism reference, not a selected replacement for the empirically fitted policy.

Macro-F1 mean ± sample SD: baseline 0.53952 ± 0.00765, calibrated 0.53952 ± 0.00765, original-parameter reference 0.53864 ± 0.01619. Calibrated recommendations were NORMAL throughout; reference seeds applied 8, 8 and 7 weight updates followed by training. SQL injection and XSS gains came with losses elsewhere. There is no demonstrated overall macro-F1 improvement.

```sh
# Resume the declared study; completed stages are skipped.
./.venv/bin/python -m tools.run_remaining_study
# Inspect saved-result invariants and regenerate the completed report.
./.venv/bin/python -m tools.check_study_results
./.venv/bin/python -m tools.report_remaining_study
```

Use a separate output directory and a new explicit protocol for changed experiments; do not overwrite the frozen evidence. The saved 29-test result is `results/remaining_study_v2/unit_tests.txt`. `integrity_checks.json` records the completed checks across nine runs. The source snapshot preserves historical loaded runner versions alongside the final reporting/runner tools. Earlier exploratory outputs above remain historical evidence only.
