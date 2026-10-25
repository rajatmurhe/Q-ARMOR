# Q-ARMOR inference deployment

The frozen seed-42 baseline bundle is available at `deployment/archive_sample_seed42/`. It contains the validation-selected checkpoint, feature order, training medians, scaler parameters, classes and SHA-256 fingerprints. Seed and method were fixed before deployment replay; no best-test model selection was performed.

The agent consumes compatible CICIoT2023 feature CSV or JSONL records, runs the original CNN/LSTM/attention checkpoint, suppresses `BenignTraffic` predictions and sends alerts over the existing signed X25519/Ed25519 session. The receiver persists alerts, records telemetry and acknowledges the received alert count. Imports no longer load a model, connect to a server or begin capture automatically.

## Verified evidence

- `results/deployment_v1/verification.json`: real checkpoint, 128 validation records, preprocessing exactly matches the original scaler transform; 121 predicted attack records delivered, seven benign predictions suppressed, confidences preserved and acknowledgement verified.
- `results/deployment_tcp_v1/verification.json`: separate agent and server processes over authenticated loopback TCP, 128 records and 121 persisted alerts; both processes exit successfully.
- `results/deployment_v1/unit_tests.txt`: 19 deployment/security tests passed, including schema rejection, imputation/order, benign suppression, transport, replay and handshake coverage.

These checks establish integration of inference and authenticated delivery. The alerts are model predictions from recorded validation features, not confirmation of 121 real attacks. They add no new accuracy or generalization claim to the research study.

## Run the complete local demo

Create and activate the Python 3.11 environment described in README.md, then run from the repository root. The demo requires the matching validation feature CSV, which is kept outside Git.

```sh
python -m tools.run_deployment_demo --output results/deployment_demo_new
```

The demo generates temporary identities, starts a server on an automatically allocated loopback port, runs the real agent on 128 validation records, records logs and removes the temporary private keys. Use a new output directory for each execution.

## Run an agent and server separately

Provision identities and verify pinned public keys through your trusted administration process. The helper below creates local demonstration identities only:

```sh
python -m tools.provision_demo_identities --directory .local-identities
```

Server terminal:

```sh
export QARMOR_IDENTITY=server QARMOR_PEER_IDENTITY=agent
export QARMOR_SIGNING_KEY=.local-identities/server.private.pem
export QARMOR_TRUSTED_PEER_KEY=.local-identities/agent.public.pem
python -m Server.server --output results/receiver_new
```

Agent terminal:

```sh
export QARMOR_IDENTITY=agent QARMOR_PEER_IDENTITY=server
export QARMOR_SIGNING_KEY=.local-identities/agent.private.pem
export QARMOR_TRUSTED_PEER_KEY=.local-identities/server.public.pem
python -m Agent.ids_agent --features data/CICIoT23_archive_sample_v1/validation/validation.csv --limit 128
```

For a streaming collector, pass `--features - --format jsonl --batch-size 1` and pipe complete feature records into standard input. Every record must contain the bundle's 25 named features with the same meaning and units used in training. Extra label and metadata fields do not enter inference. Missing required fields and malformed numeric values fail; nonfinite values use the frozen training medians. Source/destination IPs are included only when provided by the collector. `--confidence-threshold` controls alert emission without changing model predictions; its default is zero and no threshold optimization is claimed.

The server binds to loopback by default; `--host` and `--port` configure a provisioned deployment. One peer is pinned and one connection is handled at a time. Feature inference is batched, so use a small batch for low-volume streaming. Delivery failures propagate to the caller; the agent does not silently claim success or retry messages under reused counter state.

## Raw packet and full-dataset work still required

The [dataset publisher](https://www.unb.ca/cic/datasets/iotdataset-2023.html) states that DPKT extracted the dataset features and that source code belongs to its supplementary material. The publisher's download endpoint returned a form with a server error during this work. A verified raw-packet extractor and representative PCAP parity fixture are not available in this workspace. The implemented feature adapter accepts compatible extracted records; it cannot infer the dataset's aggregate features from an arbitrary single packet.

Raw packet-to-feature parity, actual interface capture, remote-network delivery, sustained load, reconnection behavior and full-dataset evaluation remain unverified. The bounded research model's reported macro-F1 is about 0.54. These limits remain relevant when choosing deployment scope.

## Recovery-safe demonstration — 21 October 2026

The GitHub snapshot lacked the original validation CSV and per-seed preprocessing pickle. On 23 October, the archive and exact validation split were restored. Fresh socketpair and separate-process TCP replay results are in `results/restored_validation_replay_20261023` and `results/restored_validation_tcp_20261023`: 128 original records, 121 predicted alerts delivered, seven benign predictions suppressed. The preprocessing pickle remains unavailable, so independent scaler parity was not rerun. Historical results are preserved.

Use this explicit integration-only demonstration with the real included checkpoint:

```sh
python -m tools.run_deployment_demo --synthetic --output results/demo_new
```

Synthetic numeric fixtures need not represent physically possible packet windows. Their CSV rows carry `_qarmor_source=synthetic_smoke`; emitted alerts preserve `source=synthetic_smoke`. The verifier checks this provenance after delivery. It does not calculate a dataset F1 or claim live attack detection. Fresh temporary identities are provisioned for each run and removed at exit.

For recovered or independently provided compatible features:

```sh
python -m tools.run_deployment_demo --features /absolute/path/to/features.csv --records 128 --output results/real_feature_replay_new
```

Both modes exercise separate agent/server processes, the signed handshake, real model inference, encrypted alerts, receiver persistence and acknowledgement. External feature rows are labelled `provided_feature_record`; their availability alone does not prove packet extraction matches CICIoT2023. Native capture still requires the original extractor and a PCAP-to-CSV parity fixture, not arbitrary values generated per packet. See the [publisher's feature extraction description](https://www.unb.ca/cic/datasets/iotdataset-2023.html).

## Replay checker after dataset restoration

```sh
python -m tools.check_deployment --features data/CICIoT23_archive_sample_v1/validation/validation.csv --output results/validation_replay_new
```

This checker uses the included fingerprinted deployment bundle and records the supplied CSV hash. Missing, empty, or explicitly synthetic input fails before loading the model or creating the result directory. Check the CSV hash against the frozen study manifest before describing an external file as the original validation split.

The deleted training `preprocessing.pkl` is no longer mandatory. To repeat the independent scaler comparison, explicitly supply a trusted local artifact with `--reference-preprocessing /path/to/preprocessing.pkl`. Without it, `schema_transform_bit_identical` is null and `reference_preprocessing_status` is `not_run_no_reference_artifact`; the historical parity result is not claimed as a fresh check.
