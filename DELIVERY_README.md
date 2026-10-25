# Q-ARMOR project delivery — 23 October 2026

Latest recovery fix: the validation replay checker now accepts configurable CSV paths and uses the surviving deployment bundle without requiring the deleted training pickle. Independent scaler parity is explicitly unexecuted unless a trusted reference is supplied. Follow-up verification passed 38 tests, source/bundle checks, and saved-study integrity (`results/verification_20261023_continue`). The exact archive and all three frozen CSV splits were restored on 23 October. Both fresh 128-record validation replays passed, with 121 predicted alerts delivered and seven benign predictions suppressed; see `results/restored_validation_replay_20261023` and `results/restored_validation_tcp_20261023`. No new training or accuracy claim is made.

This is the recovered, runnable research implementation. Start with `README.md` for setup and `DEPLOYMENT_RUNBOOK.md` for the secure inference demo.

Latest full runnable verification: `results/verification_20261023/verification.json`; all seven checks passed, including 37 unit tests. The raw archive `data/raw/CSV.zip` and exact train/validation/test CSVs are now included. Historical research results are preserved, and new synthetic checks are explicitly labelled.

## Start and verify

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m tools.verify_project --output results/verification_new
```

The verifier checks imports, source/notebook consistency, unit tests, the frozen model fingerprints and the nine saved training runs. It runs a new original-architecture training/Stackelberg feedback smoke test and a separate-process authenticated TCP demonstration using the real deployment model. It also measures crypto/telemetry processing, including durable counter state. New training smoke and network-demo inputs are explicitly synthetic and carry that provenance into saved alerts. No data-quality metrics are claimed for them.

`results/verification_20261021/verification.json` records the full runnable pipeline; `results/final_checks_20261021/verification.json` records the final 37-test pass after the recovery enhancements. See `PROJECT_STATUS.md` for current status. `requirements-verified.txt` records that environment; GitHub Actions will run the same verifier after this local workflow is pushed. The presence of the workflow alone is not a claim that cloud CI has run.

## What is included

- IDS architecture, train-only preprocessing, hybrid train balancing, epoch-by-epoch feedback controller, three-seed ablation/calibration runners and main notebook.
- Saved bounded-sample study: all three seeds per method, histories, confusion matrices, per-class metrics, plots, frozen protocols, calibration effects and saved results.
- Seed-42 inference checkpoint, preprocessing schema and original integrity fingerprints.
- Authenticated X25519/Ed25519 handshake, HKDF, AES-GCM/AAD, replay counters and persistence/concurrency tests.
- Agent/server feature-record inference and encrypted alert delivery, acknowledgement, reproducible demos and benchmarks.
- Checksum-gated dataset restoration, a fresh verification command, ZIP packager, tests and setup/run documentation.
- Surviving local review ZIPs and early integration materials under `RECOVERED_LOCAL_MATERIALS`; recovery-era package fingerprints are historical, not the current delivery manifest.

## Remaining external blockers

The original 1,430,268,604-byte `CSV.zip` has been recovered, and the exact 99,514-row development sample regenerated. The raw archive and all three CSVs match the frozen SHA-256 fingerprints. Repeated training checkpoints, the training preprocessing pickle, and later local-only XSS intervention outputs remain unavailable; they were not committed. The included deployment model, preprocessing JSON and historical research results survived.

The sample can be restored only from the matching original archive or the exact split CSVs:

```sh
python -m tools.restore_data --archive /path/to/CSV.zip
# Only for a Kaggle ZIP whose class-folder prefix is data/:
python -m tools.restore_data --mirror-archive /path/to/kaggle-download.zip
# or
python -m tools.restore_data --splits /path/to/previous/sample
```

The tool rejects fingerprints that differ from the saved study. It never silently substitutes another CICIoT2023 download or writes synthetic files into research splits. Explicit `--mirror-archive` mode supports a repack with `data/<class>/...` paths; it preserves the original canonical sampling names and rejects the entire restoration unless all three sampled CSV fingerprints match. A checked Kaggle Backdoor file matched the saved 39-column schema and 3,218-row source count; this limited check does not establish full mirror identity.

The exact source archive SHA-256 is `e211e878d2f39226ea3a854683f91ae6175b628807ee96eeecb7a04a36a28dbd`. Reconstructed sample CSVs must also match their saved hashes before installation. Original-archive restoration has now executed successfully, with every output CSV fingerprint matching. Evidence is saved in `results/restored_verification_20261023/restoration.log` and `data/CICIoT23_archive_sample_v1/restoration_manifest.json`.

The automated publisher-portal request during verification returned a registration/server-error page; a manually registered browser session may still allow download. See the [official dataset description](https://www.unb.ca/cic/datasets/iotdataset-2023.html) and [download portal](https://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/). No registration identity or terms acceptance was submitted automatically.

Raw-packet feature extraction parity and live capture remain unverified without the original extractor and a representative PCAP/CSV fixture. The current agent accepts compatible pre-extracted feature records; arbitrary per-packet approximations are not presented as the 25 training features. Optional alternative datasets and post-quantum KEM integration are not implemented.

## Interpretation of the research results

The preserved three-seed study is on a bounded archive sample, not the full original benchmark or an untouched external test set. Mean test macro-F1 was 0.53952 for both baseline and the calibrated policy, which selected NORMAL. The original-assumption reweighting reference averaged 0.53864 and did not show overall macro-F1 improvement. Some class scores improved while others worsened. Do not claim an overall improvement, fresh full-dataset evaluation, production certification or post-quantum security.

## Portable archive

```sh
python -m tools.package_project --output ../Q-ARMOR_Project_Delivery.zip
```

Every available source/result/data file is included, even if untracked by Git, along with repository history. The archive contains `DELIVERY_MANIFEST.json` with file hashes, and a separate archive `.sha256` file is created. Every ZIP member is read back and verified. The packager omits local environments/caches, private identity keys, `.env` credentials and replay-state databases; exclusions are listed. Provision fresh local demonstration identities with the included helper. Never reuse discarded counter state with an old active key/session.

## Final three-seed intervention result

See `results/final_intervention_3seed_20261023/REPORT.md`. The existing Stackelberg loop selected and applied reweighting to Recon-PingSweep in seeds 42, 43 and 44. Its F1 improved in one of three runs, so consistent recovery was not demonstrated. The macro-F1 increase is a next-epoch association and cannot be attributed to reweighting alone.
