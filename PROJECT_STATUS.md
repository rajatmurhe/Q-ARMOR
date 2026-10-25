## Final three-seed intervention requested by mentor — 23 October 2026

Completed the requested trace analysis for the weak `Recon-PingSweep` class using the existing original-assumption Stackelberg reference run (seeds 42, 43, 44). The controller selected and applied `CLASS_REWEIGHTING` in all three. Mean class F1 changed from 0.1952 ± 0.0103 to 0.1960 ± 0.0105; it improved in only one seed. Mean validation macro-F1 changed from 0.4905 ± 0.0090 to 0.5061 ± 0.0183. Because ordinary training also occurred in the following epoch, the macro-F1 change cannot be attributed to reweighting. The requested consistent recovery was **not demonstrated**. Full per-seed values and weight scales: `results/final_intervention_3seed_20261023/REPORT.md`.

# Q-ARMOR current project status — 23 October 2026

Latest restoration verification: `results/restored_verification_20261023/verification.json` passed 38 tests, source/model integrity, saved-study checks and all dataset hashes. The raw archive and exact split CSVs are included in this delivery.

Latest recovery fix: the validation replay checker now accepts configurable CSV paths and uses the surviving deployment bundle without requiring the deleted training pickle. Independent scaler parity is explicitly unexecuted unless a trusted reference is supplied. Follow-up verification passed 38 tests, source/bundle checks, and saved-study integrity (`results/verification_20261023_continue`). The exact archive and all three frozen CSV splits were restored on 23 October. Both fresh 128-record validation replays passed, with 121 predicted alerts delivered and seven benign predictions suppressed; see `results/restored_validation_replay_20261023` and `results/restored_validation_tcp_20261023`. No new training or accuracy claim is made.

Active working copy: `/Users/rajatmurhe/Projects/Q-ARMOR`. The previous Desktop copy suffered iCloud dataless-file stalls. The environment lives in `~/.venvs/qarmor` and the project `.venv` links to it.

**Fresh verification on 23 October passed all seven runnable checks, including 37 unit tests.** Evidence: `results/verification_20261023/verification.json`. The original-architecture feedback smoke, authenticated TCP deployment with 128 explicitly synthetic records, and memory/durable crypto benchmarks all passed. No new CICIoT2023 training or quality results were produced. The dataset was subsequently restored and verified; see the restoration evidence below.

**Earlier core software verification passed: 37 tests.** Full pipeline evidence is in `results/verification_20261021`; final source/unit/data-restoration checks are in `results/final_checks_20261021`. These are new local checks, distinct from the preserved 12 October research measurements.

| Component | Verified status |
|---|---|
| Source and notebook | All active cells/modules parse; notebook Cell 27 and exported adapter now agree. |
| Saved nine-run study | Frozen configuration, split isolation, next-epoch weights and all saved aggregate/per-class metrics checked against confusion matrices. |
| Training feedback | Three real Keras epochs with original architecture; synthetic fixture exercised actual solver decisions and next-epoch reweighting. |
| Inference and network | Real seed-42 checkpoint; restored validation replay delivered 121 predicted alerts from 128 original records, seven benign suppressed; socketpair and separate-process TCP passed. No new accuracy claim. |
| Crypto/replay | Signed ephemeral handshake, AES-GCM/AAD, replay rejection and SQLite persistence/concurrency tests passed. |
| Benchmarks | Fresh memory and durable-state workloads completed; median/p95/raw samples saved. |
| Data recovery | Hash-gated original ZIP import and explicit repacked-mirror mode implemented and fixture-tested; real archive and all three split hashes verified; exact sample restored. |
| Delivery | Portable packager includes available untracked results/data, excludes environments/private keys/state and verifies every ZIP member. |
| CI | Workflow added locally; no cloud run or push is claimed. |

## Preserved research conclusion

The bounded-sample study shows baseline/calibrated macro-F1 0.53952 ± 0.00765 and original-assumption reference 0.53864 ± 0.01619. The calibrated policy selected NORMAL. There is no demonstrated overall macro-F1 improvement. These are historical saved results; no new real-data study has run after the deletion.

## Dataset restoration and verification

Keep it zipped. From this project folder:

```sh
source .venv/bin/activate
python -m tools.restore_data --archive data/raw/CSV.zip
# For an explicitly repacked Kaggle data/<class>/ archive only:
python -m tools.restore_data --mirror-archive /path/to/kaggle-download.zip
```

The official source archive is 1,430,268,604 bytes, SHA-256 `e211e878d2f39226ea3a854683f91ae6175b628807ee96eeecb7a04a36a28dbd`. All three restored split files must also match their frozen hashes. There is no automatic substitution of another dataset. Large source archives are read in chunks without extracting all raw CSVs.

## Work still requiring external resources

The original predefined Kaggle split remains unavailable; the completed study used the recorded bounded archive sample. Restoring the sample does not establish full-dataset validity. Raw-packet feature parity/live capture still needs the publisher extractor and a representative PCAP/CSV fixture. Optional alternate-dataset and PQC extensions remain unimplemented. Deleted repeated checkpoints and later XSS outputs were not on GitHub.

The original failed verification (adapter mismatch) is retained under `results/recovery_failures`. Earlier evidence is preserved, not overwritten or relabelled as a current successful run.
