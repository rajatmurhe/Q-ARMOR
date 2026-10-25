# Three-seed Stackelberg intervention result

Status: **completed_intervention_analysis**

Existing Stackelberg IDS training-loop histories from original_assumptions_reference, seeds 42, 43, 44. For each seed, selected the first applied CLASS_REWEIGHTING decision targeting Recon-PingSweep and compared validation metrics immediately before the decision with the following epoch after the same model trained using the updated weights.

| Seed | F1 before | Stackelberg decision | Target / applied action | Scale | Normalized weight used next epoch | F1 after | Macro-F1 before → after |
|---:|---:|---|---|---:|---:|---:|---:|
| 42 | 0.1913 | CLASS_REWEIGHTING | Recon-PingSweep / CLASS_REWEIGHTING | 1.1213 | 1.1054 | 0.2072 | 0.4803 → 0.4949 |
| 43 | 0.1875 | CLASS_REWEIGHTING | Recon-PingSweep / CLASS_REWEIGHTING | 1.1219 | 1.1012 | 0.1865 | 0.4935 → 0.4961 |
| 44 | 0.2069 | CLASS_REWEIGHTING | Recon-PingSweep / CLASS_REWEIGHTING | 1.1190 | 1.1072 | 0.1942 | 0.4977 → 0.5272 |

Across seeds, target-class F1 was 0.1952 ± 0.0103 before and 0.1960 ± 0.0105 after (mean change +0.0008). Macro-F1 was 0.4905 ± 0.0090 before and 0.5061 ± 0.0183 after (mean change +0.0156). Values are mean ± sample SD.

## Finding

The controller detected the same weak class and applied reweighting in all three seeds, increasing its normalized training weight in the next epoch. **The weak-class recovery objective was not demonstrated:** Recon-PingSweep F1 improved in only one of three seeds. Macro-F1 increased after the following epoch in all runs, but ordinary training and other epoch effects also contribute; this sequential comparison does not identify reweighting as the cause. No seed was excluded based on its result.

This is validation-set intervention evidence from the already completed `original_assumptions_reference` three-seed run, not a newly trained experiment or held-out test result. The calibrated main-study policy selected NORMAL and remains the primary calibrated outcome; this intervention uses the explicitly labelled original-assumption reference game configuration.

Source traces: `results/remaining_study_v2/reference_ablation/{42,43,44}/proposed/history.json`. Row-level extraction: `three_seed_results.csv`; aggregate: `summary.json`.
