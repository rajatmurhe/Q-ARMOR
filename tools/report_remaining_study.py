"""Build the study report exclusively from saved, completed experiment outputs."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from game_theory.ids_training import WEAK_CLASSES, save_json


def mean_sd(values):
    return f'{np.mean(values):.5f} ± {np.std(values, ddof=1):.5f}'


def main():
    root = Path(__file__).resolve().parents[1]
    out = root/'results/remaining_study_v2'
    status = json.loads((out/'status.json').read_text())
    if status['status'] != 'complete':
        raise RuntimeError('Study is not complete; report generation refused')
    reference_declared = (out/'reference_protocol.json').exists()
    if reference_declared:
        reference_status = out/'reference_ablation/status.json'
        if not reference_status.exists() or json.loads(reference_status.read_text())['status'] != 'complete':
            raise RuntimeError('Declared reference arm is not complete')
    protocol = json.loads((out/'protocol.json').read_text())
    cal = json.loads((out/'calibration/paired_probes.json').read_text())
    config = json.loads((out/'frozen_game_config.json').read_text())
    runs = {(seed, method): json.loads((out/f'ablation/{seed}/{method}/test.json').read_text())
            for seed in protocol['ablation_seeds'] for method in ['baseline', 'proposed']}
    histories = {(seed, method): json.loads((out/f'ablation/{seed}/{method}/history.json').read_text())
                 for seed, method in runs}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, seed in zip(axes, protocol['ablation_seeds']):
        curves = [('Baseline', histories[seed, 'baseline'], '-'),
                  ('Calibrated policy', histories[seed, 'proposed'], '--')]
        if reference_declared:
            curves.append(('Original policy', json.loads(
                (out/f'reference_ablation/{seed}/proposed/history.json').read_text()), ':'))
        for label, h, style in curves:
            ax.plot([r['epoch'] for r in h], [r['validation']['macro_f1'] for r in h],
                    marker='o', label=label, linestyle=style)
        ax.set(title=f'Seed {seed}', xlabel='Epoch', ylabel='Validation macro-F1', ylim=(0,1))
        ax.legend()
    fig.tight_layout()
    fig.savefig(out/'learning_curves.png', dpi=150)
    plt.close(fig)
    keys = ['accuracy', 'macro_f1', 'weighted_f1', 'macro_precision', 'macro_recall']
    summary = {'metrics': {}, 'weak_classes': {}, 'seeds': {}, 'calibration_config': config,
               'protocol': protocol}
    lines = ['# Q-ARMOR: completed local archive-sample study', '',
             'This report is generated from the recorded calibration and completed training runs. '
             'It does not describe full-dataset or live packet-to-alert performance.', '',
             '## Fixed experiment protocol', '',
             'The 34-class sample contains 66,342 training rows and 16,586 rows in each holdout split. '
             'Train-only balancing produces 34,000 rows (1,000 per class); every row is used each epoch. '
             'Consequently, calculated ordinary balanced class weights are uniform before adaptive scaling. '
             'The unchanged CNN/LSTM/attention model trains for 10 epochs per method and seed. '
             'Calibration uses seeds 101–103, five warmup epochs and paired one-epoch probes. '
             'Baseline/proposed use fresh initializations for seeds 42–44 with identical per-seed preprocessing, '
             'sample orders, budgets and initialization. Adam uses learning rate 0.0005 and clipnorm 1.', '',
             'The validation split has two fixed stratified halves (seed 2026). One fits calibration; '
             'the other audits calibration and supplies ablation feedback/checkpoint selection. '
             'The game configuration and training budget were frozen before the new test evaluations. '
             'The test split was previously used for exploratory runs, so it is not a never-inspected external holdout.', '',
             '## Three-seed results', '', '| Metric | Baseline mean ± SD | Proposed mean ± SD | Paired difference mean ± SD |',
             '|---|---:|---:|---:|']
    for key in keys:
        a = np.array([runs[s, 'baseline']['test'][key] for s in protocol['ablation_seeds']])
        b = np.array([runs[s, 'proposed']['test'][key] for s in protocol['ablation_seeds']])
        lines.append(f'| {key} | {mean_sd(a)} | {mean_sd(b)} | {mean_sd(b-a)} |')
        summary['metrics'][key] = {n: {'mean': float(v.mean()), 'sample_sd': float(v.std(ddof=1))}
                                  for n, v in [('baseline', a), ('proposed', b), ('paired_delta', b-a)]}
    lines += ['', 'SD is sample standard deviation over three seeds. These descriptive estimates have substantial uncertainty; '
              'a positive mean difference alone is not evidence of statistical significance.', '',
              '## Seven preselected classes', '', '| Class | Baseline F1 mean ± SD | Proposed F1 mean ± SD |', '|---|---:|---:|']
    per_class = []
    for name in next(iter(runs.values()))['test']['per_class']:
        a = np.array([runs[s, 'baseline']['test']['per_class'][name]['f1'] for s in protocol['ablation_seeds']])
        b = np.array([runs[s, 'proposed']['test']['per_class'][name]['f1'] for s in protocol['ablation_seeds']])
        per_class.append({'class': name, 'baseline_mean': a.mean(), 'baseline_sd': a.std(ddof=1),
                          'proposed_mean': b.mean(), 'proposed_sd': b.std(ddof=1), 'paired_delta_mean': (b-a).mean()})
        if name in WEAK_CLASSES:
            lines.append(f'| {name} | {mean_sd(a)} | {mean_sd(b)} |')
            summary['weak_classes'][name] = per_class[-1]
    pd.DataFrame(per_class).to_csv(out/'all_class_f1.csv', index=False)
    lines += ['', 'These seven classes were specified by the existing project before this study; they were not selected by test performance.', '',
              '## Actions and learning curves', '', '| Seed | Method | Best validation epoch | Best validation macro-F1 | Final validation macro-F1 | Applied reweighting decisions |',
              '|---|---|---:|---:|---:|---:|']
    for (seed, method), history in histories.items():
        best = max(history, key=lambda r:r['validation']['macro_f1'])
        decisions = [r['decision'] for r in history if r['decision']]
        applied = sum(d['applied_action'] == 'CLASS_REWEIGHTING' for d in decisions)
        # A decision after the final epoch has no subsequent fit in this run.
        trained = sum(r['decision'] is not None and r['decision']['applied_action'] == 'CLASS_REWEIGHTING' for r in history[:-1])
        counts = dict(Counter(d['recommended_defense'] for d in decisions))
        summary['seeds'][f'{seed}/{method}'] = {'recommendations': counts, 'applied_decisions': applied,
             'decisions_with_subsequent_training': trained, 'best_epoch': best['epoch'],
             'best_validation_macro_f1': best['validation']['macro_f1'],
             'final_validation_macro_f1': history[-1]['validation']['macro_f1'],
             'training_and_validation_seconds': sum(r['seconds'] for r in history)}
        lines.append(f'| {seed} | {method} | {best["epoch"]} | {best["validation"]["macro_f1"]:.5f} | {history[-1]["validation"]["macro_f1"]:.5f} | {applied} ({trained} followed by training) |')
    lines += ['', '### Strategy probability diagnostics', '',
              '| Seed | Largest defender probability change per epoch | Largest attacker probability change per epoch | Normalized |',
              '|---|---:|---:|---|']
    for seed in protocol['ablation_seeds']:
        diag = json.loads((out/f'ablation/{seed}/proposed/probability_diagnostics.json').read_text())
        summary['seeds'][f'{seed}/proposed']['probability_diagnostics'] = diag
        lines.append(f'| {seed} | {diag["p_def"]["max_epoch_change"]:.6f} | '
                     f'{diag["p_att"]["max_epoch_change"]:.6f} | '
                     f'{diag["p_def"]["normalized"] and diag["p_att"]["normalized"]} |')
    lines += ['', '![Validation learning curves](learning_curves.png)', '',
              'Controller recommendations, actual weight changes, deterioration rollbacks and both probability/utility traces '
              'are saved per epoch. The pure Stackelberg recommendation drives the actuator; co-evolution probabilities are '
              'diagnostics and do not select a sampled action. Fixed or nearly fixed probabilities must be described as such.', '',
              '## Validation-only calibration', '',
              'The 1.15 maximum first-step scale was selected in the protocol before running probes. '
              'The actual scale is 1 + 0.15 × measured target weakness, matching the controller first update. '
              'The 1.5 probe is sensitivity evidence and does not select the candidate. Each treatment/control pair '
              'starts from identical warmup weights and optimizer state, and uses identical samples and random seeds. '
              'The weakest non-benign class is selected on the calibration half only.', '',
              '| Seed | Maximum scale | Target | Calibration target ΔF1 | Audit target ΔF1 | Calibration macro ΔF1 | Audit macro ΔF1 |',
              '|---|---:|---|---:|---:|---:|---:|']
    for r in cal:
        lines.append(f'| {r["seed"]} | {r["requested_scale"]} | {r["target"]} | {r["target_f1_delta"]:+.5f} | {r["audit_target_f1_delta"]:+.5f} | {r["macro_f1_delta"]:+.5f} | {r["audit_macro_f1_delta"]:+.5f} |')
    lines += ['', 'Both NORMAL and CLASS_REWEIGHTING effectiveness estimate the observed next-epoch reduction '
              'in target weakness, relative to the warmup state. Signed observations are averaged before clipping '
              'to the supported [0, 0.95] range. A constant response is fitted because three seeds cannot reliably '
              'identify a weakness-dependent curve. False-alarm costs use measured benign false-alarm rates. '
              'Resource/latency costs and utility weights remain declared prototype assumptions. These estimates '
              'do not establish effectiveness for every class, training state, accumulated scale or other dataset.', '',
              'Frozen configuration:', '', '```json', json.dumps(config['config'], indent=2), '```', '',
              '## Scope and limitations', '',
              '- This is a bounded archive-sample study. Seventeen classes use row-level splits because only one or two source CSVs exist; capture overlap may remain.',
              '- The raw-feature overlap audit found zero exact training-row matches in validation or test. This does not rule out correlated or near-duplicate flows.',
              '- A fixed ten-epoch budget does not establish convergence. Best/final validation scores and curves above expose remaining learning trends.',
              '- A policy that chooses NORMAL is a null intervention outcome under the measured effects and declared costs; it does not prove class reweighting can never help.',
              '- The authenticated transport and crypto measurements are separate evidence. Live packet extraction and deployment artifacts remain outside this completed offline study.',
              '', 'Detailed artifacts: [ablation](ablation/), [calibration](calibration/), [protocol](protocol.json), '
              '[all-class table](all_class_f1.csv), [test log](unit_tests.txt).', '']
    # A declared sensitivity analysis on validation states only; never chooses a new config.
    from game_theory.training_bridge import StackelbergIDSController
    import copy
    sensitivity = []
    for multiplier in [0., .1, .5, 1., 2.]:
        parameters = copy.deepcopy(config['config'])
        parameters['payoff'] = {'resource_weight': .3*multiplier,
                               'latency_weight': .3*multiplier,
                               'false_positive_weight': .5, 'security_weight': 1.}
        actions = Counter()
        for seed in protocol['ablation_seeds']:
            for row in histories[seed, 'baseline']:
                f1 = {n: v['f1'] for n, v in row['validation']['per_class'].items()}
                controller = StackelbergIDSController(list(f1), game_parameters=parameters,
                                   available_defenses=['NORMAL', 'CLASS_REWEIGHTING'])
                attacks = {n: f1[n] for n in controller.attack_classes}
                decision = controller.game.solve(attacks, defense_actions=controller.DEFENDER_ACTIONS)
                actions[decision['recommended_defense']] += 1
        sensitivity.append({'resource_latency_weight_multiplier': multiplier, **dict(actions)})
    save_json(out/'validation_cost_sensitivity.json', sensitivity)
    summary['validation_cost_sensitivity'] = sensitivity
    lines += ['', '## Declared cost sensitivity on validation states', '',
              'This post-run diagnostic varies resource/latency utility weights on the 30 baseline validation '
              'states. It does not retrain models or change the frozen configuration. No test predictions enter it.', '',
              '| Resource/latency weight multiplier | NORMAL recommendations | CLASS_REWEIGHTING recommendations |',
              '|---:|---:|---:|']
    for row in sensitivity:
        lines.append(f"| {row['resource_latency_weight_multiplier']} | {row.get('NORMAL', 0)} | {row.get('CLASS_REWEIGHTING', 0)} |")
    if reference_declared:
        reference = {seed: json.loads((out/f'reference_ablation/{seed}/proposed/test.json').read_text())
                     for seed in protocol['ablation_seeds']}
        summary['original_assumptions_reference'] = {'metrics': {}, 'seeds': {}}
        lines += ['', '## Original-assumption policy: additional mechanism reference', '',
                  'This arm was declared after calibration but before any new test outcome. It retains '
                  'the original game parameters and the same two available actions. It reuses the main '
                  'baseline for comparison; all seeds, preprocessing, initialization, sample orders, '
                  'training budgets and validation rows match. Its assumed effects/costs are not a '
                  'calibrated recommendation. It is included because the empirically calibrated '
                  'policy has a guaranteed NORMAL recommendation under the declared costs.', '',
                  '| Metric | Original-policy mean ± SD | Paired difference vs baseline mean ± SD |',
                  '|---|---:|---:|']
        for key in keys:
            a=np.array([runs[s,'baseline']['test'][key] for s in protocol['ablation_seeds']])
            b=np.array([reference[s]['test'][key] for s in protocol['ablation_seeds']])
            lines.append(f'| {key} | {mean_sd(b)} | {mean_sd(b-a)} |')
            summary['original_assumptions_reference']['metrics'][key] = {
                'mean':float(b.mean()),'sample_sd':float(b.std(ddof=1)),
                'paired_delta_mean':float((b-a).mean()),'paired_delta_sd':float((b-a).std(ddof=1))}
        lines += ['', '| Class | Original-policy F1 mean ± SD |', '|---|---:|']
        for name in WEAK_CLASSES:
            values=[reference[s]['test']['per_class'][name]['f1'] for s in protocol['ablation_seeds']]
            lines.append(f'| {name} | {mean_sd(values)} |')
        lines += ['', '| Seed | Applied decisions followed by training | First epoch matches baseline | Largest defender probability change | Largest attacker probability change |',
                  '|---|---:|---|---:|---:|']
        for seed in protocol['ablation_seeds']:
            history=json.loads((out/f'reference_ablation/{seed}/proposed/history.json').read_text())
            diag=json.loads((out/f'reference_ablation/{seed}/proposed/probability_diagnostics.json').read_text())
            applied=sum(r['decision']['applied_action']=='CLASS_REWEIGHTING' for r in history[:-1])
            matched=history[0]['validation']==histories[seed,'baseline'][0]['validation']
            summary['original_assumptions_reference']['seeds'][seed]={
                'applied_updates_followed_by_training':applied,'first_epoch_matches_baseline':matched,
                'probability_diagnostics':diag}
            lines.append(f'| {seed} | {applied} | {matched} | {diag["p_def"]["max_epoch_change"]:.6f} | {diag["p_att"]["max_epoch_change"]:.6f} |')
        from tools.run_ablation import aggregate
        comparisons=out/'reference_comparison'
        comparisons.mkdir(exist_ok=True)
        aggregate([runs[s,'baseline'] for s in protocol['ablation_seeds']]+list(reference.values()),comparisons)
        all_arms = []
        for seed in protocol['ablation_seeds']:
            for label, run in [('Baseline', runs[seed, 'baseline']),
                               ('Calibrated policy', runs[seed, 'proposed']),
                               ('Original policy', reference[seed])]:
                item = dict(run)
                item['method'] = label
                all_arms.append(item)
        all_output = out/'three_arm_comparison'
        all_output.mkdir(exist_ok=True)
        aggregate(all_arms, all_output)
        for row in per_class:
            values=np.array([reference[s]['test']['per_class'][row['class']]['f1']
                             for s in protocol['ablation_seeds']])
            row['original_policy_mean']=values.mean()
            row['original_policy_sd']=values.std(ddof=1)
        pd.DataFrame(per_class).to_csv(out/'all_class_f1.csv', index=False)
        lines += ['', 'Reference tables and comparison plot: [reference comparison](reference_comparison/). '
                  'Detailed action, probability, utility and confusion plots: [reference runs](reference_ablation/). '
                  'All three arms: [comparison tables](three_arm_comparison/).', '',
                  '![Three-arm metric comparison](three_arm_comparison/comparison.png)', '']
    save_json(out/'summary.json', summary)
    (out/'REPORT.md').write_text('\n'.join(lines))
    print(out/'REPORT.md')


if __name__ == '__main__':
    main()
