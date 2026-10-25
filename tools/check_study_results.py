"""Validate saved experiment completeness, split isolation and feedback evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from game_theory.ids_training import save_json


def read(path):
    return json.loads(path.read_text())



def check_metric_consistency(result):
    """Recompute published aggregate/per-class metrics from each saved matrix."""
    names = list(result['per_class'])
    cm = np.asarray(result['confusion_matrix'], dtype=float)
    if cm.shape != (len(names), len(names)) or np.any(cm < 0) or cm.sum() <= 0:
        raise ValueError('Invalid saved confusion matrix')
    actual, predicted, true_positive = cm.sum(axis=1), cm.sum(axis=0), cm.diagonal()
    precision = np.divide(true_positive, predicted, out=np.zeros(len(names)), where=predicted != 0)
    recall = np.divide(true_positive, actual, out=np.zeros(len(names)), where=actual != 0)
    f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros(len(names)), where=precision+recall != 0)
    expected = dict(accuracy=float(true_positive.sum()/cm.sum()), macro_precision=float(precision.mean()),
                    macro_recall=float(recall.mean()), macro_f1=float(f1.mean()),
                    weighted_f1=float(np.average(f1, weights=actual)))
    for key, value in expected.items():
        if not np.isclose(result[key], value, atol=1e-12, rtol=1e-12):
            raise ValueError(f'Saved {key} differs from its confusion matrix')
    for i, name in enumerate(names):
        for key, value in [('precision', precision[i]), ('recall', recall[i]), ('f1', f1[i]), ('support', actual[i])]:
            if not np.isclose(result['per_class'][name][key], value, atol=1e-12, rtol=1e-12):
                raise ValueError(f'Saved {name} {key} differs from its confusion matrix')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Optional new audit destination; original evidence remains unchanged')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]/'results/remaining_study_v2'
    assert read(root/'status.json')['status'] == 'complete'
    seeds = [42, 43, 44]
    cal = read(root/'calibration/paired_probes.json')
    assert {(r['seed'], r['requested_scale']) for r in cal} == {
        (seed, scale) for seed in [101, 102, 103] for scale in [1.15, 1.5]}
    partition = read(root/'calibration/validation_partition.json')
    calibration_rows, audit_rows = set(partition['calibration_rows']), set(partition['audit_rows'])
    assert not calibration_rows & audit_rows
    assert calibration_rows | audit_rows == set(range(16586))
    assert set(read(root/'ablation/validation_partition.json')['rows']) == audit_rows
    assert set(read(root/'reference_ablation/validation_partition.json')['rows']) == audit_rows
    frozen = read(root/'frozen_game_config.json')
    assert hashlib.sha256((root/'calibration/candidate_game_config.json').read_bytes()).hexdigest() == frozen['sha256']
    runs, applied = [], {}
    for seed in seeds:
        baseline = read(root/f'ablation/{seed}/baseline/test.json')['test']
        assert baseline == read(root/f'ablation/{seed}/proposed/test.json')['test']
        initial = read(root/f'ablation/{seed}/baseline/history.json')[0]['validation']
        for arm, method in [('ablation','baseline'), ('ablation','proposed'), ('reference_ablation','proposed')]:
            path = root/f'{arm}/{seed}/{method}'
            config, history, test = read(path/'config.json'), read(path/'history.json'), read(path/'test.json')['test']
            check_metric_consistency(test)
            assert config['epochs'] == 10 and config['batch_fraction'] == 1
            assert len(history) == 10 and len(test['per_class']) == 34
            assert np.asarray(test['confusion_matrix']).shape == (34,34)
            assert np.asarray(test['confusion_matrix']).sum() == 16586
            assert all(np.asarray(r['validation']['confusion_matrix']).sum() == len(audit_rows) for r in history)
            assert history[0]['validation'] == initial
            updates = 0
            for previous, following in zip(history, history[1:]):
                decision = previous['decision']
                if decision:
                    assert decision['current_class_weights'] == following['training_weights']
                    if decision['applied_action'] == 'CLASS_REWEIGHTING':
                        updates += 1
            if method == 'proposed' and arm == 'ablation':
                assert config['controller_options']['game_parameters'] == frozen['config']
                assert updates == 0
            if arm == 'reference_ablation':
                assert config['controller_options']['game_parameters'] == {}
                assert updates > 0
                applied[seed] = updates
            runs.append(str(path.relative_to(root)))
    result = {'status':'passed','completed_training_runs':len(runs),'epochs_per_run':10,
              'calibration_pairs':len(cal),'calibration_and_audit_disjoint':True,
              'validation_rows_per_run':len(audit_rows),'test_rows_per_run':16586,
              'all_first_epochs_match_per_seed':True,'frozen_config_verified':True,
              'next_epoch_weights_match_controller_decisions':True,
              'saved_metrics_recomputed_from_confusion_matrices':True,
              'reference_updates_followed_by_training':applied}
    save_json(args.output or root/'integrity_checks.json',result)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
