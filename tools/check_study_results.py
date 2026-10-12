"""Validate saved experiment completeness, split isolation and feedback evidence."""
import hashlib
import json
from pathlib import Path
import numpy as np
from game_theory.ids_training import save_json


def read(path):
    return json.loads(path.read_text())


def main():
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
              'reference_updates_followed_by_training':applied}
    save_json(root/'integrity_checks.json',result)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
