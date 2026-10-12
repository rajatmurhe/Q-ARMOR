"""Run the frozen local CPU protocol; preserve logs and resume finished runs."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from game_theory.ids_training import save_json


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / 'results/remaining_study_v2'
    out.mkdir(parents=True, exist_ok=True)
    protocol = {
        'dataset': 'data/CICIoT23_archive_sample_v1',
        'scope': '34-class bounded archive sample; capture overlap possible in 17 row-split classes',
        'calibration_seeds': [101, 102, 103], 'calibration_warmup_epochs': 5,
        'calibration_scales': [1.15, 1.5], 'candidate_scale_preselected': 1.15,
        'ablation_seeds': [42, 43, 44], 'ablation_epochs': 10,
        'batch_fraction': 1.0, 'balanced_rows_per_class': 1000,
        'architecture': 'unchanged original CNN/LSTM/attention',
        'optimizer': 'Adam learning_rate=0.0005 clipnorm=1',
        'runtime': 'CPU, 4 intra-op / 2 inter-op threads',
        'validation': 'fixed stratified halves, seed 2026; calibration fitting on first, ablation feedback/checkpoint selection on second',
        'checkpoint_selection': 'maximum validation macro-F1 within fixed 10-epoch budget',
        'test': 'fixed sample test already used by historical exploratory runs; no new test-driven tuning permitted',
        'costs': 'false alarm costs measured; resource/latency costs retain prototype assumptions',
        'interpretation': 'report a null policy outcome if game chooses NORMAL; never force an action or select settings using test results',
    }
    if (out/'protocol.json').exists() and json.loads((out/'protocol.json').read_text()) != protocol:
        raise RuntimeError('Existing protocol differs; create a new study version')
    save_json(out/'protocol.json', protocol)
    common = ['--data-dir', protocol['dataset'], '--device', 'cpu', '--threads', '4',
              '--batch-fraction', '1', '--minority-target', '1000', '--majority-target', '1000', '--resume']
    calibration = out/'calibration'
    ablation = out/'ablation'
    commands = [
        ('calibration', [sys.executable, '-u', '-m', 'tools.calibrate_game', *common,
                         '--output', str(calibration), '--seeds', '101', '102', '103',
                         '--scales', '1.15', '1.5', '--candidate-scale', '1.15', '--warmup-epochs', '5']),
        ('ablation', [sys.executable, '-u', '-m', 'tools.run_ablation', *common,
                      '--output', str(ablation), '--seeds', '42', '43', '44', '--epochs', '10',
                      '--validation-subset', 'audit', '--game-config', str(calibration/'candidate_game_config.json')]),
    ]
    for stage, command in commands:
        if stage == 'ablation':
            config = calibration/'candidate_game_config.json'
            save_json(out/'frozen_game_config.json', {'config': json.loads(config.read_text()),
                'sha256': hashlib.sha256(config.read_bytes()).hexdigest(),
                'frozen_before_new_ablation_test_evaluations': True})
        save_json(out/'status.json', {'status': 'running', 'stage': stage, 'command': command})
        print('Starting', stage, flush=True)
        with (out/(stage+'.log')).open('a') as log:
            completed = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode:
            save_json(out/'status.json', {'status': 'failed', 'stage': stage, 'exit_code': completed.returncode})
            raise SystemExit(completed.returncode)
    save_json(out/'status.json', {'status': 'complete', 'calibration': str(calibration), 'ablation': str(ablation)})
    print('Calibration and all six ablation runs completed', flush=True)


if __name__ == '__main__':
    main()
