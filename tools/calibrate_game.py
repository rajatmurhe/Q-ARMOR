"""Paired validation-only calibration, with a separate audit partition.

Both branches start from identical model/optimizer state and matched random
seeds. Effect estimates refer to the next training epoch, including ordinary
learning under NORMAL. Costs retain prototype resource/latency assumptions.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from game_theory.ids_training import metrics, predict_labels, save_json
from .data_pipeline import (resolve_data_paths, preprocess_splits, prepare_features,
                            hybrid_balance, validation_partition)


def fit_candidate(records, requested_scale):
    """Use a preselected scale; aggregate signed effects before bounding them."""
    rows = [r for r in records if np.isclose(r['requested_scale'], requested_scale)]
    if not rows:
        raise ValueError('No observations at the preselected calibration scale')
    effects, false_alarms = {}, {}
    for method, action in [('control', 'NORMAL'), ('reweighted', 'CLASS_REWEIGHTING')]:
        observed = []
        for row in rows:
            before = row['before']['per_class'][row['target']]['f1']
            after = row[method]['per_class'][row['target']]['f1']
            observed.append((after - before) / max(1 - before, 1e-8))
        # Three seeds do not identify a reliable two-parameter response curve.
        effects[action] = [float(np.clip(np.mean(observed), 0, .95)), 0.0]
        fars = [r[method]['false_alarm_rate'] for r in rows
                if r[method]['false_alarm_rate'] is not None]
        if fars:
            false_alarms[action] = float(np.mean(fars))
    return {'effectiveness': effects, 'false_positive_cost': false_alarms}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir')
    p.add_argument('--output', default='results/calibration')
    p.add_argument('--seeds', type=int, nargs='+', default=[101, 102, 103])
    p.add_argument('--scales', type=float, nargs='+', default=[1.15, 1.5])
    p.add_argument('--candidate-scale', type=float, default=1.15)
    p.add_argument('--warmup-epochs', type=int, default=5)
    p.add_argument('--batch-fraction', type=float, default=1.)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--minority-target', type=int, default=1000)
    p.add_argument('--majority-target', type=int, default=1000)
    p.add_argument('--device', choices=['auto', 'cpu', 'metal'], default='auto')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if (not 0 < args.batch_fraction <= 1 or args.warmup_epochs < 1
            or any(not 1 < s <= 2.5 for s in args.scales)
            or args.candidate_scale not in args.scales):
        p.error('Invalid warmup/fraction/scales or candidate scale absent')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    protocol = {k: v for k, v in vars(args).items() if k != 'resume'}
    paths = resolve_data_paths(args.data_dir)
    # Never open, hash or inspect the test CSV in calibration.
    protocol['source_sha256'] = {k: hashlib.sha256(paths[k].read_bytes()).hexdigest()
                                 for k in ['train', 'val']}
    if (out / 'protocol.json').exists():
        if not args.resume or json.loads((out / 'protocol.json').read_text()) != protocol:
            p.error('Output exists; use a new directory or --resume with identical inputs')
    save_json(out / 'protocol.json', protocol)
    from .training_runtime import configure_runtime
    runtime = configure_runtime(args.device, args.threads)
    save_json(out / 'runtime.json', runtime)
    import tensorflow as tf
    import pandas as pd
    from sklearn.utils.class_weight import compute_class_weight
    from .ids_model import build_enhanced_cnn_lstm_attention_model
    train, val = pd.read_csv(paths['train']), pd.read_csv(paths['val'])
    cal_idx, audit_idx = validation_partition(val.label.to_numpy())
    save_json(out / 'validation_partition.json', {
        'seed': 2026, 'calibration_rows': cal_idx.tolist(), 'audit_rows': audit_idx.tolist(),
        'purpose': 'Targets and candidate coefficients use calibration only. Audit is reported without tuning.'})
    combined, encoder = preprocess_splits(train, val, val)
    processed, features, _, _ = prepare_features(combined)
    Xt = processed[processed.dataset.eq('train')][features].to_numpy(dtype=np.float32)
    yt = processed[processed.dataset.eq('train')].label_encoded.to_numpy(dtype=int)
    Xv = processed[processed.dataset.eq('val')][features].to_numpy(dtype=np.float32)
    yv = processed[processed.dataset.eq('val')].label_encoded.to_numpy(dtype=int)
    names = encoder.classes_.tolist()
    records = json.loads((out / 'paired_probes.json').read_text()) if args.resume and (out / 'paired_probes.json').exists() else []

    def build():
        model = build_enhanced_cnn_lstm_attention_model((Xt.shape[1],), len(names))
        model.compile(optimizer=tf.keras.optimizers.Adam(5e-4, clipnorm=1.),
                      loss='categorical_crossentropy')
        model.optimizer.build(model.trainable_variables)
        return model

    def evaluate(model):
        pred = predict_labels(model, Xv, names)
        return (metrics(yv[cal_idx], pred[cal_idx], names),
                metrics(yv[audit_idx], pred[audit_idx], names))

    for seed in args.seeds:
        if all(any(r['seed'] == seed and r['requested_scale'] == s for r in records) for s in args.scales):
            continue
        seedout = out / str(seed)
        seedout.mkdir(exist_ok=True)
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)
        X, y = hybrid_balance(Xt, yt, args.minority_target, args.majority_target, seed)
        rng = np.random.default_rng(seed)
        onehot = np.eye(len(names), dtype=np.float32)[y]
        base = compute_class_weight('balanced', classes=np.arange(len(names)), y=y)
        base /= base.mean()
        model = build()
        for epoch in range(args.warmup_epochs):
            idx = rng.permutation(len(X))[:max(1, int(len(X) * args.batch_fraction))]
            # A partially completed seed restarts its warmup from its fixed seed.
            # Do not pretend weights alone restore the complete dropout RNG state.
            saved = seedout / 'warmup.weights.h5'
            started = time.perf_counter()
            model.fit(X[idx], onehot[idx], epochs=1, batch_size=args.batch_size,
                      shuffle=False, class_weight=dict(enumerate(base)), verbose=0)
            before, audit_before = evaluate(model)
            model.save_weights(saved)  # Keras saves optimizer variables with compiled weights.
            save_json(seedout / f'warmup_{epoch + 1}.json',
                      {'calibration': before, 'audit': audit_before,
                       'seconds': time.perf_counter() - started})
            print(f'Calibration seed {seed}, warmup {epoch + 1}/{args.warmup_epochs}: '
                  f'macro-F1={before["macro_f1"]:.4f}, {time.perf_counter()-started:.1f}s', flush=True)
        before, audit_before = evaluate(model)
        target = min((n for n in names if n not in {'BenignTraffic', 'Benign', 'BENIGN'}),
                     key=lambda n: before['per_class'][n]['f1'])
        snapshot = model.get_weights()
        optimizer = [v.numpy().copy() for v in model.optimizer.variables]
        idx = rng.permutation(len(X))[:max(1, int(len(X) * args.batch_fraction))]
        control = None
        for requested_scale in args.scales:
            if any(r['seed'] == seed and r['requested_scale'] == requested_scale for r in records):
                continue
            # Match the controller's first scale increase at the measured weakness.
            scale = 1 + (requested_scale - 1) * (1 - before['per_class'][target]['f1'])
            pair, audits, timings = {}, {}, {}
            for method in ['control', 'reweighted']:
                if method == 'control' and control is not None:
                    pair[method], audits[method], timings[method] = control
                    continue
                tf.keras.backend.clear_session()
                tf.keras.utils.set_random_seed(seed + 1)
                branch = build()
                branch.set_weights(snapshot)
                for var, value in zip(branch.optimizer.variables, optimizer):
                    var.assign(value)
                w = base.copy()
                if method == 'reweighted':
                    w[names.index(target)] *= scale
                w /= w.mean()
                started = time.perf_counter()
                branch.fit(X[idx], onehot[idx], epochs=1, batch_size=args.batch_size,
                           shuffle=False, class_weight=dict(enumerate(w)), verbose=0)
                timings[method] = time.perf_counter() - started
                pair[method], audits[method] = evaluate(branch)
                if method == 'control':
                    control = pair[method], audits[method], timings[method]
            c, t = pair['control'], pair['reweighted']
            ac, at = audits['control'], audits['reweighted']
            row = dict(seed=seed, requested_scale=requested_scale, scale=scale, target=target,
                       before=before, audit_before=audit_before, **pair, audit=audits,
                       training_seconds=timings,
                       target_f1_delta=t['per_class'][target]['f1']-c['per_class'][target]['f1'],
                       macro_f1_delta=t['macro_f1']-c['macro_f1'],
                       audit_target_f1_delta=at['per_class'][target]['f1']-ac['per_class'][target]['f1'],
                       audit_macro_f1_delta=at['macro_f1']-ac['macro_f1'])
            records.append(row)
            save_json(out / 'paired_probes.json', records)
            save_json(out / 'status.json', {'status': 'running', 'completed_probes': len(records),
                                          'expected_probes': len(args.seeds)*len(args.scales)})
            print(f'Calibration seed {seed}, scale {scale:.4f}: target delta={row["target_f1_delta"]:+.4f}, '
                  f'audit delta={row["audit_target_f1_delta"]:+.4f}', flush=True)
    table = pd.DataFrame([{k: r[k] for k in ['seed', 'requested_scale', 'scale', 'target',
                            'target_f1_delta', 'macro_f1_delta', 'audit_target_f1_delta',
                            'audit_macro_f1_delta']} for r in records])
    table.to_csv(out / 'effects.csv', index=False)
    table.groupby('requested_scale')[[c for c in table if c.endswith('_delta')]].agg(['mean','std','count']).to_csv(out / 'sensitivity.csv')
    candidate = fit_candidate(records, args.candidate_scale)
    save_json(out / 'candidate_game_config.json', candidate)
    save_json(out / 'status.json', {'status': 'complete_validation_probes', 'arguments': vars(args),
        'completed_probes': len(records), 'candidate_scale_preselected': args.candidate_scale,
        'note': 'Constant empirical next-epoch effects from calibration partition; signed changes averaged before bounding. Audit partition is descriptive only. Resource/latency costs remain prototype assumptions; generalization to other states/scales is unproven. No test rows used.'})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ax = table.groupby('requested_scale')[['target_f1_delta', 'audit_target_f1_delta',
                                         'macro_f1_delta', 'audit_macro_f1_delta']].mean().plot(marker='o')
    ax.set(ylabel='Paired validation F1 change', xlabel='Maximum first-step scale')
    ax.figure.tight_layout()
    ax.figure.savefig(out / 'scale_sensitivity.png')
    plt.close(ax.figure)


if __name__ == '__main__':
    main()
