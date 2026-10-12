"""One maintained epoch/validation/game/actuation loop; accepts compiled Keras models."""
from dataclasses import asdict
from pathlib import Path
import json
import time
import numpy as np
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix)
from sklearn.utils.class_weight import compute_class_weight
from .training_bridge import StackelbergIDSController

WEAK_CLASSES = ['Backdoor_Malware', 'BrowserHijacking', 'CommandInjection',
                'SqlInjection', 'Uploading_Attack', 'VulnerabilityScan', 'XSS']


def metrics(y_true, y_pred, class_names):
    labels = np.arange(len(class_names))
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if len(y_true) == 0 or y_true.shape != y_pred.shape or y_true.ndim != 1:
        raise ValueError('Metrics require nonempty, equal-length integer label vectors')
    if any(not np.isin(y, labels).all() for y in (y_true, y_pred)):
        raise ValueError('Labels outside the model encoding')
    p, r, f, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    benign = [i for i, n in enumerate(class_names) if n in {'BenignTraffic', 'Benign', 'BENIGN'}]
    far = None
    if benign and cm[benign, :].sum():
        far = float(1 - cm[np.ix_(benign, benign)].sum() / cm[benign, :].sum())
    return {'accuracy': float(accuracy_score(y_true, y_pred)),
            'macro_f1': float(f.mean()), 'weighted_f1': float(np.average(f, weights=support)),
            'macro_precision': float(p.mean()), 'macro_recall': float(r.mean()),
            'false_alarm_rate': far,
            'per_class': {name: {'precision': float(p[i]), 'recall': float(r[i]),
                                 'f1': float(f[i]), 'support': int(support[i])}
                          for i, name in enumerate(class_names)},
            'confusion_matrix': cm.tolist()}


def predict_labels(model, X, class_names, batch_size=1024):
    prob = np.asarray(model.predict(X, batch_size=batch_size, verbose=0))
    if prob.shape != (len(X), len(class_names)) or not np.isfinite(prob).all():
        raise ValueError('Prediction shape or finite values do not match label encoding')
    return prob.argmax(axis=1)


def json_value(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    raise TypeError(type(value).__name__)


def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=json_value, allow_nan=False))
    temporary.replace(path)


def train_ids(model, X_train, y_train, X_val, y_val, class_names, *, epochs=10,
              batch_size=256, batch_fraction=1.0, seed=42, adaptive=True,
              controller_options=None, output_dir='results/notebook', verbose=1,
              evidence_label='Original validation'):
    """Train same model and optimizer; test split never enters control decisions.

    Checkpoint selection uses validation macro-F1. Baseline and proposed use
    identical sampling/epochs; output weights are used only by the next epoch.
    """
    class_names = [str(n) for n in class_names]
    if len(class_names) != len(set(class_names)) or not class_names:
        raise ValueError('Unique ordered class names required')
    y_train, y_val = np.asarray(y_train), np.asarray(y_val)
    labels = np.arange(len(class_names))
    if (epochs < 1 or batch_size < 1 or not 0 < batch_fraction <= 1
            or len(X_train) != len(y_train) or len(X_val) != len(y_val)
            or not np.array_equal(np.unique(y_train), labels)
            or not np.isin(y_val, labels).all()):
        raise ValueError('Invalid training budget or label encoding; all classes must occur in training')
    weights = dict(enumerate(compute_class_weight('balanced', classes=labels, y=y_train)))
    weights = {k: float(v / np.mean(list(weights.values()))) for k, v in weights.items()}
    controller = StackelbergIDSController(class_names, weights, **(controller_options or {})) if adaptive else None
    one_hot = np.eye(len(labels), dtype=np.float32)[y_train.astype(int)]
    rng = np.random.default_rng(seed)
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    save_json(output/'config.json', dict(seed=seed, epochs=epochs, batch_size=batch_size,
        batch_fraction=batch_fraction, adaptive=adaptive, class_names=class_names,
        controller_options=controller_options or {}, base_weights=weights, evidence_label=evidence_label))
    history = []; best = -1.; checkpoint = output/'best.weights.h5'
    for epoch in range(1, epochs+1):
        start = time.perf_counter()
        used_weights = controller.current_class_weights() if controller else weights.copy()
        ids = rng.permutation(len(X_train))[:max(1, int(len(X_train)*batch_fraction))]
        fit_result = model.fit(np.asarray(X_train)[ids], one_hot[ids], batch_size=batch_size,
                  epochs=1, shuffle=False, class_weight=used_weights, verbose=verbose)
        measured = metrics(y_val, predict_labels(model, X_val, class_names), class_names)
        f1 = {name: values['f1'] for name, values in measured['per_class'].items()}
        decision = asdict(controller.step(f1)) if controller else None
        training_metrics = {k: float(v[-1]) for k, v in
                            getattr(fit_result, 'history', {}).items() if len(v)}
        row = dict(epoch=epoch, validation=measured, decision=decision,
                   training_metrics=training_metrics,
                   training_weights=used_weights, seconds=time.perf_counter()-start)
        history.append(row)
        if measured['macro_f1'] > best:
            best = measured['macro_f1']; model.save_weights(str(checkpoint))
        # Persist every epoch so interrupted runs remain inspectable.
        save_json(output/'history.json', history)
        print(f"Epoch {epoch}: validation macro-F1={measured['macro_f1']:.5f}; "
              f"defence={decision['recommended_defense'] if decision else 'baseline'}")
    model.load_weights(str(checkpoint))
    plot_history(history, class_names, output, evidence_label=evidence_label)
    return history, controller


def plot_history(history, class_names, output, evidence_label='Original validation'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output); x = [r['epoch'] for r in history]
    def plot(values, names, title, filename):
        fig, ax = plt.subplots(figsize=(12, 6))
        for i, name in enumerate(names): ax.plot(x, np.asarray(values)[:, i], label=name)
        ax.set(xlabel='Epoch', ylabel=title, title=f'{evidence_label}: {title}')
        ax.legend(fontsize=6, ncol=3); fig.tight_layout()
        fig.savefig(output/filename, dpi=150); plt.close(fig)
    weak = [n for n in WEAK_CLASSES if n in class_names] or class_names
    plot([[r['validation']['per_class'][n]['f1'] for n in weak] for r in history],
         weak, 'Validation per-class F1', 'weak_f1.png')
    if history[0]['decision']:
        actions = [item['defense_action']
                   for item in history[0]['decision']['all_action_results']]
        attacks = list(history[0]['decision']['game_f1'])
        for key, names in [('p_def', actions), ('p_att', attacks), ('U_def', actions), ('U_att', attacks)]:
            plot([r['decision'][key] for r in history], names, key, key+'.png')
        plot([[r['decision']['defender_utility'], r['decision']['attacker_utility']] for r in history],
             ['defender', 'attacker'], 'Selected equilibrium utility', 'equilibrium_utility.png')
        diag = {}
        for key in ['p_def', 'p_att']:
            a = np.asarray([r['decision'][key] for r in history])
            diag[key] = {'normalized': bool(np.allclose(a.sum(axis=1), 1)),
                         'max_epoch_change': float(np.max(np.abs(np.diff(a, axis=0)))) if len(a)>1 else 0.,
                         'unchanged_reason': 'Equal expected utilities, symmetry, small utility gaps, or convergence can give fixed probabilities; inspect logged U vectors.'}
        save_json(output/'probability_diagnostics.json', diag)
