"""Frozen checkpoint inference without unsafe deserialization of Lambda models."""
import hashlib
import json
from pathlib import Path
import numpy as np
from Agent.feature_extractor import extract_features


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class InferenceEngine:
    def __init__(self, bundle):
        from tools.training_runtime import configure_runtime
        configure_runtime('cpu', 4)
        import tensorflow as tf
        from tools.ids_model import build_enhanced_cnn_lstm_attention_model
        self.bundle = Path(bundle).resolve()
        self.manifest = json.loads((self.bundle / 'manifest.json').read_text())
        if self.manifest.get('schema_version') != 1:
            raise ValueError('Unsupported deployment bundle schema')
        for name in ['model.weights.h5', 'preprocessing.json']:
            if file_hash(self.bundle / name) != self.manifest['sha256'][name]:
                raise ValueError(f'Bundle checksum mismatch: {name}')
        architecture = Path(__file__).resolve().parents[1] / 'tools/ids_model.py'
        if file_hash(architecture) != self.manifest['architecture_sha256']:
            raise ValueError('Checkpoint architecture differs from deployment source')
        prep = json.loads((self.bundle / 'preprocessing.json').read_text())
        self.features = prep['features']
        self.classes = prep['classes']
        self.medians = prep['medians']
        self.mean = np.asarray(prep['mean'], dtype=np.float64)
        self.scale = np.asarray(prep['scale'], dtype=np.float64)
        if (not self.features or len(set(self.features)) != len(self.features)
                or self.mean.shape != (len(self.features),) or self.scale.shape != self.mean.shape
                or not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all()
                or np.any(self.scale <= 0) or not self.classes):
            raise ValueError('Invalid deployment preprocessing schema')
        self.benign = self.manifest['benign_label']
        if self.benign not in self.classes:
            raise ValueError('Benign label is absent from model classes')
        tf.keras.utils.set_random_seed(self.manifest['seed'])
        self.model = build_enhanced_cnn_lstm_attention_model((len(self.features),), len(self.classes))
        self.model.load_weights(self.bundle / 'model.weights.h5')

    def transform(self, records):
        raw = np.asarray([extract_features(r, self.features, self.medians) for r in records], dtype=np.float64)
        x = ((raw - self.mean) / self.scale).astype(np.float32)
        if not np.isfinite(x).all():
            raise ValueError('Feature values exceed representable model input range')
        return x

    def predict(self, records):
        if not records:
            return []
        probabilities = np.asarray(self.model(self.transform(records), training=False))
        if probabilities.shape != (len(records), len(self.classes)) or not np.isfinite(probabilities).all():
            raise ValueError('Invalid model output')
        ids = probabilities.argmax(axis=1)
        return [dict(label=self.classes[int(i)], confidence=float(p[int(i)]),
                     malicious=self.classes[int(i)] != self.benign)
                for i, p in zip(ids, probabilities)]
