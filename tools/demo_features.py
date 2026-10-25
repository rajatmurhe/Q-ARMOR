"""Explicitly synthetic numeric feature fixtures for integration tests only."""
import csv
import json
import random
from pathlib import Path


def write_synthetic_features(bundle, output, count=128, seed=42):
    """Generate model-schema inputs, never network traffic or research observations.

    These vectors need not represent physically possible packet windows. Their
    purpose is to exercise real checkpoint inference and transport without data.
    """
    if count < 1:
        raise ValueError('Synthetic record count must be positive')
    prep = json.loads((Path(bundle)/'preprocessing.json').read_text())
    rng = random.Random(seed)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=[*prep['features'], '_qarmor_source'])
        writer.writeheader()
        for _ in range(count):
            row = {name: float(mean) + rng.uniform(-.5, .5)*float(scale)
                   for name, mean, scale in zip(prep['features'], prep['mean'], prep['scale'])}
            row['_qarmor_source'] = 'synthetic_smoke'
            writer.writerow(row)
    return output
