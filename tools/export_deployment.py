"""Export fixed seed-42 baseline checkpoint; no selection using test scores."""
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from Agent.inference import file_hash


def export_bundle(study, train_csv, output, seed=42, method='baseline'):
    study, train_csv, output = map(Path, [study, train_csv, output])
    if output.exists():
        raise FileExistsError('Use a new output directory; refusing to overwrite a deployment bundle')
    source = study / str(seed)
    prep = joblib.load(source / 'preprocessing.pkl')  # trusted, locally produced training artifact
    config = json.loads((source / method / 'config.json').read_text())
    if prep['encoder'].classes_.tolist() != config['class_names']:
        raise ValueError('Preprocessing classes differ from checkpoint config')
    manifest = json.loads((study / 'data_manifest.json').read_text())
    if file_hash(train_csv) != manifest['train']['sha256']:
        raise ValueError('Training CSV fingerprint differs from checkpoint study')
    features = prep['features']
    if list(prep['scaler'].feature_names_in_) != features:
        raise ValueError('Scaler feature order differs from checkpoint feature order')
    frame = pd.read_csv(train_csv, usecols=features)
    numeric = frame[features].apply(pd.to_numeric, errors='raise').replace([np.inf, -np.inf], np.nan)
    medians = numeric.median().fillna(0.)
    output.mkdir(parents=True)
    shutil.copyfile(source / method / 'best.weights.h5', output / 'model.weights.h5')
    preprocessing = dict(features=features, classes=prep['encoder'].classes_.tolist(),
        medians={n:float(medians[n]) for n in features}, mean=prep['scaler'].mean_.tolist(),
        scale=prep['scaler'].scale_.tolist())
    (output / 'preprocessing.json').write_text(json.dumps(preprocessing, indent=2)+'\n')
    architecture = Path(__file__).resolve().parents[1] / 'tools/ids_model.py'
    result = dict(schema_version=1, seed=seed, method=method, benign_label='BenignTraffic',
        selection='Seed 42 baseline fixed in deployment protocol; checkpoint selected by validation macro-F1.',
        source_checkpoint=str((source / method / 'best.weights.h5').resolve()),
        train_sha256=manifest['train']['sha256'], architecture_sha256=file_hash(architecture),
        sha256={n:file_hash(output/n) for n in ['model.weights.h5','preprocessing.json']},
        scope='Bounded archive sample research model; consumes compatible extracted feature records.')
    (output / 'manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',default='results/remaining_study_v2/ablation')
    p.add_argument('--train-csv',default='data/CICIoT23_archive_sample_v1/train/train.csv')
    p.add_argument('--output',default='deployment/archive_sample_seed42')
    a=p.parse_args();print(json.dumps(export_bundle(a.study,a.train_csv,a.output),indent=2))

if __name__=='__main__':main()
