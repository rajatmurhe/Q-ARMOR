"""CICIoT2023 original split loading and train-only preprocessing/balancing."""
import os
from pathlib import Path
from functools import partial
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif


def validation_partition(labels):
    """Fixed stratified halves: calibration and independent policy audit/selection."""
    from sklearn.model_selection import StratifiedShuffleSplit
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=.5, random_state=2026)
    return next(splitter.split(np.zeros(len(labels)), labels))


def resolve_data_paths(data_dir=None):
    root = Path(data_dir or os.environ.get('QARMOR_DATA_DIR', '/kaggle/input/CICIOT23')).expanduser()
    paths = {name: root/folder/(folder+'.csv') for name, folder in
             [('train', 'train'), ('val', 'validation'), ('test', 'test')]}
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError('Original CICIoT2023 split files missing: '+', '.join(missing)+
                                '. Set QARMOR_DATA_DIR to the directory containing train/, validation/, test/. No dataset substitution was made.')
    return paths


def load_splits(data_dir=None):
    return tuple(pd.read_csv(p) for p in resolve_data_paths(data_dir).values())


def preprocess_splits(train, val, test):
    """Preserve every row and split; never learn statistics or labels from holdouts."""
    encoder = LabelEncoder().fit(train['label'])
    features = list(train.columns.drop('label'))
    if any(set(df.columns) != set(train.columns) for df in (val, test)):
        raise ValueError('Split schemas differ')
    numeric_train = train[features].apply(pd.to_numeric, errors='raise').replace([np.inf, -np.inf], np.nan)
    medians = numeric_train.median().fillna(0.)
    frames = []
    for name, df in zip(['train', 'val', 'test'], [train, val, test]):
        X = df[features].apply(pd.to_numeric, errors='raise').replace([np.inf, -np.inf], np.nan).fillna(medians)
        X['label'] = df['label'].to_numpy()
        X['label_encoded'] = encoder.transform(df['label'])
        X['dataset'] = name
        frames.append(X)
    return pd.concat(frames, ignore_index=True), encoder


def prepare_features(combined, n_features=25, seed=42):
    features = combined.columns.drop(['label', 'label_encoded', 'dataset'])
    training = combined['dataset'].eq('train')
    # Constant-column selection, MI and scaling all fit only the training mask.
    features = [c for c in features if combined.loc[training, c].nunique() > 1]
    if not features: raise ValueError('No nonconstant numeric training features')
    selector = SelectKBest(partial(mutual_info_classif, random_state=seed), k=min(n_features, len(features)))
    selector.fit(combined.loc[training, features], combined.loc[training, 'label_encoded'])
    selected = np.asarray(features)[selector.get_support()].tolist()
    scaler = StandardScaler().fit(combined.loc[training, selected])
    transformed = pd.DataFrame(scaler.transform(combined[selected]), columns=selected)
    transformed['label_encoded'] = combined['label_encoded'].to_numpy()
    transformed['dataset'] = combined['dataset'].to_numpy()
    return transformed, selected, scaler, selector


def hybrid_balance(X, y, minority_threshold=300000, target_majority=300000, seed=42):
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    if minority_threshold < 1 or target_majority < minority_threshold:
        raise ValueError('Require 1 <= minority target <= majority target')
    ids, counts = np.unique(y, return_counts=True)
    under = {int(c): target_majority for c, n in zip(ids, counts) if n > target_majority}
    if under: X, y = RandomUnderSampler(sampling_strategy=under, random_state=seed).fit_resample(X, y)
    over = {int(c): minority_threshold for c, n in zip(ids, counts) if n < minority_threshold}
    if over:
        smallest = min(n for c,n in zip(ids, counts) if c in over)
        if smallest < 2: raise ValueError('SMOTE needs at least two training examples in each minority class')
        X, y = SMOTE(sampling_strategy=over, random_state=seed, k_neighbors=min(3, smallest-1)).fit_resample(X, y)
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=int)
