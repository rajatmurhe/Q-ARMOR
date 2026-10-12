"""Validate already extracted CICIoT2023 feature records against a frozen schema."""
import math
from collections.abc import Mapping


def extract_features(record, feature_order, medians=None):
    if not isinstance(record, Mapping):
        raise TypeError('Expected a feature mapping from a compatible CICIoT2023 extractor')
    missing = [name for name in feature_order if name not in record]
    if missing:
        raise ValueError('Missing required model features: ' + ', '.join(missing))
    values = []
    for name in feature_order:
        value = record[name]
        if value is None or value == '':
            numeric = math.nan
        else:
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f'Feature {name!r} must be numeric') from exc
        if not math.isfinite(numeric):
            if medians is None or name not in medians:
                raise ValueError(f'Nonfinite feature {name!r} has no training imputation value')
            numeric = float(medians[name])
            if not math.isfinite(numeric):
                raise ValueError(f'Invalid training median for {name!r}')
        values.append(numeric)
    return values
