"""Create a bounded, reproducible CICIoT2023 development split from CSV.zip.

The source archive is organized by class and capture CSV, not by train/validation/
test. Classes with at least three source CSVs are split by source file to reduce
capture leakage. Classes with one or two source CSVs use a deterministic
row-level split and are identified as such in the manifest. Only a bounded,
uniform sample per class is written; this is a development dataset, not the
original full benchmark split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from zipfile import ZipFile

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def retain_smallest_keys(keys, sample, chunk, new_keys, limit):
    """Keep a uniform reservoir by retaining the smallest random priorities."""
    if chunk.empty:
        return keys, sample
    if len(chunk) != len(new_keys):
        raise ValueError("Sampling priorities and source rows differ in length")
    if keys is None:
        old = pd.DataFrame()
    else:
        old = pd.concat([pd.DataFrame({"__sample_key": keys}), sample.reset_index(drop=True)], axis=1)
    fresh = pd.concat([pd.DataFrame({"__sample_key": new_keys}),
                       chunk.reset_index(drop=True)], axis=1)
    old = pd.concat([old, fresh], ignore_index=True)
    priorities = old["__sample_key"].to_numpy()
    if len(old) > limit:
        chosen = np.argpartition(priorities, limit - 1)[:limit]
        old = old.iloc[chosen]
    return old["__sample_key"].reset_index(drop=True), old.drop(columns="__sample_key").reset_index(drop=True)



def canonical_sources(csv_members, source_prefix='CSV'):
    """Map a repacked archive to the original seed names without changing rows."""
    prefix = source_prefix.strip('/')+'/'
    by_class, archive_members = {}, {}
    for name in csv_members:
        if not name.startswith(prefix):
            continue
        relative = name[len(prefix):]
        parts = relative.split('/')
        if len(parts) < 2:
            continue
        canonical = 'CSV/'+relative
        if canonical in archive_members:
            raise ValueError('Duplicate canonical source member: '+canonical)
        label = 'BenignTraffic' if parts[0] == 'Benign_Final' else parts[0]
        by_class.setdefault(label, []).append(canonical)
        archive_members[canonical] = name
    return by_class, archive_members


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("CSV.zip"))
    parser.add_argument("--source-prefix", default="CSV", help="Explicit container prefix; canonical sampling seeds remain CSV/<class>/<file>")
    parser.add_argument("--output", type=Path, default=Path("data/CICIoT23_archive_sample_v1"))
    parser.add_argument("--train-per-class", type=int, default=2000)
    parser.add_argument("--validation-per-class", type=int, default=500)
    parser.add_argument("--test-per-class", type=int, default=500)
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.train_per_class, args.validation_per_class, args.test_per_class,
           args.chunksize) < 1:
        parser.error("Sample counts and chunksize must be positive")
    archive = args.archive.expanduser().resolve()
    destination = args.output.expanduser().resolve()
    if not archive.is_file():
        parser.error(f"Archive not found: {archive}")
    if destination.exists():
        parser.error(f"Output already exists; choose a new path: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    caps = {"train": args.train_per_class,
            "validation": args.validation_per_class,
            "test": args.test_per_class}
    seed_rng = np.random.default_rng(args.seed)
    temp_root = Path(tempfile.mkdtemp(prefix="qarmor_split_", dir=destination.parent if destination.parent.exists() else None))
    try:
        with ZipFile(archive) as zf:
            csv_members = [i.filename for i in zf.infolist()
                           if not i.is_dir() and i.filename.lower().endswith(".csv")]
            by_class, archive_members = canonical_sources(csv_members, args.source_prefix)
            if len(by_class) != 34:
                raise ValueError(f"Expected 34 class folders in CICIoT2023 archive; found {len(by_class)}")

            assignments = {}
            split_method = {}
            samples = {label: {} for label in by_class}
            keys = {label: {} for label in by_class}
            row_counts = {label: {split: 0 for split in caps} for label in by_class}
            headers = None

            for label in sorted(by_class):
                members = sorted(by_class[label])
                if len(members) >= 3:
                    order = seed_rng.permutation(len(members)).tolist()
                    n_holdout = max(1, round(len(members) * 0.10))
                    test_ids = set(order[:n_holdout])
                    validation_ids = set(order[n_holdout:2*n_holdout])
                    assignments[label] = {
                        name: ("test" if index in test_ids else
                               "validation" if index in validation_ids else "train")
                        for index, name in enumerate(members)
                    }
                    split_method[label] = "capture_file_grouped"
                else:
                    assignments[label] = {name: "row_split" for name in members}
                    split_method[label] = "row_level_within_source_files"
                for split in caps:
                    samples[label][split] = None
                    keys[label][split] = None

            for label in sorted(by_class):
                print(f"Reading {label}: {len(by_class[label])} source CSV(s)", flush=True)
                for member in sorted(by_class[label]):
                    assignment = assignments[label][member]
                    if assignment != "row_split":
                        split = assignment
                        limit = caps[split]
                    else:
                        split = "row_split"
                        limit = sum(caps.values())
                    # A distinct deterministic stream per source file makes the
                    # sampling independent of ZIP member traversal internals.
                    file_seed = int.from_bytes(hashlib.sha256(
                        f"{args.seed}:{member}".encode()).digest()[:8], "little")
                    file_rng = np.random.default_rng(file_seed)
                    with zf.open(archive_members[member]) as stream:
                        for chunk in pd.read_csv(stream, chunksize=args.chunksize):
                            if headers is None:
                                headers = list(chunk.columns)
                                if "label" in headers:
                                    raise ValueError("Source CSV unexpectedly already contains a label column")
                            elif list(chunk.columns) != headers:
                                raise ValueError(f"Source schema mismatch in {member}")
                            chunk["label"] = label
                            row_counts[label]["source_rows_read"] = row_counts[label].get("source_rows_read", 0) + len(chunk)
                            if split == "row_split":
                                keys[label]["row_split"], samples[label]["row_split"] = retain_smallest_keys(
                                    keys[label].get("row_split"), samples[label].get("row_split"),
                                    chunk, file_rng.random(len(chunk)), limit)
                            else:
                                row_counts[label][split] += len(chunk)
                                keys[label][split], samples[label][split] = retain_smallest_keys(
                                    keys[label][split], samples[label][split],
                                    chunk, file_rng.random(len(chunk)), limit)
                if split_method[label] == "row_level_within_source_files":
                    sampled = samples[label].pop("row_split")
                    if sampled is None or len(sampled) < len(caps):
                        actual = 0 if sampled is None else len(sampled)
                        raise ValueError(f"{label} has only {actual} rows; at least one per split is required")
                    sampled = sampled.iloc[seed_rng.permutation(len(sampled))].reset_index(drop=True)
                    total = len(sampled)
                    requested_total = sum(caps.values())
                    exact = {split: total * count / requested_total
                             for split, count in caps.items()}
                    actual_caps = {split: max(1, int(np.floor(value)))
                                   for split, value in exact.items()}
                    while sum(actual_caps.values()) > total:
                        eligible = [split for split in caps if actual_caps[split] > 1]
                        split = max(eligible, key=lambda name: actual_caps[name] - exact[name])
                        actual_caps[split] -= 1
                    while sum(actual_caps.values()) < total:
                        split = max(caps, key=lambda name: exact[name] - actual_caps[name])
                        actual_caps[split] += 1
                    start = 0
                    for split in ("train", "validation", "test"):
                        stop = start + actual_caps[split]
                        samples[label][split] = sampled.iloc[start:stop].reset_index(drop=True)
                        row_counts[label][split] = row_counts[label].get("source_rows_read", 0)
                        start = stop
                for split in caps:
                    part = samples[label][split]
                    if part is None or part.empty:
                        raise ValueError(f"No sampled {split} rows for {label}; cannot make a 34-class split")

            split_dirs = {"train": temp_root / "train",
                          "validation": temp_root / "validation",
                          "test": temp_root / "test"}
            for folder in split_dirs.values():
                folder.mkdir(parents=True, exist_ok=True)
            manifest = {
                "dataset": "CICIoT2023 CSV.zip class-folder archive",
                "archive": str(archive),
                "archive_bytes": archive.stat().st_size,
                "source_prefix": args.source_prefix,
                "archive_sha256": sha256(archive),
                "sampling": "Uniform bounded reservoir sample per split/class; seed-stable.",
                "seed": args.seed,
                "caps_per_class": caps,
                "split_policy": "Capture-file groups are held out for classes with >=3 source CSV files. Classes with 1-2 source files use deterministic row-level sampling and split; those classes may have source-capture overlap across folds.",
                "is_original_predefined_split": False,
                "is_full_dataset": False,
                "features": headers,
                "classes": {},
                "source_csvs": {label: {split: [name for name in sorted(by_class[label])
                                                 if assignments[label][name] == split]
                                         for split in caps}
                                for label in sorted(by_class)},
                "row_level_source_csvs": {label: sorted(by_class[label])
                                            for label in sorted(by_class)
                                            if split_method[label] == "row_level_within_source_files"},
            }
            for split, folder in split_dirs.items():
                out = folder / f"{folder.name}.csv"
                with out.open("w", encoding="utf-8", newline="") as target:
                    first = True
                    for label in sorted(by_class):
                        samples[label][split].to_csv(target, index=False, header=first)
                        first = False
            for label in sorted(by_class):
                manifest["classes"][label] = {
                    "source_csv_count": len(by_class[label]),
                    "split_method": split_method[label],
                    "rows_available_read": row_counts[label].get("source_rows_read", 0),
                    "rows_sampled": {split: len(samples[label][split]) for split in caps},
                }
            actual_rows = {split: sum(len(samples[label][split]) for label in by_class)
                           for split in caps}
            manifest["actual_rows_by_split"] = actual_rows
            manifest["actual_total_rows"] = sum(actual_rows.values())
            (temp_root / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temp_root, destination)
        print(f"Created bounded development splits: {destination}", flush=True)
        print(f"Per-class maximum: {caps}; actual rows: {actual_rows}; "
              f"total: {sum(actual_rows.values()):,}", flush=True)
        print("The source archive was read-only; this is not the original predefined split.", flush=True)
    except Exception:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
