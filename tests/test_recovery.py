"""Restoration safety and explicit synthetic demonstration provenance."""
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from tools.restore_data import SPLITS, dataset_status, install_splits, require_match
from tools.demo_features import write_synthetic_features


class RecoveryTests(unittest.TestCase):
    def test_restore_requires_every_hash_before_writing(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source = root/'source'; out = root/'out'; manifest = {}
            for key, relative in SPLITS.items():
                path = source/relative; path.parent.mkdir(parents=True)
                content = f'value,label\n{key},attack\n'.encode(); path.write_bytes(content)
                manifest[key] = dict(bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
            self.assertEqual(dataset_status(out, manifest)['status'], 'blocked')
            corrupted = source/SPLITS['test']; original = corrupted.read_bytes()
            corrupted.write_bytes(b'wrong')
            with self.assertRaisesRegex(ValueError, 'No substitution'):
                install_splits(source, out, manifest)
            self.assertFalse(out.exists())
            corrupted.write_bytes(original)
            self.assertEqual(install_splits(source, out, manifest)['status'], 'verified')
            self.assertEqual(install_splits(source, out, manifest)['status'], 'verified')
            (out/SPLITS['train']).write_bytes(b'keep this file')
            with self.assertRaises(ValueError): install_splits(source, out, manifest)
            self.assertEqual((out/SPLITS['train']).read_bytes(), b'keep this file')

    def test_matching_size_with_wrong_content_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/'data'; p.write_bytes(b'1234')
            expected = dict(bytes=4, sha256=hashlib.sha256(b'abcd').hexdigest())
            with self.assertRaises(ValueError): require_match(p, expected)

    def test_synthetic_fixture_explicit_provenance_and_repeatability(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            prep = dict(features=['a', 'b'], mean=[1., 2.], scale=[.5, 1.])
            (root/'preprocessing.json').write_text(json.dumps(prep))
            first = write_synthetic_features(root, root/'first.csv', count=5)
            second = write_synthetic_features(root, root/'second.csv', count=5)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            with first.open() as stream: rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 5)
            self.assertTrue(all(r['_qarmor_source'] == 'synthetic_smoke' for r in rows))
            with self.assertRaises(FileExistsError): write_synthetic_features(root, first)
            with self.assertRaises(ValueError): write_synthetic_features(root, root/'zero.csv', count=0)



class PackagingTests(unittest.TestCase):
    def test_archive_includes_untracked_data_but_not_environment_or_keys(self):
        from tools.package_project import package
        import zipfile
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)/'project'; root.mkdir()
            (root/'data').mkdir(); (root/'data/train.csv').write_text('label\nattack\n')
            (root/'.venv').mkdir(); (root/'.venv/large-file').write_text('environment')
            (root/'device.private.pem').write_text('private')
            (root/'README.md').write_text('project')
            result = package(root, Path(folder)/'project.zip')
            with zipfile.ZipFile(result['zip_path']) as archive:
                names = archive.namelist()
                self.assertIn('project/data/train.csv', names)
                self.assertIn('project/README.md', names)
                self.assertFalse(any('.venv' in n or '.private.pem' in n for n in names))
                manifest = json.loads(archive.read('project/DELIVERY_MANIFEST.json'))
                self.assertEqual(manifest['file_count'], 2)
            with self.assertRaises(FileExistsError): package(root, Path(result['zip_path']))
            with self.assertRaises(ValueError): package(root, root/'nested.zip')



class CanonicalArchiveTests(unittest.TestCase):
    def test_repacked_members_use_original_sampling_seed_names(self):
        from tools.prepare_archive_splits import canonical_sources
        original, original_map = canonical_sources(['CSV/XSS/a.csv', 'CSV/Benign_Final/b.csv'])
        mirror, mirror_map = canonical_sources(['data/XSS/a.csv', 'data/Benign_Final/b.csv'], 'data')
        self.assertEqual(original, mirror)
        self.assertEqual(mirror_map['CSV/XSS/a.csv'], 'data/XSS/a.csv')
        self.assertEqual(original_map['CSV/XSS/a.csv'], 'CSV/XSS/a.csv')
        with self.assertRaises(ValueError): canonical_sources(['data/XSS/a.csv']*2, 'data')

if __name__ == '__main__':
    unittest.main()
