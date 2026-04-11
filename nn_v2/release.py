"""
Release — package trained CNN models for live trading.

Copies CNN weights from training output to checkpoints/live_release/
with a manifest tracking what was released and when.

Usage:
    python -m nn_v2.release              # package latest models
    python -m nn_v2.release --check      # verify release integrity

The live engine loads from checkpoints/live_release/ via:
    BlendedEngine(use_cnn=True, release_dir='checkpoints/live_release/')
"""
import hashlib
import json
import os
import shutil
import time
from datetime import datetime

# Source paths (training output)
SOURCE_DIR = 'nn_v2/output/nn'
CNN_FILES = ['cnn_flip.pt', 'cnn_hold.pt', 'cnn_risk.pt']

# Release destination
RELEASE_DIR = 'checkpoints/live_release'


def _file_hash(path: str) -> str:
    """SHA256 of a file (first 12 chars)."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()[:12]


def release(source_dir: str = SOURCE_DIR, release_dir: str = RELEASE_DIR):
    """Package CNN models into release directory."""
    os.makedirs(release_dir, exist_ok=True)

    manifest = {
        'released_at': datetime.now().isoformat(),
        'released_ts': time.time(),
        'source_dir': source_dir,
        'models': {},
    }

    missing = []
    for fname in CNN_FILES:
        src = os.path.join(source_dir, fname)
        if not os.path.exists(src):
            missing.append(fname)
            continue

        dst = os.path.join(release_dir, fname)
        shutil.copy2(src, dst)

        manifest['models'][fname] = {
            'hash': _file_hash(dst),
            'size_kb': os.path.getsize(dst) // 1024,
            'source_mtime': datetime.fromtimestamp(
                os.path.getmtime(src)).isoformat(),
        }
        print(f'  {fname} -> {dst} ({manifest["models"][fname]["size_kb"]} KB)')

    if missing:
        print(f'\n  WARNING: missing models: {missing}')
        manifest['missing'] = missing

    # Write manifest
    manifest_path = os.path.join(release_dir, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f'\n  Manifest: {manifest_path}')
    print(f'  Released {len(manifest["models"])} models to {release_dir}')

    return manifest


def check(release_dir: str = RELEASE_DIR):
    """Verify release integrity."""
    manifest_path = os.path.join(release_dir, 'manifest.json')
    if not os.path.exists(manifest_path):
        print(f'  NO RELEASE found at {release_dir}')
        return False

    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f'  Release: {manifest["released_at"]}')
    ok = True
    for fname, info in manifest.get('models', {}).items():
        path = os.path.join(release_dir, fname)
        if not os.path.exists(path):
            print(f'  MISSING: {fname}')
            ok = False
            continue
        current_hash = _file_hash(path)
        match = current_hash == info['hash']
        status = 'OK' if match else 'HASH MISMATCH'
        print(f'  {fname}: {info["size_kb"]} KB | {status}')
        if not match:
            ok = False

    if manifest.get('missing'):
        print(f'  WARNING: {len(manifest["missing"])} models missing at release time')

    return ok


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Release CNN models for live trading')
    parser.add_argument('--check', action='store_true', help='Verify release integrity')
    parser.add_argument('--source', default=SOURCE_DIR, help='Source model directory')
    parser.add_argument('--dest', default=RELEASE_DIR, help='Release directory')
    args = parser.parse_args()

    if args.check:
        ok = check(args.dest)
        if not ok:
            exit(1)
    else:
        release(args.source, args.dest)
