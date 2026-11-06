"""Download the frozen experiment inputs and verify their SHA-256 hashes.

The images remain in the author's Google Drive; they are not stored in Git.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'data' / 'prepared')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    destination = args.output.resolve()
    manifest = json.loads((ROOT / 'data' / 'download_manifest.json').read_text(encoding='utf-8'))
    def retrieve(row):
        target = (destination / row['path']).resolve()
        if not target.is_relative_to(destination):
            raise ValueError('Unsafe dataset path')
        if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == row['sha256']:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        url = 'https://drive.usercontent.google.com/download?id=' + row['id'] + '&export=download&confirm=t'
        failure = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=60) as response:
                    content = response.read()
                if hashlib.sha256(content).hexdigest() != row['sha256']:
                    raise ValueError('Content hash changed or download was not an image: ' + row['path'])
                part = target.with_suffix(target.suffix + '.part')
                part.write_bytes(content)
                part.replace(target)
                return
            except Exception as error:
                failure = error
                time.sleep(attempt + 1)
        raise failure
    failed = []
    with ThreadPoolExecutor(max_workers=max(1, min(args.workers, 8))) as executor:
        futures = {executor.submit(retrieve, row): row for row in manifest}
        for index, future in enumerate(as_completed(futures), 1):
            try:
                future.result()
            except Exception as error:
                failed.append((futures[future]['path'], str(error)))
            if index % 50 == 0 or index == len(manifest):
                print(f'{index}/{len(manifest)} checked; {len(failed)} failed', flush=True)
    if failed:
        raise RuntimeError('Failed downloads: ' + repr(failed))
    print('Verified dataset:', destination)

if __name__ == '__main__':
    main()
