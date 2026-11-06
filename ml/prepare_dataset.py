"""Rebuild the leakage-reviewed split copy from the original folder tree.

This keeps the source dataset untouched and removes only the reviewed paths
listed in results/leakage_removals.csv after checking their source SHA-256.
"""
import argparse, csv, hashlib, json, shutil
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--source',required=True); ap.add_argument('--output',required=True)
    ap.add_argument('--removals',default=str(Path(__file__).resolve().parents[1]/'results'/'leakage_removals.csv'))
    args=ap.parse_args(); source=Path(args.source).resolve(); output=Path(args.output).resolve()
    if output.exists(): raise SystemExit(f'Output already exists: {output}')
    rows=list(csv.DictReader(open(args.removals,newline='',encoding='utf-8-sig')))
    for row in rows:
        p=source/Path(row['removed_path'].replace('\\','/'))
        if not p.is_file(): raise SystemExit(f'Missing source file: {p}')
        digest=hashlib.sha256(p.read_bytes()).hexdigest()
        if digest!=row['sha256']: raise SystemExit(f'Source hash changed for {row["removed_path"]}: expected {row["sha256"]}, got {digest}')
    shutil.copytree(source,output)
    for row in rows: (output/Path(row['removed_path'].replace('\\','/'))).unlink()
    shutil.copy2(args.removals,output/'leakage_removals.csv')
    summary=Path(args.removals).with_name('leakage_removals.json')
    if summary.exists(): shutil.copy2(summary,output/summary.name)
    print(json.dumps({'source':str(source),'prepared_copy':str(output),'removed_files':len(rows),'source_modified':False},indent=2))

if __name__=='__main__': main()
