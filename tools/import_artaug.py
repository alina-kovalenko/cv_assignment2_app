"""Copy manually approved, hash-verified ArtAug examples into app assets."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))
from artaug_prepare import validate_import_bundle  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True,
                        help="Folder with manifest.json and images/ from Colab")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1],
                        help="ObjectLab repo containing the frozen manifests")
    parser.add_argument("--output", type=Path,
                        help="Destination app/src/main/assets/artaug directory")
    args = parser.parse_args()
    bundle_root = args.bundle.resolve()
    repo_root = args.repo.resolve()
    output = (args.output or repo_root / "android/app/src/main/assets/artaug").resolve()
    validated = validate_import_bundle(bundle_root / "manifest.json", repo_root)
    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing ArtAug assets: {output}")

    output.mkdir(parents=True)
    (output / "images").mkdir()
    out_examples = []
    for row in validated["examples"]:
        new_row = dict(row)
        for key in ("original", "enhanced"):
            rel = new_row[key]
            target_rel = f"images/{new_row['id']}_{key}.jpg"
            target = output / target_rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(bundle_root / rel, target)
            new_row[key] = target_rel
        out_examples.append(new_row)
    validated["examples"] = out_examples
    (output / "manifest.json").write_text(
        json.dumps(validated, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "approved_example_count": len(out_examples),
                      "classes": sorted({r["class_name"] for r in out_examples})}, indent=2))


if __name__ == "__main__":
    main()
