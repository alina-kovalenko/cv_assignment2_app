"""Create a source ZIP without local caches, image data, builds or signing keys."""
import argparse
import hashlib
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = (args.output or root.parent / "ObjectLab_Source.zip").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    ignored_dirs = {".git", ".gradle", "build", "work", "runs", "__pycache__", ".venv", "venv", ".idea", ".cache", ".ipynb_checkpoints"}
    ignored_suffixes = {".apk", ".aab", ".jks", ".keystore", ".pyc", ".pyo", ".iml", ".zip"}
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(p in ignored_dirs for p in rel.parts) or path.suffix.lower() in ignored_suffixes:
            continue
        if path.name in {"local.properties", ".DS_Store", "Thumbs.db"} or rel.parts[:2] in [("data", "raw"), ("data", "prepared")]:
            continue
        if path.resolve() == output or path.name.endswith(".zip.tmp"):
            continue
        files.append((path, rel.as_posix()))
    manifest = []
    temp = output.with_suffix(output.suffix + ".tmp")
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path, relative in files:
            data = path.read_bytes()
            entry = zipfile.ZipInfo("ObjectLab/" + relative)
            entry.create_system = 3
            entry.external_attr = (0o100755 if relative == "android/gradlew" else 0o100644) << 16
            archive.writestr(entry, data, compress_type=zipfile.ZIP_DEFLATED)
            manifest.append(hashlib.sha256(data).hexdigest() + "  " + relative)
        archive.writestr("ObjectLab/SOURCE_MANIFEST.sha256", "\n".join(manifest) + "\n")
    with zipfile.ZipFile(temp) as archive:
        assert archive.testzip() is None
        assert "ObjectLab/models/baseline.onnx" in archive.namelist()
        assert "ObjectLab/models/sota.onnx" in archive.namelist()
        assert "ObjectLab/android/gradle/wrapper/gradle-wrapper.jar" in archive.namelist()
    temp.replace(output)
    print(f"Packaged {len(files)} source/artifact files: {output} ({output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
