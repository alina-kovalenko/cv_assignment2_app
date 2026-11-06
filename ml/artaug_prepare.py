"""Deterministic, training-only source selection for the ArtAug photo pilot.

This module does not generate or approve images. It selects five frozen training
images per class and verifies their Drive download hashes. Bundle validation
requires separate manual approval before any examples can be imported.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from pathlib import Path, PurePosixPath
from typing import Any

CLASS_ORDER = ["bottles", "cups", "pens", "scissors"]
SOURCE_MANIFEST = "results/dataset_manifest.csv"
DOWNLOAD_MANIFEST = "data/download_manifest.json"
DEFAULT_PROMPT = (
    "Enhance the aesthetic quality of this image while preserving the exact "
    "object identity, count, shape, color, visible labels, and scene content."
)
PILOT_PER_CLASS = 5
PILOT_SEED = 4127
GENERATOR_ID = "DiffSynth-Studio/FLUX.1-Kontext-dev-lora-ArtAug"
BASE_MODEL_ID = "black-forest-labs/FLUX.1-Kontext-dev"
SCHEMA_VERSION = 1
SOURCE_QC_EXCLUSIONS = {
    "train/pens/77.jpg": "Manual source review: dominant scissors plus pen make the class visually ambiguous for the pen pilot. Excluded from this pilot only; frozen dataset and splits remain unchanged."
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def choose_pilot_rows(repo_root: Path, per_class: int = PILOT_PER_CLASS,
                      seed: int = PILOT_SEED) -> tuple[list[dict[str, Any]], str]:
    """Choose a seeded subset from train only, stable against manifest row order."""
    frozen_path = repo_root / SOURCE_MANIFEST
    download_path = repo_root / DOWNLOAD_MANIFEST
    frozen_bytes = frozen_path.read_bytes()
    frozen_rows = read_csv(frozen_path)
    downloads = json.loads(download_path.read_text(encoding="utf-8"))
    download_by_path = {r["path"].replace("\\", "/"): r for r in downloads}
    if len(download_by_path) != len(downloads):
        raise ValueError("Download manifest contains duplicate paths")

    train_by_class: dict[str, list[dict[str, Any]]] = {name: [] for name in CLASS_ORDER}
    for row in frozen_rows:
        if row.get("split") != "train":
            continue
        label = row.get("class", "")
        path = row["path"].replace("\\", "/")
        if label not in train_by_class:
            raise ValueError(f"Unexpected class in frozen manifest: {label}")
        source = download_by_path.get(path)
        if not source or source["sha256"].lower() != row["sha256"].lower():
            raise ValueError(f"Download manifest does not match frozen hash for {path}")
        if path not in SOURCE_QC_EXCLUSIONS:
            train_by_class[label].append({**row, **source, "path": path})

    chooser = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for label in CLASS_ORDER:
        candidates = sorted(train_by_class[label], key=lambda item: item["path"])
        if len(candidates) < per_class:
            raise ValueError(f"Only {len(candidates)} train images available for {label}")
        selected.extend(chooser.sample(candidates, per_class))
    return selected, sha256_bytes(frozen_bytes)


def safe_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if (path.is_absolute() or ".." in path.parts or not value or "\\" in value
            or ":" in value or not value.startswith("images/")):
        raise ValueError(f"Unsafe relative path: {value!r}")
    return path.as_posix()


def validate_import_bundle(bundle_path: Path, source_root: Path) -> dict[str, Any]:
    """Validate the full generated bundle; returns approved examples only."""
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    if bundle.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported ArtAug bundle schema")
    if bundle.get("source_manifest_sha256") != sha256_file(source_root / SOURCE_MANIFEST):
        raise ValueError("The frozen source manifest hash does not match this repo")
    if not isinstance(bundle.get("generator"), dict):
        raise ValueError("Missing generator adapter provenance")
    generator = bundle["generator"]
    if generator.get("adapter_id") != GENERATOR_ID or generator.get("base_model_id") != BASE_MODEL_ID:
        raise ValueError("Bundle does not identify the expected ArtAug adapter and Kontext base")
    if generator.get("synthetic_images") is not True:
        raise ValueError("Generator provenance must mark these outputs as synthetic")
    examples = bundle.get("examples")
    if not isinstance(examples, list):
        raise ValueError("Bundle must contain an examples list")
    frozen = {r["path"].replace("\\", "/"): r for r in read_csv(source_root / SOURCE_MANIFEST)}
    seen: set[str] = set()
    approved: list[dict[str, Any]] = []
    for row in examples:
        identifier = row.get("id", "")
        if not re.fullmatch(r"[a-f0-9]{16}", identifier) or identifier in seen:
            raise ValueError(f"Invalid or repeated example id: {identifier!r}")
        seen.add(identifier)
        if row.get("class_name") not in CLASS_ORDER:
            raise ValueError(f"Invalid class: {row.get('class_name')!r}")
        source = frozen.get(str(row.get("source_manifest_path", "")).replace("\\", "/"))
        if source is None or source["split"] != "train" or source["class"] != row["class_name"]:
            raise ValueError(f"Example {identifier} is not linked to a training image of its class")
        if row.get("source_sha256") != source["sha256"]:
            raise ValueError(f"Source hash mismatch for {identifier}")
        if row.get("original_sha256") != source["sha256"]:
            raise ValueError(f"Original image is not byte-identical to source for {identifier}")
        for key in ("original", "enhanced"):
            rel = safe_relative_path(row.get(key, ""))
            path = bundle_path.parent / rel
            if not path.resolve().is_relative_to(bundle_path.parent.resolve()):
                raise ValueError(f"Image escapes bundle directory: {rel}")
            if not path.is_file():
                raise ValueError(f"Missing {key} image: {rel}")
            if path.suffix.lower() not in (".jpg", ".jpeg") or path.read_bytes()[:3] != b"\xff\xd8\xff":
                raise ValueError(f"Expected JPEG image bytes for {identifier}: {rel}")
            actual = sha256_file(path)
            hash_key = "original_sha256" if key == "original" else "enhanced_sha256"
            if actual != row.get(hash_key):
                raise ValueError(f"{key} image hash mismatch for {identifier}")
        if not row.get("prompt") or row.get("seed") is None:
            raise ValueError(f"Missing prompt or seed for {identifier}")
        if not isinstance(row.get("approved"), bool):
            raise ValueError(f"Approval must be a JSON boolean for {identifier}")
        if row.get("approved") is True:
            approved.append(row)
    if not approved:
        raise ValueError("No examples have been manually approved")
    return {**bundle, "examples": approved}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=PILOT_PER_CLASS)
    parser.add_argument("--seed", type=int, default=PILOT_SEED)
    args = parser.parse_args()
    rows, fingerprint = choose_pilot_rows(args.repo, args.per_class, args.seed)
    print(json.dumps({
        "source_manifest": SOURCE_MANIFEST,
        "source_manifest_sha256": fingerprint,
        "seed": args.seed,
        "per_class": args.per_class,
        "class_order": CLASS_ORDER,
        "pilot_source_qc_exclusions": SOURCE_QC_EXCLUSIONS,
        "selected": [{"class_name": row["class"], "path": row["path"],
                      "drive_id": row["id"], "sha256": row["sha256"]} for row in rows],
        "generation_enabled": False,
    }, indent=2))


if __name__ == "__main__":
    main()
