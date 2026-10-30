"""Dataset structure and exact-duplicate audit for class-folder X-ray datasets."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2

EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
SPLITS = ("train", "val", "test")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_dataset(root: Path) -> dict[str, Any]:
    """Summarize class folders, image readability, dimensions, and exact duplicates."""
    class_sets: dict[str, set[str]] = {}
    split_info: dict[str, Any] = {}
    hash_locations: dict[str, list[dict[str, str]]] = defaultdict(list)
    unreadable: list[str] = []
    dimensions: Counter[str] = Counter()

    for split in SPLITS:
        split_dir = root / split
        if not split_dir.is_dir():
            raise ValueError(f"Missing required split directory: {split_dir}")
        classes = sorted(path.name for path in split_dir.iterdir() if path.is_dir())
        class_sets[split] = set(classes)
        counts: dict[str, int] = {}
        image_count = 0
        for class_name in classes:
            class_dir = split_dir / class_name
            images = sorted(p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
            counts[class_name] = len(images)
            image_count += len(images)
            for path in images:
                relative = path.relative_to(root).as_posix()
                hash_locations[_sha256(path)].append({"split": split, "path": relative, "class": class_name})
                image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    unreadable.append(relative)
                else:
                    height, width = image.shape
                    dimensions[f"{width}x{height}"] += 1
        split_info[split] = {"image_count": image_count, "class_counts": counts}

    duplicate_groups = [locations for locations in hash_locations.values() if len(locations) > 1]
    cross_split_duplicates = [
        locations for locations in duplicate_groups
        if len({item["split"] for item in locations}) > 1
    ]
    reference_classes = class_sets["train"]
    class_mismatches = {
        split: {"missing": sorted(reference_classes - class_sets[split]),
                "unexpected": sorted(class_sets[split] - reference_classes)}
        for split in SPLITS if class_sets[split] != reference_classes
    }
    empty_class_folders = [
        f"{split}/{class_name}"
        for split, details in split_info.items()
        for class_name, count in details["class_counts"].items()
        if count == 0
    ]
    total_image_count = sum(details["image_count"] for details in split_info.values())
    return {
        "dataset_root": str(root),
        "total_image_count": total_image_count,
        "empty_class_folders": empty_class_folders,
        "splits": split_info,
        "class_mismatches": class_mismatches,
        "image_dimensions": dict(dimensions),
        "unreadable_images": unreadable,
        "exact_duplicate_groups": duplicate_groups,
        "cross_split_exact_duplicate_groups": cross_split_duplicates,
        "notes": [
            "Exact byte-level hashes do not detect re-encoded or near-duplicate images.",
            "Filename hashes do not establish patient-level separation; verify patient IDs and acquisition groups separately.",
        ],
    }


def write_audit(root: Path, output: Path) -> dict[str, Any]:
    result = audit_dataset(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
