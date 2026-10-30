"""Prepare a reproducible class-stratified pilot split from the Kaggle layout."""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _class_image_directories(source: Path) -> dict[str, Path]:
    if not source.is_dir():
        raise ValueError(f"Source directory does not exist: {source}")
    result = {}
    for class_dir in sorted(path for path in source.iterdir() if path.is_dir()):
        image_dir = class_dir / "images"
        if image_dir.is_dir():
            result[class_dir.name] = image_dir
    if len(result) < 2:
        raise ValueError(
            "Expected class folders containing images/ subfolders, for example "
            "COVID-19_Radiography_Dataset/COVID/images/"
        )
    return result


def prepare_pilot_split(source: Path, output: Path, seed: int = 42) -> dict[str, Any]:
    """Hard-link (or copy across filesystems) images into a hash-grouped split.

    This is an image-level pilot split because the Kaggle bundle has no reliable
    patient identifiers. It must not be described as patient-level evaluation.
    """
    source = source.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a new path: {output}")
    class_dirs = _class_image_directories(source)
    rng = random.Random(seed)
    assignments: dict[str, dict[str, list[Path]]] = {split: defaultdict(list) for split in SPLIT_RATIOS}
    class_counts: dict[str, dict[str, int]] = {}
    duplicate_group_count = 0
    total = 0

    for class_name, image_dir in class_dirs.items():
        files = sorted(p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
        if not files:
            raise ValueError(f"No supported X-ray images found in {image_dir}")
        by_hash: dict[str, list[Path]] = defaultdict(list)
        for path in files:
            by_hash[_sha256(path)].append(path)
        groups = list(by_hash.values())
        if len(groups) < 3:
            raise ValueError(f"Class {class_name} has fewer than 3 unique images; cannot make all splits")
        duplicate_group_count += sum(len(group) > 1 for group in groups)
        rng.shuffle(groups)
        n = len(groups)
        n_train = max(1, int(n * SPLIT_RATIOS["train"]))
        n_val = max(1, int(n * SPLIT_RATIOS["val"]))
        if n_train + n_val >= n:
            n_train, n_val = n - 2, 1
        partitions = {
            "train": groups[:n_train],
            "val": groups[n_train:n_train + n_val],
            "test": groups[n_train + n_val:],
        }
        class_counts[class_name] = {}
        for split, split_groups in partitions.items():
            split_files = [path for group in split_groups for path in group]
            assignments[split][class_name].extend(split_files)
            class_counts[class_name][split] = len(split_files)
            total += len(split_files)

    # Reject an identical byte sequence assigned different labels.
    hash_labels: dict[str, str] = {}
    for class_name, image_dir in class_dirs.items():
        for path in image_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in EXTENSIONS:
                digest = _sha256(path)
                previous = hash_labels.setdefault(digest, class_name)
                if previous != class_name:
                    raise ValueError(
                        f"Exact duplicate image has conflicting labels: {previous} and {class_name} ({path.name})"
                    )

    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        for split, class_map in assignments.items():
            for class_name, paths in class_map.items():
                for path in paths:
                    relative = path.relative_to(class_dirs[class_name])
                    destination = temp / split / class_name / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        os.link(path, destination)
                    except OSError:
                        shutil.copy2(path, destination)
        manifest = {
            "source": str(source),
            "output": str(output),
            "seed": seed,
            "method": "class-stratified by unique SHA-256 image groups; 70/15/15 target ratios",
            "split_unit": "image hash group",
            "patient_level": False,
            "limitation": "The Kaggle bundle does not provide reliable patient identifiers. This split is exploratory and may contain patient-level leakage.",
            "duplicate_groups_kept_together": duplicate_group_count,
            "total_images": total,
            "class_split_counts": class_counts,
            "note": "Only images/ subfolders were used; masks/ were excluded.",
        }
        (temp / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        temp.rename(output)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return manifest
