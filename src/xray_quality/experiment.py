"""Reproducible single-label classification experiments for class-folder data."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models

from .pipeline import assess_quality, choose_strategy, preprocess, read_grayscale


class ChestXrayDataset(Dataset):
    """ImageFolder-like dataset; class names are discovered from train split."""

    extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

    def __init__(self, root: Path, class_to_idx: dict[str, int], arm: str,
                 fixed_strategy: str, config: dict[str, Any], corruption: str | None = None,
                 seed: int = 42, pretrained: bool = True):
        self.root = root
        self.class_to_idx = class_to_idx
        self.arm = arm
        self.fixed_strategy = fixed_strategy
        self.config = config
        self.corruption = corruption
        self.seed = seed
        self.pretrained = pretrained
        self.samples = [
            (path, class_to_idx[class_dir.name])
            for class_dir in sorted(root.iterdir()) if class_dir.is_dir()
            for path in sorted(class_dir.rglob("*"))
            if path.is_file() and path.suffix.lower() in self.extensions
        ]
        # Validate the top-level folders too.
        expected = set(class_to_idx)
        actual = {p.name for p in root.iterdir() if p.is_dir()}
        if actual != expected:
            raise ValueError(f"Class folders in {root} differ from train: {sorted(actual)} vs {sorted(expected)}")
        if not self.samples:
            raise ValueError(f"No supported images found in {root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = read_grayscale(path)
        if self.corruption == "underexposure":
            image = np.clip(image.astype(np.float32) * 0.55, 0, 255).astype(np.uint8)
        elif self.corruption == "gaussian_blur":
            image = cv2.GaussianBlur(image, (0, 0), sigmaX=1.5)
        elif self.corruption == "gaussian_noise":
            rng = np.random.default_rng(self.seed + index)
            noise = rng.normal(0.0, 20.0, image.shape)
            image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        quality = assess_quality(image)
        if self.arm == "original":
            image = cv2.resize(image, (self.config["image_size"],) * 2, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        else:
            strategy = self.fixed_strategy if self.arm == "fixed" else choose_strategy(quality, self.config["thresholds"])
            image = preprocess(image, strategy, self.config)
        # ImageNet pretrained backbones expect three channels. Repeat grayscale
        # values without colorizing the radiograph.
        tensor = torch.from_numpy(np.ascontiguousarray(image[None, ...])).repeat(3, 1, 1)
        if self.pretrained:
            mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype)[:, None, None]
            std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype)[:, None, None]
            tensor = (tensor - mean) / std
        else:
            tensor = (tensor - 0.5) / 0.5
        relative_path = (Path(self.root.name) / path.relative_to(self.root)).as_posix()
        return tensor, label, quality.brightness, quality.contrast, quality.noise, quality.sharpness, relative_path


def create_model(name: str, num_classes: int, pretrained: bool) -> nn.Module:
    # The official Python.org macOS build may have an empty/missing CA bundle.
    # PyTorch downloads weights through urllib, which honors SSL_CERT_FILE.
    # Use certifi's public CA bundle by default while respecting an explicit
    # certificate configuration supplied by the user or their organization.
    if pretrained and not os.environ.get("SSL_CERT_FILE"):
        import certifi

        os.environ["SSL_CERT_FILE"] = certifi.where()
    if name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif name == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {name}")
    return model


def _collect(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    labels, probabilities, quality_values, paths = [], [], {"brightness": [], "contrast": [], "noise": [], "sharpness": []}, []
    with torch.no_grad():
        for images, target, brightness, contrast, noise, sharpness, path in loader:
            logits = model(images.to(device))
            labels.extend(target.numpy().tolist())
            probabilities.extend(torch.softmax(logits, dim=1).cpu().numpy())
            for name, values in (("brightness", brightness), ("contrast", contrast),
                                 ("noise", noise), ("sharpness", sharpness)):
                quality_values[name].extend(values.numpy().tolist())
            paths.extend(path)
    return np.asarray(labels), np.asarray(probabilities), {name: np.asarray(values) for name, values in quality_values.items()}, paths


def _metrics(labels: np.ndarray, probabilities: np.ndarray, class_names: list[str]) -> dict[str, Any]:
    predictions = probabilities.argmax(axis=1)
    precision, recall, class_f1, support = precision_recall_fscore_support(
        labels, predictions, labels=list(range(len(class_names))), zero_division=0
    )
    result: dict[str, Any] = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=list(range(len(class_names)))).tolist(),
        "classes": class_names,
        "per_class": {
            name: {"precision": float(precision[i]), "recall": float(recall[i]),
                  "f1": float(class_f1[i]), "support": int(support[i])}
            for i, name in enumerate(class_names)
        },
    }
    try:
        if len(class_names) == 2:
            result["roc_auc"] = float(roc_auc_score(labels, probabilities[:, 1]))
        else:
            result["roc_auc"] = float(roc_auc_score(labels, probabilities, multi_class="ovr", average="macro", labels=list(range(len(class_names)))))
    except ValueError:
        result["roc_auc"] = None
    return result


def run_experiment(data_dir: Path, output_dir: Path, model_name: str, arm: str,
                   fixed_strategy: str, epochs: int, batch_size: int, learning_rate: float,
                   seed: int, pretrained: bool, config_path: Path) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    with config_path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    train_dir, val_dir, test_dir = (data_dir / split for split in ("train", "val", "test"))
    class_names = sorted(p.name for p in train_dir.iterdir() if p.is_dir())
    if len(class_names) < 2:
        raise ValueError("At least two class folders are required")
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    datasets = {
        split: ChestXrayDataset(root, class_to_idx, arm, fixed_strategy, config, pretrained=pretrained)
        for split, root in (("train", train_dir), ("val", val_dir), ("test", test_dir))
    }
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    loaders = {
        split: DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"), num_workers=0)
        for split, ds in datasets.items()
    }
    model = create_model(model_name, len(class_names), pretrained).to(device)
    print(
        f"Training {model_name} ({arm}) on {device}: "
        f"{len(datasets['train'])} train, {len(datasets['val'])} validation, "
        f"{len(datasets['test'])} test images",
        flush=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    best_f1 = -1.0
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_model.pt"
    training_history = []
    run_config = {
        "data_dir": str(data_dir), "model": model_name, "preprocessing_arm": arm,
        "fixed_strategy": fixed_strategy if arm == "fixed" else None,
        "image_size": config["image_size"], "classes": class_names,
        "class_to_index": class_to_idx,
        "split_image_counts": {split: len(dataset) for split, dataset in datasets.items()},
        "split_class_counts": {
            split: {
                class_name: sum(1 for _, label in dataset.samples if label == class_to_idx[class_name])
                for class_name in class_names
            }
            for split, dataset in datasets.items()
        },
        "epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate,
        "optimizer": "AdamW", "loss": "CrossEntropyLoss", "seed": seed,
        "pretrained_weights": pretrained, "device": str(device),
        "python_version": __import__("platform").python_version(),
        "torch_version": torch.__version__, "torchvision_version": torchvision.__version__,
        "opencv_version": cv2.__version__, "numpy_version": np.__version__,
        "preprocessing_config": config,
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_batches = len(loaders["train"])
        progress_interval = max(1, total_batches // 20)
        for batch_index, (images, labels, *_) in enumerate(loaders["train"], start=1):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            if batch_index == 1 or batch_index % progress_interval == 0 or batch_index == total_batches:
                print(
                    f"epoch {epoch + 1}/{epochs} batch {batch_index}/{total_batches} "
                    f"loss={loss.item():.4f}",
                    flush=True,
                )
        y_val, p_val, _, _ = _collect(model, loaders["val"], device)
        validation_metrics = _metrics(y_val, p_val, class_names)
        val_f1 = validation_metrics["macro_f1"]
        training_history.append({
            "epoch": epoch + 1,
            "train_loss": total_loss / len(datasets["train"]),
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_macro_f1": val_f1,
            "validation_roc_auc": validation_metrics["roc_auc"],
        })
        print(f"epoch {epoch + 1}/{epochs} train_loss={total_loss / len(datasets['train']):.4f} val_macro_f1={val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({"model": model.state_dict(), "classes": class_names, "config": config,
                        "model_name": model_name, "arm": arm, "fixed_strategy": fixed_strategy}, checkpoint_path)
    pd.DataFrame(training_history).to_csv(output_dir / "training_history.csv", index=False)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    y_test, p_test, quality_values, paths = _collect(model, loaders["test"], device)
    sharpness = quality_values["sharpness"]
    test_metrics = _metrics(y_test, p_test, class_names)
    # Test-set tertiles make the quality stratification descriptive; report the
    # cut points alongside results and avoid using these bins to tune the model.
    cut1, cut2 = np.quantile(sharpness, [1 / 3, 2 / 3])
    quality_bins = np.where(sharpness <= cut1, "low_sharpness", np.where(sharpness <= cut2, "mid_sharpness", "high_sharpness"))
    stratified = {}
    for group in ("low_sharpness", "mid_sharpness", "high_sharpness"):
        mask = quality_bins == group
        if mask.any():
            stratified[group] = {"n": int(mask.sum()), **_metrics(y_test[mask], p_test[mask], class_names)}
    corruption_results = {}
    for corruption in ("underexposure", "gaussian_blur", "gaussian_noise"):
        corrupt_dataset = ChestXrayDataset(test_dir, class_to_idx, arm, fixed_strategy,
                                           config, corruption=corruption, seed=seed,
                                           pretrained=pretrained)
        corrupt_loader = DataLoader(corrupt_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        y_corrupt, p_corrupt, _, _ = _collect(model, corrupt_loader, device)
        corruption_results[corruption] = _metrics(y_corrupt, p_corrupt, class_names)
    results = {
        "model": model_name, "preprocessing_arm": arm,
        "fixed_strategy": fixed_strategy if arm == "fixed" else None,
        "seed": seed, "device": str(device), "best_validation_macro_f1": best_f1,
        "test": test_metrics, "test_by_sharpness_tertile": stratified,
        "synthetic_corruption": corruption_results,
        "sharpness_tertile_cut_points": [float(cut1), float(cut2)],
    }
    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    prediction_rows = []
    for i, path in enumerate(paths):
        row = {"image": path, "true_label": class_names[int(y_test[i])],
               "predicted_label": class_names[int(p_test[i].argmax())]}
        row.update({name: float(values[i]) for name, values in quality_values.items()})
        row.update({f"prob_{name}": float(p_test[i, j]) for j, name in enumerate(class_names)})
        prediction_rows.append(row)
    pd.DataFrame(prediction_rows).to_csv(output_dir / "predictions.csv", index=False)
    return results


def run_comparison(data_dir: Path, output_dir: Path, models_to_run: list[str],
                   arms: list[str], fixed_strategy: str | None, epochs: int,
                   batch_size: int, learning_rate: float, seeds: list[int],
                   pretrained: bool, config_path: Path,
                   fixed_strategies: list[str] | None = None) -> pd.DataFrame:
    """Run model/arm/strategy/seed combinations and summarize test metrics."""
    rows = []
    strategies = fixed_strategies or [fixed_strategy or "clahe_wiener"]
    for model_name in models_to_run:
        for arm in arms:
            arm_strategies = strategies if arm == "fixed" else [""]
            for strategy in arm_strategies:
                for seed in seeds:
                    run_dir = output_dir / model_name / arm
                    if strategy:
                        run_dir = run_dir / strategy
                    run_dir = run_dir / f"seed_{seed}"
                    result = run_experiment(
                        data_dir=data_dir, output_dir=run_dir, model_name=model_name,
                        arm=arm, fixed_strategy=strategy or (fixed_strategy or "clahe_wiener"), epochs=epochs,
                        batch_size=batch_size, learning_rate=learning_rate, seed=seed,
                        pretrained=pretrained, config_path=config_path,
                    )
                    row = {
                        "model": model_name, "arm": arm,
                        "fixed_strategy": strategy,
                        "seed": seed,
                        "accuracy": result["test"]["accuracy"],
                        "macro_f1": result["test"]["macro_f1"],
                        "roc_auc": result["test"]["roc_auc"],
                    }
                    for corruption, metrics in result["synthetic_corruption"].items():
                        row[f"{corruption}_accuracy"] = metrics["accuracy"]
                        row[f"{corruption}_macro_f1"] = metrics["macro_f1"]
                    rows.append(row)
    summary = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "comparison.csv", index=False)
    metrics = [column for column in summary.columns if column not in {"model", "arm", "fixed_strategy", "seed"}]
    aggregate = summary.groupby(["model", "arm", "fixed_strategy"], dropna=False)[metrics].agg(["mean", "std"])
    aggregate.columns = [f"{metric}_{stat}" for metric, stat in aggregate.columns]
    aggregate.reset_index().to_csv(output_dir / "comparison_summary.csv", index=False)
    return summary
