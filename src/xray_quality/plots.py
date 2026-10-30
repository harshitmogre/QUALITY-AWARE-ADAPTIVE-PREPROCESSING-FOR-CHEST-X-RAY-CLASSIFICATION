"""Publication-oriented plots from the comparison runner's per-seed CSV."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def create_comparison_plots(csv_path: Path, output_dir: Path) -> list[Path]:
    data = pd.read_csv(csv_path)
    required = {"model", "arm", "accuracy", "macro_f1", "roc_auc"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Comparison CSV is missing required columns: {sorted(missing)}")
    if data.empty:
        raise ValueError("Comparison CSV contains no experiment rows")
    output_dir.mkdir(parents=True, exist_ok=True)
    plots: list[Path] = []
    keys = ["model", "arm", "fixed_strategy"]
    labels = data[keys].fillna("").astype(str).agg(" / ".join, axis=1)
    groups = data.assign(label=labels).groupby("label", sort=False)

    metrics = [column for column in ("accuracy", "macro_f1", "roc_auc") if data[column].notna().any()]
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5), squeeze=False)
    for axis, metric in zip(axes[0], metrics):
        means = groups[metric].mean()
        stds = groups[metric].std().fillna(0)
        axis.bar(np.arange(len(means)), means.values, yerr=stds.values, capsize=4, color="#3676a8")
        axis.set_title(metric.replace("_", " ").title())
        axis.set_ylabel("Score")
        axis.set_ylim(0, 1.05)
        axis.set_xticks(np.arange(len(means)), means.index, rotation=35, ha="right")
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Held-out test performance (mean ± SD across seeds)")
    fig.tight_layout()
    clean_path = output_dir / "clean_performance.png"
    fig.savefig(clean_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    plots.append(clean_path)

    robustness = [
        ("macro_f1", "Clean"),
        ("underexposure_macro_f1", "Underexposure"),
        ("gaussian_blur_macro_f1", "Blur"),
        ("gaussian_noise_macro_f1", "Noise"),
    ]
    available = [(column, label) for column, label in robustness if column in data and data[column].notna().any()]
    if len(available) > 1:
        group_labels = list(dict.fromkeys(labels.tolist()))
        x = np.arange(len(group_labels))
        width = 0.8 / len(available)
        fig, axis = plt.subplots(figsize=(max(9, len(group_labels) * 1.5), 5))
        for idx, (column, label) in enumerate(available):
            means = data.assign(label=labels).groupby("label")[column].mean().reindex(group_labels)
            stds = data.assign(label=labels).groupby("label")[column].std().fillna(0).reindex(group_labels)
            axis.bar(x + (idx - (len(available) - 1) / 2) * width, means.values,
                     width=width, yerr=stds.values, capsize=3, label=label)
        axis.set_title("Macro F1 under synthetic image degradation (mean ± SD)")
        axis.set_ylabel("Macro F1")
        axis.set_ylim(0, 1.05)
        axis.set_xticks(x, group_labels, rotation=35, ha="right")
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
        fig.tight_layout()
        robustness_path = output_dir / "robustness_macro_f1.png"
        fig.savefig(robustness_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        plots.append(robustness_path)
    return plots
