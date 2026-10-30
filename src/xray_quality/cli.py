"""Command line entry point for quality analysis and adaptive preprocessing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import pandas as pd

from .pipeline import analyze_file

EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze = subparsers.add_parser("analyze", help="score and adapt a directory of images")
    analyze.add_argument("input", type=Path)
    analyze.add_argument("--output", type=Path, default=Path("reports/quality.csv"))
    analyze.add_argument("--processed-dir", type=Path, default=Path("processed"))
    analyze.add_argument("--config", type=Path, default=Path("configs/default.json"))
    train = subparsers.add_parser("train", help="train one classification experiment and evaluate its held-out test split")
    train.add_argument("data", type=Path, help="directory containing train/, val/, and test/ class folders")
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--model", choices=["efficientnet_b0", "densenet121"], default="efficientnet_b0")
    train.add_argument("--arm", choices=["original", "fixed", "adaptive"], required=True)
    train.add_argument("--strategy", choices=["standard", "clahe", "denoise", "clahe_denoise", "wiener", "clahe_wiener"], default="clahe_wiener")
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--learning-rate", type=float, default=0.0001)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--config", type=Path, default=Path("configs/default.json"))
    train.add_argument("--no-pretrained", action="store_true", help="initialize model weights randomly")
    compare = subparsers.add_parser("compare", help="run model/preprocessing comparisons and synthetic robustness checks")
    compare.add_argument("data", type=Path)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--models", nargs="+", choices=["efficientnet_b0", "densenet121"], default=["efficientnet_b0"])
    compare.add_argument("--arms", nargs="+", choices=["original", "fixed", "adaptive"], default=["original", "fixed", "adaptive"])
    compare.add_argument("--strategy", choices=["standard", "clahe", "denoise", "clahe_denoise", "wiener", "clahe_wiener"], default=None, help="one fixed strategy (legacy option; use --strategies to compare several)")
    compare.add_argument("--strategies", nargs="+", choices=["standard", "clahe", "denoise", "clahe_denoise", "wiener", "clahe_wiener"], help="fixed preprocessing strategies to compare")
    compare.add_argument("--seeds", nargs="+", type=int, default=[42])
    compare.add_argument("--epochs", type=int, default=10)
    compare.add_argument("--batch-size", type=int, default=16)
    compare.add_argument("--learning-rate", type=float, default=0.0001)
    compare.add_argument("--config", type=Path, default=Path("configs/default.json"))
    compare.add_argument("--no-pretrained", action="store_true")
    audit = subparsers.add_parser("audit", help="audit dataset structure and exact duplicate leakage")
    audit.add_argument("data", type=Path, help="directory containing train/, val/, and test/")
    audit.add_argument("--output", type=Path, default=Path("reports/data_audit.json"))
    prepare = subparsers.add_parser("prepare", help="make a reproducible pilot split from class/images/ folders")
    prepare.add_argument("source", type=Path, help="Kaggle dataset folder containing class/images/ subfolders")
    prepare.add_argument("--output", type=Path, default=Path("data/pilot_split"))
    prepare.add_argument("--seed", type=int, default=42)
    plot = subparsers.add_parser("plot", help="plot clean and robustness results from comparison.csv")
    plot.add_argument("comparison_csv", type=Path)
    plot.add_argument("--output-dir", type=Path, default=Path("reports/figures"))
    web = subparsers.add_parser("serve", help="start the local presentation website and image prediction demo")
    web.add_argument("--checkpoint", type=Path, default=Path("runs/expanded_comparison/efficientnet_b0/adaptive/seed_42/best_model.pt"))
    web.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.command == "serve":
        from .serve import serve

        serve(args.checkpoint, port=args.port)
        return

    if args.command in {"train", "compare"}:
        from .experiment import run_comparison, run_experiment

        if args.command == "train":
            run_experiment(
                data_dir=args.data, output_dir=args.output, model_name=args.model,
                arm=args.arm, fixed_strategy=args.strategy, epochs=args.epochs,
                batch_size=args.batch_size, learning_rate=args.learning_rate,
                seed=args.seed, pretrained=not args.no_pretrained, config_path=args.config,
            )
            print(f"Metrics and predictions saved under {args.output}")
        else:
            run_comparison(
                data_dir=args.data, output_dir=args.output, models_to_run=args.models,
                arms=args.arms, fixed_strategy=args.strategy, fixed_strategies=args.strategies, epochs=args.epochs,
                batch_size=args.batch_size, learning_rate=args.learning_rate,
                seeds=args.seeds, pretrained=not args.no_pretrained, config_path=args.config,
            )
            print(f"Comparison summary saved to {args.output / 'comparison.csv'}")
        return

    if args.command == "audit":
        from .audit import write_audit

        result = write_audit(args.data, args.output)
        print(f"Audit report saved to {args.output}")
        print(f"Total supported images: {result['total_image_count']}")
        print(f"Cross-split exact duplicate groups: {len(result['cross_split_exact_duplicate_groups'])}")
        print(f"Unreadable images: {len(result['unreadable_images'])}")
        if result["total_image_count"] == 0:
            raise SystemExit("No supported images found. Add PNG, JPG, JPEG, TIFF, or BMP files under the class folders before training.")
        return

    if args.command == "prepare":
        from .prepare import prepare_pilot_split

        manifest = prepare_pilot_split(args.source, args.output, args.seed)
        print(f"Prepared {manifest['total_images']} images in {args.output}")
        print("Split counts by class:")
        for class_name, counts in manifest["class_split_counts"].items():
            print(f"  {class_name}: " + ", ".join(f"{split}={count}" for split, count in counts.items()))
        print("Note: this is an image-level pilot split; patient identifiers are unavailable.")
        return

    if args.command == "plot":
        from .plots import create_comparison_plots

        paths = create_comparison_plots(args.comparison_csv, args.output_dir)
        for path in paths:
            print(f"Saved plot: {path}")
        return

    if not args.input.is_dir():
        parser.error(f"Input directory does not exist: {args.input}")
    with args.config.open(encoding="utf-8") as stream:
        config = json.load(stream)

    files = sorted(p for p in args.input.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not files:
        parser.error(f"No supported images found under {args.input}")

    rows = []
    for path in files:
        try:
            quality, strategy, image = analyze_file(path, config)
            destination = args.processed_dir / path.relative_to(args.input).with_suffix(".png")
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(destination), (image * 255).round().astype("uint8")):
                raise OSError(f"Could not write output: {destination}")
            rows.append({"image": str(path), **quality.as_dict(), "strategy": strategy, "output": str(destination)})
        except (ValueError, OSError) as error:
            rows.append({"image": str(path), "error": str(error)})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Processed {len(files)} images; report: {args.output}")


if __name__ == "__main__":
    main()
