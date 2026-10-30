# Quality-Aware Adaptive Preprocessing for Chest X-Ray AI Diagnosis

## Presentation website

Start the local presentation site from the project root after the expanded comparison has created its EfficientNet checkpoint:

```bash
python -m xray_quality.cli serve
```

Then open `http://127.0.0.1:8000` in a browser. The page includes the completed seven-configuration EfficientNet comparison, the quality-routing demo, and an image upload panel that runs the adaptive EfficientNet checkpoint locally. For each upload, it also compares the same model on the unenhanced image and on the selected preprocessing path, showing predicted dataset label, input-preparation time, inference time, and total time. These are single-pass illustrative timings, not benchmark results; device, system load, and warm-up state affect them. Shared file decoding and browser/network overhead are excluded. The comparison runs two inferences and can take longer than the original demo. The model returns one of the four dataset categories and its softmax scores. Those scores are uncalibrated research outputs, not clinical diagnoses or probabilities of disease. Uploaded image bytes are processed in memory by the local server and are not saved. Keep the server bound to `127.0.0.1`; do not expose it to a public network. Stop it with `Ctrl+C` after the presentation. You can load another comparison CSV from the page; repeated rows are averaged by model/arm/strategy.

Research project investigating whether selecting preprocessing from per-image quality measurements improves chest X-ray classification compared with a fixed preprocessing pipeline.

> **Research use only.** This code is not a medical device and must not be used to diagnose, triage, or guide patient care. Preprocessing can remove clinically relevant details; every strategy must be evaluated against unprocessed images and reviewed with qualified domain experts.

## Research question

Does a quality-aware adaptive pipeline improve classification performance and robustness to poor-quality images compared with conventional fixed preprocessing?

## Pipeline included

`xray_quality` measures brightness, contrast, a noise proxy, and sharpness; routes each image to a configurable preprocessing strategy; and produces a CSV quality report. The initial rule-based router is a baseline to study, not a learned or clinically validated policy.

Strategies: `standard` (grayscale, resize, normalize), `clahe`, bilateral `denoise`, `clahe_denoise`, `wiener`, and `clahe_wiener`. The `clahe_wiener` arm follows the uploaded paper's sequence: grayscale, CLAHE, then a local Wiener filter. Routing uses brightness, contrast, and estimated noise; high noise selects Wiener filtering, with CLAHE added when contrast or brightness is low. Sharpness is measured and reported, but does not trigger sharpening because that can create misleading detail. Thresholds are configurable in `configs/default.json`.

## Reference paper and reproduction limits

The uploaded paper by Gamara, Loresco, and Bandala (2022) proposes grayscale conversion → CLAHE → Wiener filtering. It compares CLAHE alone with CLAHE plus Wiener on five example images using MSE, PSNR, and entropy, then reports a separate classification comparison on 6,000 COVID-19 Radiography Database images: five training epochs, a 30% validation split, and validation accuracy of about 50% for raw images versus 78% for enhanced images. The paper does not provide enough detail to fully reproduce the classifier (for example, a named architecture, exact split procedure, random seeds, or a held-out test evaluation). Its image-metric results are based on five samples, so our project treats those measures as preliminary rather than evidence of diagnostic benefit.

This project includes a local Wiener-filter implementation as a comparable baseline, but it is not a bit-for-bit reproduction of the paper's MATLAB pipeline. Compare arms on the same saved split and training settings, and do not tune on the test set. For this Kaggle bundle, the available split is image-level because patient identifiers are unavailable.

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

The project includes `certifi` so pretrained-weight downloads can verify HTTPS certificates on Python installations that lack a configured certificate bundle. If you installed the project before this dependency was added, update the environment with `pip install certifi` before downloading pretrained weights.

## Prepare the downloaded Kaggle dataset

For the COVID-19 Radiography Database layout, point `prepare` at the folder that contains the four class folders. It reads only each class's `images/` folder and excludes `masks/`:

```text
data/row/COVID-19_Radiography_Dataset/
  COVID/images/*.png
  COVID/masks/*.png
  Lung_Opacity/images/*.png
  Normal/images/*.png
  Viral Pneumonia/images/*.png
```

Create a deterministic, class-stratified pilot split (70% train, 15% validation, 15% test):

```bash
python -m xray_quality.cli prepare data/row/COVID-19_Radiography_Dataset --output data/pilot_split --seed 42
python -m xray_quality.cli audit data/pilot_split --output reports/data_audit.json
```

The split keeps byte-identical images together and uses hard links when possible to avoid duplicating the downloaded files. **The dataset does not include reliable patient IDs, so this is an image-level exploratory pilot, not a patient-level evaluation.** The classes also come from different source datasets, which can confound the classification results. State both limitations in any report and do not claim generalization to new patients or hospitals. Do not commit patient data; check dataset terms before sharing results.

## Run the quality-aware preprocessing report

```bash
python -m xray_quality.cli analyze data/pilot_split/train --output reports/train_quality.csv
```

Each row records quality measurements, chosen strategy, and output path. Transformed images are written under `processed/` with their class subfolders preserved. Supported formats include common PNG, JPEG, and TIFF files.

## Audit the dataset

Run this before training to check the folder structure, class counts, unreadable images, dimensions, and exact duplicate images across splits:

```bash
python -m xray_quality.cli audit data/pilot_split --output reports/data_audit.json
```

The audit detects byte-identical copies only. It cannot verify patient-level separation or catch re-encoded/near-duplicate images; check patient identifiers and acquisition groups independently.

## Train and evaluate a model

The training command supports single-label class-folder datasets. It selects the best checkpoint by validation macro F1, then evaluates that checkpoint once on the held-out test split. Each run writes `metrics.json`, `predictions.csv`, and `best_model.pt`.

```bash
python -m xray_quality.cli train data/pilot_split --arm original --model efficientnet_b0 --output runs/efficientnet_original --epochs 5
python -m xray_quality.cli train data/pilot_split --arm fixed --strategy clahe_wiener --model efficientnet_b0 --output runs/efficientnet_clahe_wiener --epochs 5
python -m xray_quality.cli train data/pilot_split --arm adaptive --model efficientnet_b0 --output runs/efficientnet_adaptive --epochs 5
```

Use the same saved split, model, seed, and training settings for every arm. The supplied Kaggle data supports only the exploratory image-level pilot described above. The comparison command can compare multiple fixed strategies, and writes a separate output directory for each model, arm, strategy, and seed:

```bash
python -m xray_quality.cli compare data/pilot_split --models efficientnet_b0 densenet121 --arms original fixed adaptive --strategies clahe denoise clahe_denoise wiener clahe_wiener --seeds 42 123 --epochs 5 --batch-size 8 --output runs/expanded_comparison
```

This is a large run: with two models, two seeds, and five epochs, it trains 28 model/arm combinations (original and adaptive once each per model/seed, plus five fixed strategies). The completed EfficientNet-B0 sweep is in `runs/expanded_comparison/`; DenseNet and additional seeds remain to be run.

Each trained checkpoint is also evaluated on deterministic synthetic underexposure, Gaussian blur, and Gaussian noise variants of the test images. These are controlled stress tests, not substitutes for naturally degraded external data. The runner writes per-seed `comparison.csv` and aggregated mean/std `comparison_summary.csv`. Add `--model densenet121` to a single `train` run or select it with `--models` in `compare`. Pretrained weights are enabled by default and may require a first-time download; add `--no-pretrained` to train from random initialization. The trainer uses CUDA when available, otherwise Apple's MPS GPU on supported Macs, otherwise CPU. If memory is limited, reduce `--batch-size` (for example, `--batch-size 8`). The quality-stratified scores use test-set sharpness tertiles and are descriptive only. These commands assume mutually exclusive class labels; multi-label datasets need a different loss and metric setup.

If pretrained-weight downloading still fails with an SSL certificate error, update the environment and set a trusted CA bundle before retrying:

```bash
python -m pip install certifi
export SSL_CERT_FILE="$(python -c 'import certifi; print(certifi.where())')"
```

Keep certificate verification enabled. As an offline fallback, add `--no-pretrained`; this starts training from random weights and is a different training setup.

Create figures from the per-seed comparison file:

```bash
python -m xray_quality.cli plot runs/comparison/comparison.csv --output-dir reports/figures
```

This writes clean-set accuracy, macro F1, and ROC-AUC charts plus a macro-F1 chart for synthetic degradation conditions. Bars show means and standard deviations across seeds. Avoid interpreting small differences as evidence of improvement without suitable confidence intervals and a prespecified statistical analysis.

## Evaluation protocol

Compare the following arms on identical train/validation/test splits (patient-level whenever reliable IDs are available; the supplied Kaggle pilot split is image-level):

1. Original image with model-appropriate resizing/normalization.
2. One fixed conventional preprocessing strategy.
3. Quality-aware adaptive preprocessing.

Use identical model initialization, augmentation, optimizer, epochs, and seeds for each arm. Report accuracy, macro F1, ROC-AUC (when class probabilities and class definitions support it), confusion matrix, and results stratified by image-quality groups. Include confidence intervals or repeated seeds where feasible. Keep the test set untouched until model and routing choices are frozen. For multi-label datasets, use a multi-label setup and per-label metrics rather than treating labels as mutually exclusive.

Candidate models: EfficientNet and DenseNet. The single-seed EfficientNet-B0 sweep is available in `runs/expanded_comparison/`; DenseNet and repeated-seed comparisons remain pending. Fixed CLAHE had the best clean accuracy and macro F1 in the completed sweep, while the adaptive arm did not improve macro F1 over the original baseline. These results do not establish patient-level generalization.

## Current limitations

- Quality metrics are simple proxies; they are not validated clinical image-quality measures.
- The routing thresholds are initial configurable heuristics, not optimized parameters.
- CLAHE and denoising may suppress subtle pathology. Always include an unprocessed baseline and inspect representative transformations.
- The completed sweep includes EfficientNet-B0; DenseNet and repeated-seed comparisons are pending. There is no clinical validation, and this is a research prototype rather than a diagnostic tool.
- The downloaded dataset combines several source datasets and has no reliable patient identifiers; a random image split can overstate generalization and may capture source-specific patterns.

## Project layout

```text
configs/default.json     Routing thresholds and preprocessing options
src/xray_quality/        Quality metrics, transformations, and CLI
reports/                 Generated CSV reports (not tracked)
processed/               Generated images (not tracked)
docs/PROJECT_REPORT_TEMPLATE.md  Report outline with paper metrics and project result tables
```

The training workflow saves `run_config.json` (including software versions and split/class counts), `training_history.csv`, per-class test metrics in `metrics.json`, and `predictions.csv` with each test image's brightness, contrast, noise, and sharpness scores. These files provide the measured numbers and run settings for the report.
