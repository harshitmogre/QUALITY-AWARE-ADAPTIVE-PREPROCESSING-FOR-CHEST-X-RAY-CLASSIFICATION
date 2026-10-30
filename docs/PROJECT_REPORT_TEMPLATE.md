# Quality-Aware Adaptive Preprocessing for Chest X-Ray AI Diagnosis

**Author(s):** [Names]  
**Course / institution:** [Details]  
**Submission date:** [Date]

> Replace bracketed fields only with verified project information. Do not present paper results as results of this project. Do not claim that enhancement improves diagnosis unless your held-out experiments support that claim.

> **Results status:** The numbers below include the completed single-seed EfficientNet-B0 comparison across seven configurations (Original, five fixed strategies, and Adaptive). DenseNet-121 and repeated-seed comparisons remain pending; this is not the final model comparison.

## Abstract

This project evaluated whether per-image quality-aware preprocessing improves chest X-ray classification compared with original images and five fixed enhancement strategies. A pretrained EfficientNet-B0 was evaluated on the four-class COVID-19 Radiography Database using a seed-42, image-level 70/15/15 split (21,165 images total; 3,174 test images). Fixed CLAHE achieved the strongest clean-test accuracy (95.43%) and macro F1 (0.9576); adaptive preprocessing reached 95.12% accuracy and 0.9545 macro F1, while the original arm reached 95.05% and 0.9547. The strongest synthetic underexposure score came from fixed CLAHE (93.67%); CLAHE + Wiener scored highest under synthetic blur (92.03%); Wiener filtering scored highest under synthetic Gaussian noise (20.29%), although all noise scores were poor. This one-seed, image-level comparison does not establish patient-level or hospital-level generalization.

## 1. Introduction

### 1.1 Background

Chest X-ray images can differ in brightness, contrast, noise, and sharpness because of acquisition and source variation. Applying the same enhancement to every image may help some images while degrading others. Gamara et al. (2022) studied CLAHE followed by Wiener filtering and reported higher validation accuracy for enhanced than raw images in their experiment; however, the paper does not report a held-out test evaluation or enough model and split detail for exact replication. This project therefore evaluates both a fixed CLAHE + Wiener pipeline and a transparent per-image quality-based routing rule against an unenhanced baseline.

### 1.2 Aim and research question

**Aim:** Evaluate whether a quality-aware adaptive preprocessing pipeline improves chest X-ray classification performance and robustness compared with fixed preprocessing.

**Research question:** Does per-image quality-aware adaptive preprocessing improve clean-set classification performance or robustness to controlled image degradation compared with original images and fixed preprocessing strategies?

**Exploratory expectation:** Adaptive preprocessing may improve macro F1 and performance under selected image degradations by applying enhancement only when quality proxies indicate it is needed. This expectation was not preregistered, so findings are treated as exploratory.

## 2. Related work

### 2.1 CLAHE and Wiener filtering paper

Gamara, Loresco, and Bandala propose grayscale conversion followed by CLAHE and Wiener filtering. They compare CLAHE alone with CLAHE plus Wiener using five example images and report MSE, PSNR, and entropy. For classification, they report results on 6,000 images from the COVID-19 Radiography Database, five training epochs, a 30% validation split, and about 50% validation accuracy on raw images versus 78% on enhanced images. The paper does not name a specific network architecture or report a held-out test evaluation, so these results are a reference point rather than a directly reproducible benchmark.

**Image-enhancement results reported in the paper:**

| Sample | CLAHE MSE | CLAHE PSNR (dB) | CLAHE entropy | CLAHE + Wiener MSE | CLAHE + Wiener PSNR (dB) | CLAHE + Wiener entropy |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 961.856 | 18.300 | 7.281 | 921.992 | 18.484 | 7.287 |
| 2 | 1114.710 | 17.659 | 7.585 | 1059.738 | 17.879 | 7.601 |
| 3 | 1063.859 | 17.862 | 7.666 | 1026.354 | 18.018 | 7.671 |
| 4 | 1112.880 | 17.666 | 7.829 | 1080.030 | 17.796 | 7.866 |
| 5 | 1197.072 | 17.350 | 7.562 | 1155.016 | 17.505 | 7.565 |

These are the paper's five-sample image-metric results; they are not measurements from this project. The paper treats lower MSE and higher PSNR/entropy as better, but these image metrics alone do not establish clinical relevance or preservation of pathology.

### 2.2 Project contribution

The project measures brightness, contrast, a high-pass-residual noise proxy, and sharpness for each image. Its deterministic router selects standard preprocessing, CLAHE, Wiener filtering, or CLAHE + Wiener using configured brightness, contrast, and noise thresholds. The available fixed strategies are standard, CLAHE, bilateral denoising, CLAHE + bilateral denoising, Wiener, and CLAHE + Wiener. Sharpness is measured and used for descriptive stratification; it does not trigger sharpening because blur cannot reliably be reversed without risking invented detail. The thresholds are heuristic and have not been clinically validated.

## 3. Materials and methods

### 3.1 Dataset

| Item | Project detail |
|---|---|
| Dataset name and source | COVID-19 Radiography Database, [Kaggle dataset page](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) |
| Version / access date | [Add the version and date you downloaded it, if recorded] |
| Inclusion / exclusion | Used supported X-ray images from each class's `images/` folder; excluded segmentation masks. |
| Label classes | COVID, Lung_Opacity, Normal, Viral Pneumonia |
| Total images | 21,165 |
| Patient count | Not available in this dataset bundle |
| Train / validation / test counts | 14,815 / 3,176 / 3,174 |
| Split unit and method | Seed-42 class-stratified 70/15/15 image-level split; byte-identical images were kept together by SHA-256 hash. This is not patient-level. |
| Duplicate / unreadable-image audit | 0 exact duplicate groups across splits; 0 unreadable images (`reports/data_audit.json`). |

### 3.2 Image quality measurements and adaptive rule

Brightness is the grayscale pixel mean (0–255); contrast is the grayscale standard deviation; noise is a robust MAD-based estimate of the high-pass residual; sharpness is the variance of the Laplacian. These are image statistics, not validated clinical quality scores. The fixed `configs/default.json` thresholds were low brightness < 70, high brightness > 190, low contrast < 25, and high noise > 18. Images with a brightness/contrast flag and high noise received CLAHE + Wiener; with only a brightness/contrast flag they received CLAHE; with only high noise they received Wiener; otherwise they received standard preprocessing. Sharpness was measured but did not affect routing. Thresholds were treated as heuristics and were not tuned on test outcomes.

### 3.3 Experiment arms

| Arm | Preprocessing |
|---|---|
| Original | Grayscale conversion, resize to 224 × 224, and ImageNet model normalization only |
| Fixed preprocessing | Five separate fixed arms: CLAHE, bilateral denoising, CLAHE + bilateral denoising, Wiener filtering, and CLAHE + Wiener filtering. Parameters are in `configs/default.json`. |
| Adaptive | Rule-based routing from brightness, contrast, and noise thresholds in `configs/default.json`; the selected operation was standard, CLAHE, Wiener, or CLAHE + Wiener. Sharpness was measured but did not trigger sharpening. |

The completed single-seed comparison used all seven listed configurations with EfficientNet-B0. DenseNet-121 and repeated seeds remain pending.

### 3.4 Models and training settings

The completed comparison used EfficientNet-B0 with pretrained weights, 224 × 224 input, cross-entropy loss, AdamW (learning rate 0.0001), batch size 8, five epochs, seed 42, and no data augmentation. The checkpoint with the highest validation macro F1 was evaluated on the test split. Training ran on Apple MPS. Recorded versions: Python 3.13.7, PyTorch 2.14.1, Torchvision 0.29.1, OpenCV 5.0.0, and NumPy 2.5.4. See each configuration's `run_config.json` and `training_history.csv` under `runs/expanded_comparison/efficientnet_b0/`. DenseNet and repeated-seed experiments remain pending.

### 3.5 Evaluation metrics and statistical analysis

The ROC-AUC is macro one-vs-rest. This experiment used one seed, so it has no across-seed standard deviation or inferential confidence interval; treat comparisons as descriptive. Sharpness-stratified and synthetic-corruption results are reported separately from naturally degraded or external test data.

### 3.6 Interactive per-image timing display

The presentation webpage compares one uploaded image through the same model with no enhancement and with the selected preprocessing path. It displays input-preparation time, model-inference time, total time, and each path's top dataset label. File decoding is shared and excluded from both path measurements; adaptive preparation includes quality analysis and route selection. These single-pass values are illustrative and can vary with hardware, system load, and warm-up state. They are not the project's benchmark evidence. A defensible latency result would time both paths over the same held-out test set with warm-up, repeated runs, and reported device/software settings.

## 4. Results

### 4.1 Dataset audit

The audit found 21,165 readable images, no empty class folders, no class mismatches, no unreadable files, and no exact byte-identical groups crossing splits. The class counts were COVID 3,616; Lung_Opacity 6,012; Normal 10,192; and Viral Pneumonia 1,345. Test support was 3,174 images: COVID 539, Lung_Opacity 903, Normal 1,530, and Viral Pneumonia 202. Exact duplicates were grouped before splitting. The audit cannot detect re-encoded near-duplicates or patient overlap.

### 4.2 Main held-out test results

| Model | Preprocessing arm | Seed(s) | Test n | Accuracy | Macro F1 | ROC-AUC | 95% CI / SD |
|---|---|---|---:|---:|---:|---:|---|
| EfficientNet-B0 | Original | 42 | 3,174 | 95.05% | 0.9547 | 0.9941 | One seed; no CI |
| EfficientNet-B0 | Fixed CLAHE | 42 | 3,174 | **95.43%** | **0.9576** | 0.9940 | One seed; no CI |
| EfficientNet-B0 | Fixed bilateral denoising | 42 | 3,174 | 95.02% | 0.9565 | 0.9940 | One seed; no CI |
| EfficientNet-B0 | Fixed CLAHE + bilateral denoising | 42 | 3,174 | 94.96% | 0.9558 | 0.9939 | One seed; no CI |
| EfficientNet-B0 | Fixed Wiener | 42 | 3,174 | 94.74% | 0.9518 | 0.9937 | One seed; no CI |
| EfficientNet-B0 | Fixed CLAHE + Wiener | 42 | 3,174 | 94.27% | 0.9484 | 0.9923 | One seed; no CI |
| EfficientNet-B0 | Quality-aware adaptive | 42 | 3,174 | 95.12% | 0.9545 | 0.9936 | One seed; no CI |

These are the measured project test results, not results from the reference paper. Fixed CLAHE had the highest accuracy and macro F1 among the seven configurations. Adaptive was 0.06 percentage points higher in accuracy than Original, but its macro F1 was 0.0002 lower. These small single-seed differences are descriptive and do not establish a reliable improvement.

The following confusion matrices use row and column order **COVID, Lung_Opacity, Normal, Viral Pneumonia** (rows are true labels; columns are predictions):

| Arm | Confusion matrix |
|---|---|
| Original | `[[528, 2, 5, 4], [6, 801, 96, 0], [3, 29, 1494, 4], [0, 0, 8, 194]]` |
| Fixed CLAHE | `[[530, 3, 2, 4], [4, 832, 67, 0], [5, 47, 1472, 6], [0, 0, 7, 195]]` |
| Fixed bilateral denoising | `[[531, 2, 5, 1], [5, 836, 62, 0], [3, 67, 1452, 8], [1, 0, 4, 197]]` |
| Fixed CLAHE + bilateral denoising | `[[526, 2, 9, 2], [6, 804, 93, 0], [4, 34, 1490, 2], [0, 0, 8, 194]]` |
| Fixed Wiener | `[[517, 6, 13, 3], [6, 804, 93, 0], [1, 33, 1494, 2], [1, 0, 9, 192]]` |
| Fixed CLAHE + Wiener | `[[527, 4, 5, 3], [5, 825, 73, 0], [5, 72, 1450, 3], [0, 0, 12, 190]]` |
| Adaptive | `[[531, 2, 3, 3], [14, 830, 59, 0], [3, 58, 1465, 4], [1, 0, 8, 193]]` |

Charts generated from this run are [clean test performance](../reports/figures/clean_performance.png) and [synthetic robustness macro F1](../reports/figures/robustness_macro_f1.png).

### 4.3 Results by image quality

The test-set sharpness tertiles each contained 1,058 images. Accuracy by low / mid / high sharpness was:

| Configuration | Low sharpness | Mid sharpness | High sharpness |
|---|---:|---:|---:|
| Original | 96.12% | 94.99% | 94.05% |
| Fixed CLAHE | 96.50% | 95.84% | 93.95% |
| Fixed bilateral denoising | 96.22% | 94.61% | 94.23% |
| Fixed CLAHE + bilateral denoising | 96.12% | 94.90% | 93.86% |
| Fixed Wiener | 95.46% | 94.90% | 93.86% |
| Fixed CLAHE + Wiener | 95.56% | 94.71% | 92.53% |
| Adaptive | 96.12% | 95.37% | 93.86% |

These groups are descriptive and were not used to tune the models.

### 4.4 Robustness results

| Model | Arm | Test condition | Accuracy | Macro F1 |
|---|---|---|---:|---:|
| EfficientNet-B0 | Original | Synthetic underexposure | 82.67% | 0.8204 |
| EfficientNet-B0 | Fixed CLAHE | Synthetic underexposure | **93.67%** | **0.9415** |
| EfficientNet-B0 | Fixed bilateral denoising | Synthetic underexposure | 90.96% | 0.8957 |
| EfficientNet-B0 | Fixed CLAHE + bilateral denoising | Synthetic underexposure | 92.97% | 0.9323 |
| EfficientNet-B0 | Fixed Wiener | Synthetic underexposure | 84.09% | 0.8193 |
| EfficientNet-B0 | Fixed CLAHE + Wiener | Synthetic underexposure | 88.82% | 0.8795 |
| EfficientNet-B0 | Adaptive | Synthetic underexposure | 90.42% | 0.9020 |
| EfficientNet-B0 | Original | Synthetic Gaussian blur | 32.70% | 0.4195 |
| EfficientNet-B0 | Fixed CLAHE | Synthetic Gaussian blur | 20.60% | 0.2457 |
| EfficientNet-B0 | Fixed bilateral denoising | Synthetic Gaussian blur | 88.41% | 0.8709 |
| EfficientNet-B0 | Fixed CLAHE + bilateral denoising | Synthetic Gaussian blur | 91.27% | 0.8989 |
| EfficientNet-B0 | Fixed Wiener | Synthetic Gaussian blur | 86.74% | 0.8473 |
| EfficientNet-B0 | Fixed CLAHE + Wiener | Synthetic Gaussian blur | **92.03%** | **0.9217** |
| EfficientNet-B0 | Adaptive | Synthetic Gaussian blur | 22.97% | 0.2740 |
| EfficientNet-B0 | Original | Synthetic Gaussian noise | 12.76% | 0.1049 |
| EfficientNet-B0 | Fixed CLAHE | Synthetic Gaussian noise | 7.34% | 0.0575 |
| EfficientNet-B0 | Fixed bilateral denoising | Synthetic Gaussian noise | 17.39% | 0.1013 |
| EfficientNet-B0 | Fixed CLAHE + bilateral denoising | Synthetic Gaussian noise | 17.14% | 0.0848 |
| EfficientNet-B0 | Fixed Wiener | Synthetic Gaussian noise | **20.29%** | **0.2387** |
| EfficientNet-B0 | Fixed CLAHE + Wiener | Synthetic Gaussian noise | 19.25% | 0.1854 |
| EfficientNet-B0 | Adaptive | Synthetic Gaussian noise | 17.23% | 0.1393 |

Synthetic conditions were generated by the experiment code (underexposure multiplier 0.55, Gaussian blur σ = 1.5, and Gaussian noise standard deviation 20 on 8-bit pixels). These controlled stress tests are not external validation. Fixed CLAHE scored highest under underexposure, fixed CLAHE + Wiener scored highest under blur, and fixed Wiener scored highest under Gaussian noise. The noise accuracies remain poor (7.34%–20.29%), so the highest result should not be interpreted as robust performance.

## 5. Discussion

On this split, fixed CLAHE had the highest clean-test accuracy (95.43%) and macro F1 (0.9576) among the seven configurations. Adaptive preprocessing did not improve macro F1 over Original (0.9545 vs. 0.9547), and its 0.06 percentage point accuracy difference is too small to support a meaningful claim from one seed. The best fixed strategy depended on the synthetic degradation: CLAHE for underexposure, CLAHE + Wiener for blur, and Wiener for Gaussian noise. All approaches performed poorly on synthetic noise. These results are from one seed and an image-level split. Source composition and possible patient overlap limit interpretation; repeated seeds and patient-level or external validation would be needed for stronger conclusions.

## 6. Limitations and ethics

This dataset is not a patient-level evaluation set: reliable patient identifiers are unavailable, so images from the same patient may cross splits. The classes also draw from different source collections, allowing the model to learn source-specific visual cues. Labels and source composition may be heterogeneous, and the test set comes from the same downloaded collection rather than an independent hospital. Results use one random seed with no confidence intervals or repeated-seed analysis. The quality measures are simple image statistics, and routing thresholds were not clinically validated. CLAHE and denoising may alter diagnostic details. Synthetic underexposure, blur, and noise are controlled perturbations and may not represent naturally degraded images. This project has no clinical validation and must not be used for diagnosis or patient care.

## 7. Conclusion

In this single-seed image-level EfficientNet-B0 comparison, fixed CLAHE achieved the strongest clean-test accuracy and macro F1, while adaptive preprocessing did not improve macro F1 over the original arm. The strongest preprocessing choice varied across synthetic degradation types, and all configurations performed poorly under synthetic Gaussian noise. The findings are preliminary and do not establish patient-level or clinical benefit. DenseNet-121 and repeated-seed comparisons remain necessary before drawing a broader conclusion.

## References

Gamara, R. P. C., Loresco, P. J. M., & Bandala, A. A. (2022). Medical chest X-ray image enhancement based on CLAHE and Wiener filter for deep learning data preprocessing. *2022 IEEE 14th International Conference on Humanoid, Nanotechnology, Information Technology, Communication and Control, Environment, and Management (HNICEM)*, 1–6. https://doi.org/10.1109/HNICEM57413.2022.10109585
