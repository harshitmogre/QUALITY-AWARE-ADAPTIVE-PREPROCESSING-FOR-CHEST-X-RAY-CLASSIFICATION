"""Image quality proxies and configurable adaptive preprocessing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.signal import wiener


@dataclass(frozen=True)
class Quality:
    brightness: float
    contrast: float
    noise: float
    sharpness: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def read_grayscale(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def assess_quality(image: np.ndarray) -> Quality:
    """Return simple image-quality proxies; these are not clinical measures."""
    if image.ndim != 2 or image.size == 0:
        raise ValueError("Expected a non-empty grayscale image")
    work = image.astype(np.float32)
    brightness = float(work.mean())
    contrast = float(work.std())
    # MAD of a high-pass residual provides a rough noise proxy. Fine anatomy also
    # contributes, so interpret it comparatively rather than as true noise.
    smooth = cv2.GaussianBlur(work, (0, 0), sigmaX=1.0)
    residual = work - smooth
    noise = float(np.median(np.abs(residual - np.median(residual))) * 1.4826)
    sharpness = float(cv2.Laplacian(work, cv2.CV_32F).var())
    return Quality(brightness, contrast, noise, sharpness)


def choose_strategy(q: Quality, thresholds: dict[str, float]) -> str:
    """Map quality proxies to a transparent, deterministic baseline strategy."""
    low_brightness = q.brightness < thresholds["low_brightness"]
    high_brightness = q.brightness > thresholds["high_brightness"]
    low_contrast = q.contrast < thresholds["low_contrast"]
    noisy = q.noise > thresholds["high_noise"]
    # CLAHE is selected for contrast/brightness issues; denoising only for the
    # noise proxy. Blur is reported, not aggressively sharpened, to avoid
    # inventing high-frequency detail.
    needs_clahe = low_contrast or low_brightness or high_brightness
    if needs_clahe and noisy:
        return "clahe_wiener"
    if needs_clahe:
        return "clahe"
    if noisy:
        return "wiener"
    return "standard"


def preprocess(image: np.ndarray, strategy: str, config: dict[str, Any]) -> np.ndarray:
    """Apply one named strategy and return float32 grayscale pixels in [0, 1]."""
    if image.ndim != 2 or image.size == 0:
        raise ValueError("Expected a non-empty grayscale image")
    if strategy not in {"standard", "clahe", "denoise", "clahe_denoise", "wiener", "clahe_wiener"}:
        raise ValueError(f"Unknown preprocessing strategy: {strategy}")
    result = image.copy()
    if strategy in {"denoise", "clahe_denoise"}:
        result = cv2.bilateralFilter(
            result,
            int(config["bilateral_diameter"]),
            float(config["bilateral_sigma_color"]),
            float(config["bilateral_sigma_space"]),
        )
    if strategy in {"clahe", "clahe_denoise", "clahe_wiener"}:
        clahe = cv2.createCLAHE(
            clipLimit=float(config["clahe_clip_limit"]),
            tileGridSize=(int(config["clahe_tile_grid_size"]),) * 2,
        )
        result = clahe.apply(result)
    if strategy in {"wiener", "clahe_wiener"}:
        window = int(config["wiener_window_size"])
        filtered = wiener(result.astype(np.float32), mysize=(window, window))
        result = np.clip(filtered, 0, 255).astype(np.uint8)
    size = int(config["image_size"])
    result = cv2.resize(result, (size, size), interpolation=cv2.INTER_AREA)
    return result.astype(np.float32) / 255.0


def analyze_file(path: str | Path, config: dict[str, Any]) -> tuple[Quality, str, np.ndarray]:
    image = read_grayscale(path)
    quality = assess_quality(image)
    strategy = choose_strategy(quality, config["thresholds"])
    processed = preprocess(image, strategy, config)
    return quality, strategy, processed
