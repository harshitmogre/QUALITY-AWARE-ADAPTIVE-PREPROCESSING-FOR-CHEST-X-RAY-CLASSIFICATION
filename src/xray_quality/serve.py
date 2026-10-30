"""Local presentation website with single-image research-model inference."""

from __future__ import annotations

import base64
import json
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import cv2
import numpy as np
import torch

from .experiment import create_model
from .pipeline import assess_quality, preprocess

WEB_DIR = Path(__file__).resolve().parents[2] / "web"
MAX_UPLOAD_BYTES = 12 * 1024 * 1024


class Predictor:
    def __init__(self, checkpoint_path: Path):
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else
            "cpu"
        )
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.classes = checkpoint["classes"]
        self.config = checkpoint["config"]
        self.arm = checkpoint["arm"]
        self.fixed_strategy = checkpoint.get("fixed_strategy") or "standard"
        self.model_name = checkpoint["model_name"]
        self.model = create_model(self.model_name, len(self.classes), pretrained=False).to(self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            torch.mps.synchronize()

    def _prepare_tensor(self, image: np.ndarray, strategy: str) -> torch.Tensor:
        if strategy == "standard":
            image = cv2.resize(
                image, (self.config["image_size"],) * 2, interpolation=cv2.INTER_AREA
            ).astype(np.float32) / 255.0
        else:
            image = preprocess(image, strategy, self.config)
        tensor = torch.from_numpy(np.ascontiguousarray(image[None, ...])).repeat(3, 1, 1)
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype)[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype)[:, None, None]
        return ((tensor - mean) / std).unsqueeze(0).to(self.device)

    @torch.inference_mode()
    def _run_path(self, image: np.ndarray, strategy: str) -> tuple[list[float], dict[str, float]]:
        preparation_start = time.perf_counter()
        tensor = self._prepare_tensor(image, strategy)
        preparation_ms = (time.perf_counter() - preparation_start) * 1000

        self._synchronize()
        inference_start = time.perf_counter()
        scores = torch.softmax(self.model(tensor), dim=1)[0].cpu().tolist()
        self._synchronize()
        inference_ms = (time.perf_counter() - inference_start) * 1000
        return scores, {
            "preparation_ms": preparation_ms,
            "inference_ms": inference_ms,
            "total_ms": preparation_ms + inference_ms,
        }

    @torch.inference_mode()
    def predict(self, image_bytes: bytes) -> dict[str, Any]:
        encoded = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
        if image is None or image.size == 0:
            raise ValueError("The selected file is not a readable supported image.")
        quality_start = time.perf_counter()
        quality = assess_quality(image)
        strategy = self.fixed_strategy if self.arm == "fixed" else "standard"
        if self.arm == "adaptive":
            from .pipeline import choose_strategy
            strategy = choose_strategy(quality, self.config["thresholds"])
        decision_ms = (time.perf_counter() - quality_start) * 1000
        # Decode once, then compare the same classifier on identical pixels.
        # Shared file decoding is excluded from both path timings.
        raw_scores, raw_timing = self._run_path(image, "standard")
        processed_scores, processed_timing = self._run_path(image, strategy)
        if self.arm == "adaptive":
            processed_timing["preparation_ms"] += decision_ms
            processed_timing["total_ms"] += decision_ms
        scores = processed_scores
        ranking = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
        raw_ranking = sorted(range(len(raw_scores)), key=lambda index: raw_scores[index], reverse=True)
        return {
            "model": self.model_name,
            "preprocessing": strategy,
            "top_label": self.classes[ranking[0]],
            "top_score": float(scores[ranking[0]]),
            "scores": [{"label": self.classes[index], "score": float(scores[index])} for index in ranking],
            "timing_comparison": {
                "device": str(self.device),
                "raw": {
                    "top_label": self.classes[raw_ranking[0]],
                    "top_score": float(raw_scores[raw_ranking[0]]),
                    **raw_timing,
                },
                "processed": {
                    "top_label": self.classes[ranking[0]],
                    "top_score": float(scores[ranking[0]]),
                    **processed_timing,
                },
                "note": "Single-pass measurement; includes resize, enhancement when selected, and model input preparation; adaptive preparation also includes quality analysis and route selection. Shared file decoding and browser/network overhead are excluded. Timing varies by device and system load.",
            },
            "quality": quality.as_dict(),
        }


def serve(checkpoint_path: Path, host: str = "127.0.0.1", port: int = 8000) -> None:
    if host not in {"127.0.0.1", "localhost"}:
        raise ValueError("The research demo server can only bind to localhost.")
    predictor = Predictor(checkpoint_path)

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(WEB_DIR), **kwargs)

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/api/predict":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > MAX_UPLOAD_BYTES * 2:
                    raise ValueError("Upload is empty or too large (maximum 12 MB).")
                request = json.loads(self.rfile.read(length))
                image_bytes = base64.b64decode(request.get("image_base64", ""), validate=True)
                if not image_bytes or len(image_bytes) > MAX_UPLOAD_BYTES:
                    raise ValueError("Upload is empty or too large (maximum 12 MB).")
                result = predictor.predict(image_bytes)
                self._json(200, result)
            except (ValueError, json.JSONDecodeError, KeyError) as error:
                self._json(400, {"error": str(error)})
            except Exception as error:
                self._json(500, {"error": f"Prediction failed: {error}"})

        def _json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            print(f"[web] {self.address_string()} {format % args}", flush=True)

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Research demo ready at http://{host}:{port}", flush=True)
    print(f"Model: {predictor.model_name} · preprocessing: {predictor.fixed_strategy} · device: {predictor.device}", flush=True)
    print("Uploads are handled in memory by this local server and are not saved.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping local presentation server.", flush=True)
    finally:
        server.server_close()
