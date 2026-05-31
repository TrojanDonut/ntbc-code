"""Image-quality metrics used in the evaluation pipeline.

PSNR and SSIM are required and use scikit-image. LPIPS and FLIP are optional
and load lazily -- both are heavyweight dependencies (LPIPS pulls a pretrained
network, FLIP pulls a custom evaluator) so we tolerate ImportError and report
``None`` instead of crashing the whole eval run.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def _as_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    return np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _as_3ch(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 1:
        return np.repeat(arr, 3, axis=-1)
    return arr


def psnr(reference: np.ndarray, prediction: np.ndarray) -> float:
    reference = _as_uint8(reference)
    prediction = _as_uint8(prediction)
    return float(peak_signal_noise_ratio(reference, prediction, data_range=255))


def ssim(reference: np.ndarray, prediction: np.ndarray) -> float:
    reference = _as_uint8(reference)
    prediction = _as_uint8(prediction)
    # Squeeze trailing single-channel dim so scikit-image treats the array as
    # 2D grayscale rather than a 3-axis (H, W, 1) spatial volume.
    if reference.ndim == 3 and reference.shape[-1] == 1:
        reference = reference[..., 0]
        prediction = prediction[..., 0]
    channel_axis = -1 if reference.ndim == 3 else None
    return float(
        structural_similarity(
            reference, prediction, data_range=255, channel_axis=channel_axis
        )
    )


_lpips_model = None


def lpips(reference: np.ndarray, prediction: np.ndarray) -> float | None:
    """Optional LPIPS metric. Returns ``None`` if the package is unavailable."""
    global _lpips_model
    try:
        if _lpips_model is None:
            mod = importlib.import_module("lpips")
            _lpips_model = mod.LPIPS(net="alex", verbose=False).eval()
    except (ImportError, OSError, RuntimeError):
        return None

    import torch

    reference = _as_3ch(_as_uint8(reference)).astype(np.float32) / 127.5 - 1.0
    prediction = _as_3ch(_as_uint8(prediction)).astype(np.float32) / 127.5 - 1.0
    a = torch.from_numpy(reference).permute(2, 0, 1).unsqueeze(0)
    b = torch.from_numpy(prediction).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        d = _lpips_model(a, b)
    return float(d.detach().cpu().item())


def flip(reference: np.ndarray, prediction: np.ndarray) -> float | None:
    """Optional FLIP metric. Returns ``None`` if the package is unavailable."""
    try:
        mod = importlib.import_module("flip_evaluator")
    except ImportError:
        try:
            mod = importlib.import_module("flip")
        except ImportError:
            return None
    if not hasattr(mod, "evaluate"):
        return None
    reference = _as_3ch(_as_uint8(reference)).astype(np.float32) / 255.0
    prediction = _as_3ch(_as_uint8(prediction)).astype(np.float32) / 255.0
    try:
        result = mod.evaluate(reference, prediction, "LDR")
    except Exception:
        return None
    # flip_evaluator.evaluate returns (error_map, mean_error, parameters);
    # the older `flip` API returned (error_map, mean_error).
    if isinstance(result, tuple) and len(result) >= 2:
        return float(result[1])
    return None


@dataclass
class TextureMetrics:
    psnr: float
    ssim: float
    lpips: float | None
    flip: float | None


def compute_all(reference: np.ndarray, prediction: np.ndarray) -> TextureMetrics:
    return TextureMetrics(
        psnr=psnr(reference, prediction),
        ssim=ssim(reference, prediction),
        lpips=lpips(reference, prediction),
        flip=flip(reference, prediction),
    )
