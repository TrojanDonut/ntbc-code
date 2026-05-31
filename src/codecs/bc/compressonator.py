"""Optional wrapper around the AMD Compressonator CLI (``compressonatorcli``).

If the binary is on PATH we use it as a higher-quality BC1/BC4 baseline. If not,
the rest of the pipeline falls back to the pure-Python reference encoders in
``bc1.py`` and ``bc4.py``.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


def has_compressonator() -> bool:
    """True if a ``compressonatorcli`` binary is available on PATH."""
    return shutil.which("compressonatorcli") is not None


def _save_temp(arr: np.ndarray, path: Path) -> None:
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    if arr.ndim == 2:
        Image.fromarray(arr, mode="L").save(path)
    else:
        Image.fromarray(arr, mode="RGB").save(path)


def compressonator_roundtrip(
    image: np.ndarray, format_name: str, *, refine_steps: int = 2
) -> np.ndarray | None:
    """Run an encode/decode round-trip through Compressonator.

    Returns ``None`` if the binary is unavailable or the call fails so callers
    can transparently fall back to the Python reference encoder.
    """
    if not has_compressonator():
        return None
    if format_name not in {"BC1", "BC4"}:
        raise ValueError(f"Unsupported Compressonator format: {format_name}")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src = tmp / "src.png"
        compressed = tmp / f"compressed.dds"
        decoded = tmp / "decoded.png"
        _save_temp(image, src)
        try:
            subprocess.run(
                [
                    "compressonatorcli",
                    "-fd", format_name,
                    "-RefineSteps", str(refine_steps),
                    str(src), str(compressed),
                ],
                check=True, capture_output=True,
            )
            subprocess.run(
                ["compressonatorcli", str(compressed), str(decoded)],
                check=True, capture_output=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
        if not decoded.exists():
            return None
        decoded_img = np.array(Image.open(decoded))
        if format_name == "BC4" and decoded_img.ndim == 3:
            decoded_img = decoded_img[:, :, 0]
        if format_name == "BC4" and decoded_img.ndim == 2:
            decoded_img = decoded_img[:, :, None]
        return decoded_img
