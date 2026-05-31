"""Download free PBR materials from ambientCG.

ambientCG materials are CC0 and offered as direct zip downloads under URLs like
``https://ambientcg.com/get?file=<MaterialName>_<resolution>-PNG.zip``. Inside,
files follow the convention ``<MaterialName>_<resolution>-PNG_<Role>.png``.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import requests
from tqdm import tqdm


AMBIENTCG_GET_URL = "https://ambientcg.com/get"


# Curated small list of CC0 materials known to have full PBR coverage at 1K-PNG.
# Each is a stress-tested combination of color + normal + roughness + displacement
# + (optionally) AO and metalness, matching the texture mix used in the NTBC paper.
DEFAULT_MATERIALS: tuple[str, ...] = (
    "MetalPlates013",
    "Wood062",
    "Bricks075A",
    "Carpet015",
    "Rocks023",
)

# Materials highlighted in the NTBC paper (Fujieda & Harada 2024, Fig. 1/4/5).
PAPER_MATERIALS: tuple[str, ...] = (
    "MetalPlates013",
    "Carpet015",
)

# ambientCG metals / surfaces that typically ship metalness + full PBR stack (6+ layers).
HIGH_LAYER_MATERIALS: tuple[str, ...] = (
    "MetalPlates013",
    "MetalPlates007",
    "Metal026",
    "Metal016",
    "Metal032",
    "Metal007",
    "Ground037",
    "Tiles107",
)


@dataclass
class DownloadResult:
    name: str
    path: Path
    skipped: bool
    error: str | None = None


def _download_zip(material: str, resolution: str = "1K-PNG", timeout: int = 60) -> bytes:
    file_arg = f"{material}_{resolution}.zip"
    params = {"file": file_arg}
    headers = {"User-Agent": "ntbc-seminar/0.1 (+local research)"}
    with requests.get(
        AMBIENTCG_GET_URL,
        params=params,
        headers=headers,
        stream=True,
        timeout=timeout,
    ) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        buf = io.BytesIO()
        with tqdm(
            total=total, unit="B", unit_scale=True, desc=material, leave=False
        ) as pbar:
            for chunk in response.iter_content(chunk_size=1 << 14):
                if not chunk:
                    continue
                buf.write(chunk)
                pbar.update(len(chunk))
        return buf.getvalue()


def _extract_pngs(blob: bytes, dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    extracted = 0
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        for info in zf.infolist():
            name = Path(info.filename).name
            if not name.lower().endswith(".png"):
                continue
            with zf.open(info) as src, (dest / name).open("wb") as dst:
                dst.write(src.read())
            extracted += 1
    return extracted


def download_material(
    name: str,
    out_dir: Path,
    *,
    resolution: str = "1K-PNG",
    overwrite: bool = False,
) -> DownloadResult:
    """Download a single ambientCG material zip and extract its PNG layers."""
    dest = Path(out_dir) / name
    if dest.exists() and any(dest.glob("*.png")) and not overwrite:
        return DownloadResult(name=name, path=dest, skipped=True)
    try:
        blob = _download_zip(name, resolution=resolution)
        n = _extract_pngs(blob, dest)
        if n == 0:
            return DownloadResult(
                name=name, path=dest, skipped=False, error="zip contained no PNGs"
            )
        return DownloadResult(name=name, path=dest, skipped=False)
    except (requests.RequestException, zipfile.BadZipFile, OSError) as exc:
        return DownloadResult(name=name, path=dest, skipped=False, error=str(exc))


def download_materials(
    names: list[str],
    out_dir: Path,
    *,
    resolution: str = "1K-PNG",
    overwrite: bool = False,
) -> list[DownloadResult]:
    """Download multiple ambientCG materials, returning per-material results."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[DownloadResult] = []
    for name in names:
        result = download_material(
            name, out_dir, resolution=resolution, overwrite=overwrite
        )
        results.append(result)
    return results
