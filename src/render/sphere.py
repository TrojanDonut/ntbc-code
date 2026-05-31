"""Minimal NumPy PBR sphere renderer used to produce a qualitative figure.

Reasoning: hardware-accelerated renderers (pyrender, moderngl) require OpenGL
contexts that often fail on headless Linux. For the seminar deliverable we
only need a single, deterministic, reproducible image showing how the
reference / BC-baseline / NTBC textures look when applied to geometry, so a
short NumPy raycaster is more reliable.

Shading uses a Lambertian diffuse + Cook-Torrance GGX specular BRDF with a
single directional light. The implementation deliberately mirrors the
``image-based lighting``-free path used in the NTBC paper figures.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..data.material import BCFormat, Material, Role
from ..eval.inference import baseline_reconstruct, ntbc_reconstruct
from ..models.ntbc import NTBC


@dataclass
class RenderConfig:
    """Camera and lighting settings for the sphere render."""

    size: int = 384
    light_dir: tuple[float, float, float] = (-0.5, 0.7, 1.0)
    light_color: tuple[float, float, float] = (1.5, 1.5, 1.5)
    ambient: tuple[float, float, float] = (0.05, 0.05, 0.05)
    background: tuple[float, float, float] = (0.15, 0.15, 0.18)


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n > 1e-8, n, 1.0)


def _sample_bilinear(tex: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Bilinear lookup. ``tex``: (H, W) or (H, W, C); ``uv``: (..., 2) in [0, 1]."""
    h, w = tex.shape[:2]
    u = np.clip(uv[..., 0], 0.0, 1.0) * (w - 1)
    v = np.clip(uv[..., 1], 0.0, 1.0) * (h - 1)
    x0 = np.floor(u).astype(np.int64)
    y0 = np.floor(v).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    fx = (u - x0)[..., None] if tex.ndim == 3 else (u - x0)
    fy = (v - y0)[..., None] if tex.ndim == 3 else (v - y0)
    t00 = tex[y0, x0]
    t01 = tex[y0, x1]
    t10 = tex[y1, x0]
    t11 = tex[y1, x1]
    top = t00 * (1.0 - fx) + t01 * fx
    bot = t10 * (1.0 - fx) + t11 * fx
    return top * (1.0 - fy) + bot * fy


def _ggx_smith_render(
    normal_world: np.ndarray,
    albedo: np.ndarray,
    roughness: np.ndarray,
    metalness: np.ndarray,
    ao: np.ndarray,
    cfg: RenderConfig,
) -> np.ndarray:
    """Simple Cook-Torrance GGX shader, all arrays HxWx(3 or 1) in [0, 1]."""
    light_dir = _normalize(np.array(cfg.light_dir, dtype=np.float32)[None, None, :])
    view_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)[None, None, :]
    half_dir = _normalize(light_dir + view_dir)

    n_dot_l = np.clip((normal_world * light_dir).sum(axis=-1, keepdims=True), 0.0, 1.0)
    n_dot_v = np.clip((normal_world * view_dir).sum(axis=-1, keepdims=True), 0.0, 1.0)
    n_dot_h = np.clip((normal_world * half_dir).sum(axis=-1, keepdims=True), 0.0, 1.0)
    v_dot_h = np.clip((view_dir * half_dir).sum(axis=-1, keepdims=True), 0.0, 1.0)

    a = (roughness ** 2)
    a2 = (a ** 2) + 1e-6
    denom = (n_dot_h ** 2) * (a2 - 1.0) + 1.0
    D = a2 / (math.pi * denom ** 2 + 1e-6)

    k = ((roughness + 1.0) ** 2) / 8.0
    G1_v = n_dot_v / (n_dot_v * (1.0 - k) + k + 1e-6)
    G1_l = n_dot_l / (n_dot_l * (1.0 - k) + k + 1e-6)
    G = G1_v * G1_l

    f0_dielectric = np.full_like(albedo, 0.04)
    F0 = f0_dielectric * (1.0 - metalness) + albedo * metalness
    F = F0 + (1.0 - F0) * np.power(1.0 - v_dot_h, 5.0)

    specular = (D * G * F) / (4.0 * n_dot_l * n_dot_v + 1e-6)
    kd = (1.0 - F) * (1.0 - metalness)
    diffuse = kd * albedo / math.pi

    light_color = np.array(cfg.light_color, dtype=np.float32)[None, None, :]
    ambient = np.array(cfg.ambient, dtype=np.float32)[None, None, :]
    radiance = (diffuse + specular) * light_color * n_dot_l + ambient * albedo * ao
    return np.clip(radiance, 0.0, 1.0)


def _sphere_uvs_and_normals(size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render a sphere at the origin using an orthographic camera.

    Returns ``(uv, world_normal, mask)`` arrays of shape ``(size, size, ...)``.
    """
    ys = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    xs = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    r2 = gx * gx + gy * gy
    mask = r2 <= 1.0
    gz = np.where(mask, np.sqrt(np.clip(1.0 - r2, 0.0, 1.0)), 0.0)
    normal = np.stack([gx, gy, gz], axis=-1)
    # Spherical UV: u = atan2(x, z) / (2pi) + 0.5; v = asin(y) / pi + 0.5.
    u = np.arctan2(gx, gz) / (2.0 * math.pi) + 0.5
    v = np.arcsin(np.clip(gy, -1.0, 1.0)) / math.pi + 0.5
    uv = np.stack([u, v], axis=-1)
    return uv, normal, mask


def _to_float01(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    return arr.astype(np.float32)


def _normal_map_to_world(
    normal_tex: np.ndarray, sphere_normal: np.ndarray
) -> np.ndarray:
    """Apply a tangent-space normal map onto the sphere normal field.

    For a sphere we use a simple TBN approximation: tangent = d(pos)/du,
    bitangent = d(pos)/dv. The normal map is expected to use the OpenGL
    convention (R = +X, G = +Y, B = +Z).
    """
    # Reasonable tangent/bitangent approximation for an orthographic sphere
    # view: tangent points east, bitangent points north, both perpendicular to
    # the surface normal in the image plane.
    n = sphere_normal
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)[None, None, :]
    tangent = _normalize(np.cross(up, n))
    bitangent = _normalize(np.cross(n, tangent))
    nmap = normal_tex * 2.0 - 1.0
    perturbed = (
        nmap[..., 0:1] * tangent
        + nmap[..., 1:2] * bitangent
        + nmap[..., 2:3] * n
    )
    return _normalize(perturbed)


def render_sphere(
    layers: dict[Role, np.ndarray], cfg: RenderConfig | None = None
) -> np.ndarray:
    """Render a sphere shaded with the provided material layers."""
    cfg = cfg or RenderConfig()
    uv, sphere_normal, mask = _sphere_uvs_and_normals(cfg.size)
    sphere_normal = _normalize(sphere_normal)

    def _get(role: Role, default: float, ch: int = 3) -> np.ndarray:
        if role in layers:
            tex = _to_float01(layers[role])
            return _sample_bilinear(tex, uv)
        return np.full((cfg.size, cfg.size, ch), default, dtype=np.float32)

    albedo = _get(Role.COLOR, 0.8, ch=3)
    if albedo.ndim == 2 or albedo.shape[-1] == 1:
        albedo = np.broadcast_to(albedo.reshape(cfg.size, cfg.size, 1), (cfg.size, cfg.size, 3)).copy()
    roughness = _get(Role.ROUGHNESS, 0.5, ch=1)[..., :1]
    metalness = _get(Role.METALNESS, 0.0, ch=1)[..., :1]
    ao_layer = _get(Role.AO, 1.0, ch=1)[..., :1]

    if Role.NORMAL in layers:
        normal_tex = _to_float01(layers[Role.NORMAL])
        normal_tex = _sample_bilinear(normal_tex, uv)
        normal_world = _normal_map_to_world(normal_tex, sphere_normal)
    else:
        normal_world = sphere_normal

    radiance = _ggx_smith_render(
        normal_world, albedo, roughness, metalness, ao_layer, cfg
    )
    bg = np.array(cfg.background, dtype=np.float32)[None, None, :]
    out = np.where(mask[..., None], radiance, bg)
    return (out * 255.0 + 0.5).clip(0, 255).astype(np.uint8)


def _layers_by_role(
    material: Material, reconstructions: dict[str, np.ndarray]
) -> dict[Role, np.ndarray]:
    result: dict[Role, np.ndarray] = {}
    for layer in material.layers:
        if layer.name in reconstructions:
            result[layer.role] = reconstructions[layer.name]
    return result


def render_triptych(model: NTBC, material: Material, cfg: RenderConfig | None = None) -> np.ndarray:
    """Render Reference / BC baseline / NTBC sphere shots side-by-side."""
    cfg = cfg or RenderConfig()

    reference_layers: dict[Role, np.ndarray] = {}
    baseline_layers: dict[Role, np.ndarray] = {}
    for layer in material.layers:
        arr = (layer.data.permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        reference_layers[layer.role] = arr
        baseline_layers[layer.role] = baseline_reconstruct(layer)

    ntbc_recons = ntbc_reconstruct(model, material)
    ntbc_layers = _layers_by_role(material, ntbc_recons)

    ref_img = render_sphere(reference_layers, cfg)
    base_img = render_sphere(baseline_layers, cfg)
    ntbc_img = render_sphere(ntbc_layers, cfg)
    return np.concatenate([ref_img, base_img, ntbc_img], axis=1)


@torch.no_grad()
def render_triptych_to_path(
    model: NTBC, material: Material, out_path: Path, cfg: RenderConfig | None = None
) -> Path:
    img = render_triptych(model, material, cfg)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(out_path)
    return out_path
