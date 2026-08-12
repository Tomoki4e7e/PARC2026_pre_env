"""Deterministic visual domain randomization for local LIBERO collection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class VisualDomainConfig:
    mode: str = "mixed"
    seed: int = 2026
    light_strength: float = 0.35
    background_strength: float = 0.18
    non_target_strength: float = 0.22
    camera_position_shift_m: float = 0.012
    camera_fovy_shift_deg: float = 2.0


def _name(model: Any, kind: str, index: int) -> str:
    resolver = getattr(model, f"{kind}_id2name", None)
    if resolver is None:
        return ""
    value = resolver(index)
    return "" if value is None else str(value).lower()


def apply_visual_domain_randomization(
    env: Any, obj_of_interest: set[str], config: VisualDomainConfig
) -> dict[str, Any]:
    """Change rendering properties only; robot/object poses remain untouched."""
    if config.mode not in {"bright", "dark", "mixed"}:
        raise ValueError(f"unknown visual domain mode: {config.mode}")
    rng = np.random.default_rng(config.seed)
    model = env.sim.model
    if config.mode == "bright":
        light_scale = rng.uniform(1.0, 1.0 + config.light_strength)
    elif config.mode == "dark":
        light_scale = rng.uniform(max(0.25, 1.0 - config.light_strength), 0.9)
    else:
        light_scale = 1.0 + rng.choice([-1.0, 1.0]) * rng.uniform(0.12, config.light_strength)

    for attribute in ("light_diffuse", "light_ambient", "light_specular"):
        values = getattr(model, attribute, None)
        if values is not None:
            values[:] = np.clip(values * light_scale, 0.0, 1.0)

    interested_names = {name.lower() for name in obj_of_interest}
    background_geoms, non_target_geoms = [], []
    background_materials: set[int] = set()
    geom_rgba = getattr(model, "geom_rgba", None)
    geom_bodyid = getattr(model, "geom_bodyid", None)
    if geom_rgba is not None and geom_bodyid is not None:
        for geom_id in range(int(getattr(model, "ngeom", len(geom_rgba)))):
            geom_name = _name(model, "geom", geom_id)
            body_name = _name(model, "body", int(geom_bodyid[geom_id]))
            combined = f"{geom_name} {body_name}"
            if int(geom_bodyid[geom_id]) == 0 or any(
                token in combined for token in ("table", "floor", "wall", "counter")
            ):
                jitter = rng.uniform(1.0 - config.background_strength,
                                     1.0 + config.background_strength, size=3)
                geom_rgba[geom_id, :3] = np.clip(geom_rgba[geom_id, :3] * jitter, 0.0, 1.0)
                background_geoms.append(geom_name or f"geom_{geom_id}")
                geom_matid = getattr(model, "geom_matid", None)
                if geom_matid is not None and int(geom_matid[geom_id]) >= 0:
                    background_materials.add(int(geom_matid[geom_id]))
                continue
            if any(token in combined for token in ("robot", "panda", "gripper", "mount")):
                continue
            if interested_names and any(name in combined for name in interested_names):
                continue
            jitter = rng.uniform(1.0 - config.non_target_strength,
                                 1.0 + config.non_target_strength, size=3)
            geom_rgba[geom_id, :3] = np.clip(geom_rgba[geom_id, :3] * jitter, 0.0, 1.0)
            non_target_geoms.append(geom_name or f"geom_{geom_id}")

    # Add a mild, low-frequency procedural texture only to materials used by
    # static background geoms.  This preserves task-object color semantics.
    background_textures = []
    mat_texid = getattr(model, "mat_texid", None)
    tex_rgb = getattr(model, "tex_rgb", None)
    tex_adr = getattr(model, "tex_adr", None)
    tex_height = getattr(model, "tex_height", None)
    tex_width = getattr(model, "tex_width", None)
    if all(value is not None for value in (mat_texid, tex_rgb, tex_adr, tex_height, tex_width)):
        texture_ids: set[int] = set()
        for material_id in background_materials:
            ids = np.asarray(mat_texid[material_id]).reshape(-1)
            texture_ids.update(int(value) for value in ids if int(value) >= 0)
        for texture_id in sorted(texture_ids):
            height = int(tex_height[texture_id])
            width = int(tex_width[texture_id])
            address = int(tex_adr[texture_id])
            size = height * width * 3
            if height <= 0 or width <= 0 or address < 0 or address + size > len(tex_rgb):
                continue
            texture = tex_rgb[address : address + size].reshape(height, width, 3)
            grid_h, grid_w = min(8, height), min(8, width)
            coarse = rng.uniform(-1.0, 1.0, size=(grid_h, grid_w, 1))
            coarse = np.repeat(np.repeat(coarse, int(np.ceil(height / grid_h)), axis=0),
                               int(np.ceil(width / grid_w)), axis=1)[:height, :width]
            strength = config.background_strength * 45.0
            texture[:] = np.clip(texture.astype(np.float32) + strength * coarse, 0, 255).astype(
                texture.dtype
            )
            background_textures.append(texture_id)

    camera_changes = []
    cam_pos = getattr(model, "cam_pos", None)
    cam_fovy = getattr(model, "cam_fovy", None)
    if cam_pos is not None:
        for camera_id in range(int(getattr(model, "ncam", len(cam_pos)))):
            shift = rng.uniform(-config.camera_position_shift_m,
                                config.camera_position_shift_m, size=3)
            cam_pos[camera_id] += shift
            if cam_fovy is not None:
                cam_fovy[camera_id] = np.clip(
                    cam_fovy[camera_id]
                    + rng.uniform(-config.camera_fovy_shift_deg, config.camera_fovy_shift_deg),
                    20.0, 90.0,
                )
            camera_changes.append({
                "camera": _name(model, "camera", camera_id) or f"camera_{camera_id}",
                "position_shift": shift.tolist(),
            })
    env.sim.forward()
    return {
        "config": asdict(config),
        "light_scale": float(light_scale),
        "background_geoms": background_geoms,
        "background_textures": background_textures,
        "non_target_geoms": non_target_geoms,
        "camera_changes": camera_changes,
    }
