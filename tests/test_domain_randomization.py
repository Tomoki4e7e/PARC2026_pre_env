import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
_EXAMPLES = _ROOT / "examples" / "smolvla_libero_spatial_lora_local"
sys.path.insert(0, str(_EXAMPLES))

from domain_randomization import VisualDomainConfig, apply_visual_domain_randomization


class _Model:
    nlight = 1
    light_diffuse = np.full((1, 3), 0.5)
    light_ambient = np.full((1, 3), 0.2)
    light_specular = np.full((1, 3), 0.3)
    ngeom = 3
    geom_rgba = np.full((3, 4), 0.5)
    geom_bodyid = np.asarray([0, 1, 2])
    geom_matid = np.asarray([0, -1, -1])
    mat_texid = np.asarray([0])
    tex_adr = np.asarray([0])
    tex_height = np.asarray([2])
    tex_width = np.asarray([2])
    tex_rgb = np.full(12, 128, dtype=np.uint8)
    ncam = 1
    cam_pos = np.zeros((1, 3))
    cam_fovy = np.asarray([45.0])

    def geom_id2name(self, index):
        return ["floor", "black_bowl_visual", "cream_cheese_visual"][index]

    def body_id2name(self, index):
        return ["world", "black_bowl", "cream_cheese"][index]

    def camera_id2name(self, index):
        return "agentview"


class _Sim:
    def __init__(self):
        self.model = _Model()

    def forward(self):
        pass


class _Env:
    def __init__(self):
        self.sim = _Sim()


def test_visual_randomization_keeps_target_color_and_changes_rendering():
    env = _Env()
    target_before = env.sim.model.geom_rgba[1].copy()
    texture_before = env.sim.model.tex_rgb.copy()
    manifest = apply_visual_domain_randomization(
        env, {"black_bowl"}, VisualDomainConfig(mode="bright", seed=7)
    )
    assert manifest["light_scale"] >= 1.0
    assert np.array_equal(env.sim.model.geom_rgba[1], target_before)
    assert not np.array_equal(env.sim.model.geom_rgba[0], np.full(4, 0.5))
    assert not np.array_equal(env.sim.model.tex_rgb, texture_before)
    assert manifest["background_textures"] == [0]
    assert manifest["camera_changes"]
