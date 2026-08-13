#!/usr/bin/env python3
"""LeRobot 向けローカル環境パッチ（NTFS / pyav 等）。"""

from __future__ import annotations

import sys
from pathlib import Path


def patch_train_utils_ntfs(lerobot_dir: Path) -> None:
    path = lerobot_dir / "src" / "lerobot" / "common" / "train_utils.py"
    if not path.is_file():
        raise FileNotFoundError(path)

    text = path.read_text(encoding="utf-8")
    marker = "Could not create checkpoint symlink on this filesystem"
    if marker in text:
        print(f"[patch] already applied: {path.name}")
        return

    old = """    relative_target = checkpoint_dir.relative_to(checkpoint_dir.parent)
    last_checkpoint_dir.symlink_to(relative_target)"""
    new = """    relative_target = checkpoint_dir.relative_to(checkpoint_dir.parent)
    try:
        last_checkpoint_dir.symlink_to(relative_target)
    except OSError:
        # NTFS 等 symlink 非対応 FS ではスキップ（step 番号 dir は保存済み）
        import logging

        logging.getLogger(__name__).warning(
            "Could not create checkpoint symlink on this filesystem; "
            "use checkpoints/%s/ instead of checkpoints/last/",
            checkpoint_dir.name,
        )"""
    if old not in text:
        raise RuntimeError(f"Patch anchor not found in {path}")

    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"[patch] applied NTFS checkpoint fix: {path.name}")


def patch_image_augmentations(lerobot_dir: Path) -> None:
    """Add gamma and texture jitter to LeRobot's configurable transforms.

    Keep this as an idempotent source patch because ``work/lerobot`` is an ignored
    checkout.  Training therefore remains reproducible after setup on a new host.
    """
    path = lerobot_dir / "src" / "lerobot" / "transforms" / "transforms.py"
    if not path.is_file():
        raise FileNotFoundError(path)

    text = path.read_text(encoding="utf-8")
    marker = "class GammaJitter(Transform):"
    if marker not in text:
        anchor = "\n\n@dataclass\nclass ImageTransformConfig:"
        addition = r'''

class GammaJitter(Transform):
    """Apply a gamma sampled uniformly from the configured interval."""

    def __init__(self, gamma: Sequence[float] = (0.8, 1.2)) -> None:
        super().__init__()
        if len(gamma) != 2 or not 0 < gamma[0] <= gamma[1]:
            raise ValueError(f"gamma must be a positive (min, max) pair, got {gamma}")
        self.gamma = float(gamma[0]), float(gamma[1])

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        return {"gamma": torch.empty(1).uniform_(*self.gamma).item()}

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        return self._call_kernel(F.adjust_gamma, inpt, gamma=params["gamma"])


class TextureJitter(Transform):
    """Add low/high-frequency sensor-like texture without changing geometry."""

    def __init__(self, strength: Sequence[float] = (0.0, 0.035), coarse_scale: int = 16) -> None:
        super().__init__()
        if len(strength) != 2 or not 0 <= strength[0] <= strength[1]:
            raise ValueError(f"strength must be a non-negative (min, max) pair, got {strength}")
        self.strength = float(strength[0]), float(strength[1])
        self.coarse_scale = int(coarse_scale)

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        return {"strength": torch.empty(1).uniform_(*self.strength).item()}

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        if not isinstance(inpt, torch.Tensor) or inpt.ndim < 3:
            return inpt
        strength = params["strength"]
        original_dtype = inpt.dtype
        image = inpt.float() / 255.0 if not original_dtype.is_floating_point else inpt.float()
        h, w = image.shape[-2:]
        leading = image.shape[:-3]
        coarse_h = max(2, h // self.coarse_scale)
        coarse_w = max(2, w // self.coarse_scale)
        noise_shape = (*leading, 1, coarse_h, coarse_w)
        coarse = torch.randn(noise_shape, device=image.device, dtype=image.dtype)
        coarse = torch.nn.functional.interpolate(
            coarse.reshape(-1, 1, coarse_h, coarse_w), size=(h, w), mode="bilinear", align_corners=False
        ).reshape(*leading, 1, h, w)
        fine = torch.randn_like(image) * 0.35
        output = (image + strength * (coarse + fine)).clamp(0.0, 1.0)
        if not original_dtype.is_floating_point:
            output = (output * 255.0).round().to(original_dtype)
        return output
'''
        if anchor not in text:
            raise RuntimeError(f"Image augmentation class anchor not found in {path}")
        text = text.replace(anchor, addition + anchor, 1)

    old_factory = '    if cfg.type == "SharpnessJitter":\n        return SharpnessJitter(**cfg.kwargs)\n'
    new_factory = '''    custom_transforms = {
        "SharpnessJitter": SharpnessJitter,
        "GammaJitter": GammaJitter,
        "TextureJitter": TextureJitter,
    }
    if cfg.type in custom_transforms:
        return custom_transforms[cfg.type](**cfg.kwargs)
'''
    if "custom_transforms = {" not in text:
        if old_factory not in text:
            raise RuntimeError(f"Image augmentation factory anchor not found in {path}")
        text = text.replace(old_factory, new_factory, 1)

    sharpness_block = '''            "sharpness": ImageTransformConfig(
                weight=1.0,
                type="SharpnessJitter",
                kwargs={"sharpness": (0.5, 1.5)},
            ),
'''
    extra_defaults = '''            "gamma": ImageTransformConfig(
                weight=1.0,
                type="GammaJitter",
                kwargs={"gamma": (0.75, 1.35)},
            ),
            "texture": ImageTransformConfig(
                weight=1.0,
                type="TextureJitter",
                kwargs={"strength": (0.01, 0.045), "coarse_scale": 16},
            ),
'''
    if '"texture": ImageTransformConfig(' not in text:
        if sharpness_block not in text:
            raise RuntimeError(f"Image augmentation defaults anchor not found in {path}")
        text = text.replace(sharpness_block, sharpness_block + extra_defaults, 1)

    path.write_text(text, encoding="utf-8")
    print(f"[patch] gamma/texture image augmentations ready: {path.name}")


def patch_pairwise_preference_training(lerobot_dir: Path) -> None:
    """Teach the ignored LeRobot checkout to consume chosen/rejected action chunks."""
    factory = lerobot_dir / "src/lerobot/datasets/factory.py"
    train = lerobot_dir / "src/lerobot/scripts/lerobot_train.py"
    if not factory.is_file() or not train.is_file():
        raise FileNotFoundError(f"LeRobot training sources missing under {lerobot_dir}")

    factory_text = factory.read_text(encoding="utf-8")
    factory_marker = 'if key == "preference.rejected_action"'
    if factory_marker not in factory_text:
        anchor = '''        if key == ACTION and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
'''
        addition = anchor + '''        if key == "preference.rejected_action" and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
'''
        if anchor not in factory_text:
            raise RuntimeError(f"Pairwise delta timestamp anchor missing: {factory}")
        factory.write_text(factory_text.replace(anchor, addition, 1), encoding="utf-8")

    train_text = train.read_text(encoding="utf-8")
    if "import os\n" not in train_text:
        train_text = train_text.replace("import logging\n", "import logging\nimport os\n", 1)
    if 'if "preference.rejected_action" in batch:' not in train_text:
        loss_anchor = '''        if sample_weights is not None:
'''
        loss_block = '''        if "preference.rejected_action" in batch:
            rejected_action = batch.pop("preference.rejected_action")
            preference_weight = batch.pop("preference.weight").reshape(-1)
            cpu_rng = torch.random.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
            chosen_loss, output_dict = policy.forward(batch, reduction="none")
            torch.random.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng)
            rejected_batch = dict(batch)
            rejected_batch["action"] = rejected_action
            rejected_loss, _ = policy.forward(rejected_batch, reduction="none")
            beta = float(os.environ.get("PARC_PREFERENCE_BETA", "5.0"))
            margin = float(os.environ.get("PARC_PREFERENCE_MARGIN", "0.05"))
            bc_weight = float(os.environ.get("PARC_PREFERENCE_BC_WEIGHT", "0.5"))
            preference = torch.nn.functional.softplus(
                margin - beta * (rejected_loss - chosen_loss)
            ) / beta
            active = preference_weight.sum().clamp_min(1.0)
            pairwise_loss = (preference * preference_weight).sum() / active
            loss = bc_weight * chosen_loss.mean() + pairwise_loss
            output_dict.update({
                "pairwise_loss": pairwise_loss.item(),
                "chosen_loss": chosen_loss.mean().item(),
                "rejected_loss": rejected_loss.mean().item(),
                "preference_gap": (rejected_loss - chosen_loss).mean().item(),
            })
        elif sample_weights is not None:
'''
        if loss_anchor not in train_text:
            raise RuntimeError(f"Pairwise loss anchor missing: {train}")
        train_text = train_text.replace(loss_anchor, loss_block, 1)
        preprocess_anchor = '''        batch = preprocessor(batch)
        train_tracker.dataloading_s = time.perf_counter() - start_time
'''
        preprocess_block = '''        if "preference.rejected_action" in batch:
            rejected_raw = batch.pop("preference.rejected_action")
            preference_weight = batch.pop("preference.weight")
            raw_batch = dict(batch)
            batch = preprocessor(dict(raw_batch))
            rejected_batch = dict(raw_batch)
            rejected_batch["action"] = rejected_raw
            rejected_batch = preprocessor(rejected_batch)
            batch["preference.rejected_action"] = rejected_batch["action"]
            batch["preference.weight"] = preference_weight.to(batch["action"].device)
        else:
            batch = preprocessor(batch)
        train_tracker.dataloading_s = time.perf_counter() - start_time
'''
        if preprocess_anchor not in train_text:
            raise RuntimeError(f"Pairwise preprocessing anchor missing: {train}")
        train_text = train_text.replace(preprocess_anchor, preprocess_block, 1)
    train.write_text(train_text, encoding="utf-8")
    print("[patch] pairwise preference training ready")


def main() -> None:
    lerobot_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "work" / "lerobot"
    patch_train_utils_ntfs(lerobot_dir)
    patch_image_augmentations(lerobot_dir)
    patch_pairwise_preference_training(lerobot_dir)


if __name__ == "__main__":
    main()
