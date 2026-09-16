from __future__ import annotations

from pathlib import Path
from typing import Protocol
import json

import numpy as np
import sys

from .action_schema import ActionSchema, GripperEncoding, validate_observation_schema


class BasePolicy(Protocol):
    def reset(self, instruction: str = "", seed: int | None = None) -> None: ...

    def get_action(self, obs: dict[str, np.ndarray]) -> np.ndarray: ...


class FrozenSmolVLAPolicy:
    def __init__(
        self,
        model_dir: str | Path,
        *,
        gripper_encoding: GripperEncoding | None = None,
        gripper_hold_steps: int | None = None,
    ):
        submission_dir = Path(__file__).resolve().parents[2] / "submission_template"
        if str(submission_dir) not in sys.path:
            sys.path.insert(0, str(submission_dir))
        from lerobot_bootstrap import setup_vendored_lerobot

        model_path = Path(model_dir)
        schema_data: dict = {}
        if gripper_encoding is None:
            schema_path = model_path / "action_schema.json"
            if not schema_path.is_file():
                raise ValueError(
                    "Checkpoint has no action_schema.json; pass gripper_encoding explicitly"
                )
            schema_data = json.loads(schema_path.read_text())
            gripper_encoding = schema_data["gripper_encoding"]
        if gripper_hold_steps is None:
            gripper_hold_steps = int(schema_data.get("gripper_hold_steps", 0))
        if not (model_path / "config.json").is_file():
            raise FileNotFoundError(f"SmolVLA checkpoint is incomplete: {model_path}")
        setup_vendored_lerobot()
        import torch
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from submission_template.smolvla_mypolicy import _load_pre_post_processors

        config = json.loads((model_path / "config.json").read_text())
        visual_keys = [
            key
            for key, feature in config["input_features"].items()
            if feature["type"] == "VISUAL" and "empty" not in key and "camera3" not in key
        ]
        processor_config = json.loads((model_path / "policy_preprocessor.json").read_text())
        rename_map = next(
            (
                step["config"].get("rename_map", {})
                for step in processor_config["steps"]
                if step["registry_name"] == "rename_observations_processor"
            ),
            {},
        )
        source_keys = [source for source, destination in rename_map.items() if destination in visual_keys]
        self.image_keys = source_keys or visual_keys
        if len(self.image_keys) != 2:
            raise ValueError(f"Expected two policy camera inputs, got {self.image_keys}")
        self.wrist_image_key = next(
            (key for key in self.image_keys if "wrist" in key.lower()), self.image_keys[1]
        )
        self.front_image_key = next(
            key for key in self.image_keys if key != self.wrist_image_key
        )
        configured_state_dim = int(config["input_features"]["observation.state"]["shape"][0])
        # The pinned upstream checkpoint declares 6 here, while its serialized
        # normalizer statistics contain the actual LIBERO 8-D state.
        self.state_dim = 8 if configured_state_dim == 6 else configured_state_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = SmolVLAPolicy.from_pretrained(str(model_path)).to(self.device).eval()
        self.preprocess, self.postprocess = _load_pre_post_processors(
            str(model_path), str(self.device)
        )
        self.action_schema = ActionSchema(gripper_encoding)
        self.gripper_hold_steps = gripper_hold_steps
        self._gripper_hold_remaining = 0
        self.instruction = ""

    def reset(self, instruction: str = "", seed: int | None = None) -> None:
        import torch
        from submission_template.smolvla_mypolicy import _normalize_instruction

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        self.instruction = _normalize_instruction(instruction)
        self._gripper_hold_remaining = 0
        self.policy.reset()

    def get_action(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        import torch
        from lerobot.utils.constants import ACTION
        from submission_template.smolvla_mypolicy import (
            _prepare_image,
            _quat_to_axis_angle,
            _to_preprocessor_frame,
        )

        validate_observation_schema(obs)
        state = np.concatenate(
            (
                np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
                _quat_to_axis_angle(obs["robot0_eef_quat"]),
                np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
            )
        )[: self.state_dim]
        frame = _to_preprocessor_frame(
            {
                self.front_image_key: _prepare_image(obs["agentview_image"]),
                self.wrist_image_key: _prepare_image(obs["robot0_eye_in_hand_image"]),
                "observation.state": state,
                "task": self.instruction,
            }
        )
        with torch.inference_mode():
            batch = self.preprocess(frame)
            action = self.policy.select_action(batch)
            action = self.postprocess(action)
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        result = np.asarray(action, dtype=np.float32).reshape(-1)
        if result.shape != (7,):
            raise ValueError(f"Expected action (7,), got {result.shape}")
        converted = self.action_schema.to_libero(result)
        if converted[6] > 0:
            self._gripper_hold_remaining = self.gripper_hold_steps
        elif self._gripper_hold_remaining > 0:
            converted[6] = 1.0
        self._gripper_hold_remaining = max(0, self._gripper_hold_remaining - 1)
        return converted


class ZeroBasePolicy:
    """Dependency-free policy used only by environment integration tests."""

    def reset(self, instruction: str = "", seed: int | None = None) -> None:
        del seed
        self.instruction = instruction

    def get_action(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        del obs
        return np.zeros(7, dtype=np.float32)


class RemoteBasePolicy:
    """Keeps the CUDA SmolVLA environment separate from the LIBERO environment."""

    def __init__(self, server_url: str = "http://127.0.0.1:8000", timeout_sec: float = 10.0):
        from pipeline.remote_policy import RemotePolicyClient

        self.client = RemotePolicyClient(server_url, timeout_sec)
        self.client.wait_for_server()

    def reset(self, instruction: str = "", seed: int | None = None) -> None:
        self.client.reset(instruction=instruction, seed=seed)

    def get_action(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        return self.client.get_action(obs)

