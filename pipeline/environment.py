import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from libero.libero import benchmark as libero_benchmark
from libero.libero.envs import OffScreenRenderEnv

from .config import EvalConfig, PerturbationConfig

logger = logging.getLogger(__name__)


def _ensure_libero_t1_registered() -> bool:
    try:
        from compe.t1 import register_t1
        register_t1()
        logger.info("libero_t1 スイートを登録しました。")
        return True
    except ImportError:
        logger.debug("compe.t1 が見つかりません（Track 1 カスタムスイート未対応環境）。")
        return False
    except Exception as exc:
        logger.warning("register_t1() に失敗しました: %s", exc)
        return False


def parse_obj_of_interest(bddl_file: str | Path) -> set[str]:
    txt = Path(bddl_file).read_text()
    m = re.search(r"\(:obj_of_interest(.*?)\)\s*\(:init", txt, re.S)
    return set(m.group(1).split()) if m else set()


@dataclass
class TaskInfo:

    task_id: int
    name: str
    language: str
    bddl_file: str
    init_states: torch.Tensor
    benchmark_name: str


class EnvironmentManager:

    def __init__(self, eval_config: EvalConfig):
        self.config = eval_config

        _ensure_libero_t1_registered()
        self._benchmark_dict = libero_benchmark.get_benchmark_dict()

    def get_task_suite(self, benchmark_name: str) -> Any:
        if benchmark_name not in self._benchmark_dict:
            available = list(self._benchmark_dict.keys())
            raise ValueError(
                f"不明なベンチマーク: {benchmark_name}. 利用可能: {available}"
            )
        return self._benchmark_dict[benchmark_name](
            task_order_index=self.config.task_order_index
        )

    def get_task_infos(self, benchmark_name: str) -> list[TaskInfo]:
        suite = self.get_task_suite(benchmark_name)
        tasks = []

        for task_id in range(suite.get_num_tasks()):
            task = suite.get_task(task_id)
            bddl_path = suite.get_task_bddl_file_path(task_id)
            init_states = suite.get_task_init_states(task_id)

            tasks.append(
                TaskInfo(
                    task_id=task_id,
                    name=task.name,
                    language=task.language,
                    bddl_file=bddl_path,
                    init_states=init_states,
                    benchmark_name=benchmark_name,
                )
            )

        logger.info("%s から %d タスクを読み込み", benchmark_name, len(tasks))
        return tasks

    def get_obj_of_interest(self, task_info: TaskInfo) -> set[str]:
        return parse_obj_of_interest(task_info.bddl_file)

    def create_env(self, task_info: TaskInfo) -> OffScreenRenderEnv:
        import os

        env_args = {
            "bddl_file_name": task_info.bddl_file,
            "camera_heights": self.config.camera_height,
            "camera_widths": self.config.camera_width,
            "control_freq": int(os.environ.get("LIBERO_CONTROL_FREQ", "20")),
        }

        env = OffScreenRenderEnv(**env_args)
        env.seed(self.config.seed)
        logger.info("環境を作成: %s", task_info.name)
        return env

    def get_perturbed_init_states(
        self,
        task_info: TaskInfo,
        perturbation: PerturbationConfig,
        n_episodes: int,
    ) -> np.ndarray:
        if isinstance(task_info.init_states, torch.Tensor):
            init_states = task_info.init_states.numpy()
        else:
            init_states = np.asarray(task_info.init_states)
        n_available = init_states.shape[0]


        indices = np.arange(n_episodes) % n_available
        sampled_states = init_states[indices].copy()

        if perturbation.robot_init_pos_noise > 0:


            noise_scale = perturbation.robot_init_pos_noise
            logger.info("ロボット初期位置に摂動を適用: noise_scale=%.3f", noise_scale)


        if perturbation.object_pos_noise > 0:
            noise_scale = perturbation.object_pos_noise
            logger.info("物体位置に摂動を適用: noise_scale=%.3f", noise_scale)


        return sampled_states

    def apply_observation_noise(
        self,
        obs: dict[str, np.ndarray],
        perturbation: PerturbationConfig,
    ) -> dict[str, np.ndarray]:
        if perturbation.observation_noise <= 0:
            return obs

        noised_obs = {}
        for key, value in obs.items():
            if "image" in key or "rgb" in key:

                noise = np.random.normal(
                    0, perturbation.observation_noise * 255, value.shape
                )
                noised_obs[key] = np.clip(value + noise, 0, 255).astype(value.dtype)
            else:

                noise = np.random.normal(0, perturbation.observation_noise, value.shape)
                noised_obs[key] = (value + noise).astype(value.dtype)

        return noised_obs

    def apply_action_noise(
        self,
        action: np.ndarray,
        perturbation: PerturbationConfig,
    ) -> np.ndarray:
        if perturbation.action_noise <= 0:
            return action

        noise = np.random.normal(0, perturbation.action_noise, action.shape)
        return action + noise
