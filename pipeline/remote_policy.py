import logging
import time

import msgpack
import numpy as np
import requests

logger = logging.getLogger(__name__)


def serialize_obs(obs: dict[str, np.ndarray]) -> bytes:
    serialized = {}
    for key, arr in obs.items():
        serialized[key] = {
            "data": arr.tobytes(),
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
        }
    return msgpack.packb(serialized, use_bin_type=True)


def deserialize_action(data: bytes) -> np.ndarray:
    unpacked = msgpack.unpackb(data, raw=False)
    return np.frombuffer(unpacked["data"], dtype=np.float32).copy()


class RemotePolicyClient:

    def __init__(self, server_url: str = "http://localhost:8000", timeout_sec: float = 10.0):
        self.server_url = server_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self._session = requests.Session()

    def wait_for_server(self, timeout: float = 120.0) -> None:
        deadline = time.time() + timeout
        interval = 1.0
        while time.time() < deadline:
            try:
                resp = self._session.get(
                    f"{self.server_url}/health", timeout=5.0
                )
                if resp.status_code == 200:
                    logger.info("ポリシーサーバーに接続: %s", self.server_url)
                    return
            except requests.ConnectionError:
                pass
            time.sleep(interval)
            interval = min(interval * 1.5, 5.0)

        raise RuntimeError(
            f"ポリシーサーバー ({self.server_url}) に {timeout:.0f}秒以内に接続できませんでした。"
            " サーバーが起動しているか確認してください。"
        )

    def get_action(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        payload = serialize_obs(obs)
        try:
            resp = self._session.post(
                f"{self.server_url}/act",
                data=payload,
                headers={"Content-Type": "application/x-msgpack"},
                timeout=self.timeout_sec,
            )
        except requests.ConnectionError as e:
            raise RuntimeError(
                f"ポリシーサーバーとの接続が切断されました: {e}"
            ) from e
        except requests.ReadTimeout as e:
            raise RuntimeError(
                f"ポリシーサーバーが {self.timeout_sec}秒以内に応答しませんでした。"
                " モデル推論に時間がかかりすぎている可能性があります。"
            ) from e

        resp.raise_for_status()
        action = deserialize_action(resp.content)

        if action.shape != (7,):
            raise ValueError(
                f"action の shape が不正です: 期待 (7,), 実際 {action.shape}"
            )
        return action

    def reset(self, instruction: str = "", seed: int | None = None) -> None:
        try:
            body = {"instruction": instruction}
            if seed is not None:
                body["seed"] = seed
            resp = self._session.post(
                f"{self.server_url}/reset",
                json=body,
                timeout=self.timeout_sec,
            )
            resp.raise_for_status()
        except requests.ConnectionError as e:
            raise RuntimeError(
                f"ポリシーサーバーとの接続が切断されました: {e}"
            ) from e
