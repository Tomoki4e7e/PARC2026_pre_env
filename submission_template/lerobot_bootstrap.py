"""LeRobot 同梱版を Python 3.10 採点環境で import 可能にするブートストラップ。

lerobot==0.6.0 は PyPI 上 Python>=3.12 必須のため、提出 zip 内 vendor/lerobot/
にソースを同梱し、本モジュールで sys.path と typing 拡張を整えてから import する。
"""

from __future__ import annotations

import sys
import typing
from pathlib import Path


def _patch_transformers_compat() -> None:
    """採点環境の transformers (4.45–4.46) に無い API を補う。"""
    try:
        from transformers.utils import torch_compilable_check  # noqa: F401
    except ImportError:
        import transformers.utils as transformers_utils

        def torch_compilable_check(condition, message=""):
            if not condition:
                raise ValueError(message() if callable(message) else message)

        transformers_utils.torch_compilable_check = torch_compilable_check


def _patch_typing_for_py310() -> None:
    try:
        from typing_extensions import Self, Unpack
    except ImportError as exc:
        raise RuntimeError(
            "typing_extensions が必要です（requirements.txt に含めてください）"
        ) from exc

    if not hasattr(typing, "Self"):
        typing.Self = Self
    if not hasattr(typing, "Unpack"):
        typing.Unpack = Unpack


def _stub_policies_package(vendor_root: Path) -> None:
    """policies/__init__.py が全 policy config を eager import するのを回避する。

    SmolVLA 推論に不要な eo1 / wall_x 等の config は transformers 新 API に依存し、
    採点環境 (transformers 4.45–4.46) では ImportError になる。
    """
    import types

    policies_dir = vendor_root / "lerobot" / "policies"
    if not policies_dir.is_dir():
        return

    name = "lerobot.policies"
    if name in sys.modules:
        return

    pkg = types.ModuleType(name)
    pkg.__path__ = [str(policies_dir)]
    pkg.__package__ = name
    sys.modules[name] = pkg


def setup_vendored_lerobot() -> None:
    _patch_typing_for_py310()
    _patch_transformers_compat()

    vendor_root = Path(__file__).resolve().parent / "vendor"
    if not vendor_root.is_dir():
        return
    vendor_str = str(vendor_root)
    if vendor_str not in sys.path:
        sys.path.insert(0, vendor_str)

    _stub_policies_package(vendor_root)
