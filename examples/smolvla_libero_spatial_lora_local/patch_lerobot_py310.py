"""LeRobot vendor/ 向け Python 3.10 互換パッチを適用する。"""

from __future__ import annotations

import sys
from pathlib import Path


def patch_video_py(vendor: Path) -> None:
    path = vendor / "configs" / "video.py"
    text = path.read_text()
    old = "from typing import Any, ClassVar, Self"
    new = (
        "from typing import Any, ClassVar\n"
        "try:\n"
        "    from typing import Self\n"
        "except ImportError:\n"
        "    from typing_extensions import Self"
    )
    if old in text:
        path.write_text(text.replace(old, new, 1))


def patch_io_utils_py(vendor: Path) -> None:
    path = vendor / "utils" / "io_utils.py"
    text = path.read_text()
    old_def = "def deserialize_json_into_object[T: JsonLike](fpath: Path, obj: T) -> T:"
    new_def = "def deserialize_json_into_object(fpath: Path, obj: T) -> T:"
    if old_def not in text:
        return
    if "T = TypeVar" not in text:
        text = text.replace("from typing import Any", "from typing import Any, TypeVar", 1)
        anchor = (
            'JsonLike = str | int | float | bool | None | list["JsonLike"] | '
            'dict[str, "JsonLike"] | tuple["JsonLike", ...]\n'
        )
        text = text.replace(anchor, anchor + '\nT = TypeVar("T", bound=JsonLike)\n', 1)
    path.write_text(text.replace(old_def, new_def, 1))


def patch_pipeline_py(vendor: Path) -> None:
    path = vendor / "processor" / "pipeline.py"
    text = path.read_text()
    old = "class DataProcessorPipeline[TInput, TOutput](HubMixin):"
    new = "class DataProcessorPipeline(HubMixin, Generic[TInput, TOutput]):"
    if old in text:
        text = text.replace(
            "from typing import Any, TypedDict, TypeVar, cast",
            "from typing import Any, Generic, TypedDict, TypeVar, cast",
            1,
        )
        path.write_text(text.replace(old, new, 1))


def patch_motors_bus_py(vendor: Path) -> None:
    path = vendor / "motors" / "motors_bus.py"
    text = path.read_text()
    text = text.replace("type NameOrID = str | int\n", "NameOrID = str | int\n")
    text = text.replace("type Value = int | float\n", "Value = int | float\n")
    path.write_text(text)


def patch_policy_lazy_imports(vendor: Path) -> None:
    """Import configuration だけで全 policy の modeling を読み込まないよう lazy 化する。

    採点環境の transformers は PyPI 最新 (4.46.x) より古く、eo1 等の modeling が
    torch_compilable_check 等の未公開 API に依存して ImportError になる。
    """
    import re

    policies_dir = vendor / "policies"
    if not policies_dir.is_dir():
        return

    lazy_src_re = re.compile(r"^from \.(modeling_\w+|processor_\w+) import (.+)$")
    patched = 0

    for init_path in sorted(policies_dir.glob("*/__init__.py")):
        text = init_path.read_text()
        if "_LAZY_IMPORTS" in text:
            continue

        kept: list[str] = []
        lazy_imports: dict[str, str] = {}
        pending_module: str | None = None
        pending_names: list[str] = []

        def flush_pending() -> None:
            nonlocal pending_module, pending_names
            if pending_module is None:
                return
            for name in pending_names:
                name = name.strip()
                if name:
                    lazy_imports[name] = pending_module
            pending_module = None
            pending_names = []

        for line in text.splitlines():
            stripped = line.strip()
            if pending_module is not None:
                pending_names.extend(part.strip() for part in stripped.split(","))
                if ")" in stripped:
                    flush_pending()
                continue

            match = lazy_src_re.match(stripped)
            if not match:
                kept.append(line)
                continue

            module, rest = match.group(1), match.group(2).strip()
            if rest.startswith("("):
                pending_module = module
                inner = rest.lstrip("(").rstrip(",").strip()
                if inner and inner != ")":
                    pending_names.append(inner)
                if ")" in rest:
                    flush_pending()
                continue

            for name in rest.split(","):
                name = name.strip()
                if name:
                    lazy_imports[name] = module

        if not lazy_imports:
            continue

        lazy_block = [
            "",
            "_LAZY_IMPORTS = {",
            *[f'    "{name}": "{module}",' for name, module in sorted(lazy_imports.items())],
            "}",
            "",
            "def __getattr__(name: str):",
            "    if name in _LAZY_IMPORTS:",
            "        from importlib import import_module",
            "",
            "        module = import_module(f\".{_LAZY_IMPORTS[name]}\", __name__)",
            "        return getattr(module, name)",
            "    raise AttributeError(f\"module {__name__!r} has no attribute {name!r}\")",
        ]

        out: list[str] = []
        inserted = False
        for line in kept:
            if not inserted and line.strip().startswith("__all__"):
                out.extend(lazy_block)
                inserted = True
            out.append(line)
        if not inserted:
            out.extend(lazy_block)

        init_path.write_text("\n".join(out).rstrip() + "\n")
        patched += 1

    if patched:
        print(f"[patch] lazy policy imports applied to {patched} packages under {policies_dir}")


def patch_streaming_dataset_py(vendor: Path) -> None:
    path = vendor / "datasets" / "streaming_dataset.py"
    text = path.read_text()
    old = "class Backtrackable[T]:"
    new = "class Backtrackable(Generic[T]):"
    if old in text:
        text = text.replace(
            "from pathlib import Path\n",
            "from pathlib import Path\nfrom typing import Generic, TypeVar\n\nT = TypeVar(\"T\")\n",
            1,
        )
        path.write_text(text.replace(old, new, 1))


def main() -> None:
    vendor = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("vendor/lerobot")
    patch_video_py(vendor)
    patch_io_utils_py(vendor)
    patch_pipeline_py(vendor)
    patch_motors_bus_py(vendor)
    patch_streaming_dataset_py(vendor)
    patch_policy_lazy_imports(vendor)
    print(f"[patch] Python 3.10 patches applied to {vendor}")


if __name__ == "__main__":
    main()
