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


def main() -> None:
    lerobot_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "work" / "lerobot"
    patch_train_utils_ntfs(lerobot_dir)


if __name__ == "__main__":
    main()
