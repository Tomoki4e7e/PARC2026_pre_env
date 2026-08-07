#!/usr/bin/env python3
"""集約データセットのカメラキーを SmolVLA preprocessor 向けに整列する。

nvidia/LIBERO_LeRobot_v3 は ``observation.images.image`` /
``observation.images.wrist_image`` を使うが、SmolVLA の pretrained
preprocessor は ``front`` / ``wrist`` を期待する。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import config

CAMERA_RENAME = {
    "observation.images.image": "observation.images.front",
    "observation.images.wrist_image": "observation.images.wrist",
}


def align_dataset(root: Path) -> bool:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"dataset info がありません: {info_path}")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    features = info["features"]
    if (
        "observation.images.front" in features
        and "observation.images.image" not in features
    ):
        print(f"[align] already SmolVLA-ready: {root}")
        return False

    videos_root = root / "videos"
    for old, new in CAMERA_RENAME.items():
        src = videos_root / old
        dst = videos_root / new
        if src.exists() and not dst.exists():
            src.rename(dst)
            print(f"[align] renamed video dir {old} -> {new}")
        elif not dst.exists():
            raise FileNotFoundError(f"video dir missing: {src}")

    info["features"] = {
        CAMERA_RENAME.get(key, key): value for key, value in features.items()
    }
    info_path.write_text(json.dumps(info, indent=4) + "\n", encoding="utf-8")
    print("[align] updated meta/info.json")

    stats_path = root / "meta" / "stats.json"
    if stats_path.is_file():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        stats = {CAMERA_RENAME.get(key, key): value for key, value in stats.items()}
        stats_path.write_text(json.dumps(stats, indent=4) + "\n", encoding="utf-8")
        print("[align] updated meta/stats.json")

    for parquet_path in sorted((root / "meta" / "episodes").rglob("*.parquet")):
        frame = pd.read_parquet(parquet_path)
        rename_cols: dict[str, str] = {}
        for column in frame.columns:
            renamed = column
            for old, new in CAMERA_RENAME.items():
                renamed = renamed.replace(old, new)
            if renamed != column:
                rename_cols[column] = renamed
        if rename_cols:
            frame = frame.rename(columns=rename_cols)
            frame.to_parquet(parquet_path, index=False)
            print(f"[align] updated {parquet_path.name} ({len(rename_cols)} cols)")

    return True


def main() -> None:
    root = config.PHASE4_DATASET_ROOT
    changed = align_dataset(root)
    print(f"[align] done (changed={changed})")


if __name__ == "__main__":
    main()
