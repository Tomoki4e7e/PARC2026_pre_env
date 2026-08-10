#!/usr/bin/env python3
"""Bundle the lightweight SmolVLM processor/tokenizer assets for offline inference."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


ASSET_NAMES = (
    "added_tokens.json",
    "chat_template.json",
    "config.json",
    "merges.txt",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)
LOCAL_PROCESSOR = "model_weights/smolvla_spatial_lora/vlm_processor"


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: bundle_offline_processor.py MODEL_DIR VLM_SNAPSHOT")
    model_dir = Path(sys.argv[1]).resolve()
    snapshot = Path(sys.argv[2]).resolve()
    destination = model_dir / "vlm_processor"
    destination.mkdir(parents=True, exist_ok=True)

    for name in ASSET_NAMES:
        source = snapshot / name
        if not source.is_file():
            raise FileNotFoundError(f"offline processor asset missing: {source}")
        shutil.copy2(source, destination / name)

    config_path = model_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["vlm_model_name"] = LOCAL_PROCESSOR
    config_path.write_text(json.dumps(config, indent=4) + "\n", encoding="utf-8")

    preprocessor_path = model_dir / "policy_preprocessor.json"
    preprocessor = json.loads(preprocessor_path.read_text(encoding="utf-8"))
    found = False
    for step in preprocessor.get("steps", []):
        if step.get("registry_name") == "tokenizer_processor":
            step["config"]["tokenizer_name"] = LOCAL_PROCESSOR
            found = True
    if not found:
        raise RuntimeError("tokenizer_processor not found in policy_preprocessor.json")
    preprocessor_path.write_text(json.dumps(preprocessor, indent=2) + "\n", encoding="utf-8")
    print(f"[bundle] offline SmolVLM processor: {destination}")


if __name__ == "__main__":
    main()
