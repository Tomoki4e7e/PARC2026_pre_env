#!/usr/bin/env bash
# LeRobot インストール後に torch が cu130 等へ上がり GPU が使えなくなった場合の修復スクリプト。
#
# 使い方:
#   source examples/smolvla_libero_spatial_lora_local/env_smolvla.sh
#   conda activate ~/.local/smolvla-conda-env
#   bash examples/smolvla_libero_spatial_lora_local/repair_torch.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env_smolvla.sh"

PY="${PYTHON:-python3}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"

echo "[repair] 現在: $("$PY" -c "import torch; print(torch.__version__)" 2>/dev/null || echo 'torch 未インストール')"

echo "[repair] torch 2.6.0+cu124 を再インストール..."
"$PY" -m pip install \
    "torch==2.6.0" \
    "torchvision==0.21.0" \
    "torchaudio==2.6.0" \
    --index-url "$TORCH_INDEX"

"$PY" - <<'PY'
import torch
print(f"torch:  {torch.__version__}")
if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA がまだ利用できません。\n"
        "  nvidia-smi でドライバを確認し、CUDA 12.x 対応の PyTorch (cu124) を使用してください。"
    )
print(f"GPU:    {torch.cuda.get_device_name(0)}")
PY

echo "[repair] 完了。"
