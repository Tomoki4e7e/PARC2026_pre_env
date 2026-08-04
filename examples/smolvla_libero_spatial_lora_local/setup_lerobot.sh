#!/usr/bin/env bash
# LeRobot v0.6.0 を学習用にセットアップする（Colab ノートブック §3 相当）
#
# 使い方:
#   bash examples/smolvla_libero_spatial_lora_local/setup_lerobot.sh
#
# 前提:
#   - Python 3.12 以上（lerobot v0.6.0 の推奨）
#   - NVIDIA GPU + CUDA ドライバ
#   - git, ffmpeg
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env_smolvla.sh"

WORK_DIR="$SMOLVLA_WORK_DIR"
LEROBOT_DIR="${WORK_DIR}/lerobot"
LEROBOT_TAG="v0.6.0"

PY="${PYTHON:-python3}"
if ! "$PY" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)'; then
    echo "[setup] Python 3.12 以上が必要です。例: conda create -n smolvla python=3.12"
    exit 1
fi

if ! "$PY" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
    echo "[setup] CUDA 対応 PyTorch が見つかりません。"
    echo "        先に create_env.sh を実行してください:"
    echo "          bash examples/smolvla_libero_spatial_lora_local/create_env.sh"
    exit 1
fi

mkdir -p "$WORK_DIR"

echo "[setup] システム依存（ffmpeg 等）を確認..."
if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq ffmpeg git unzip libgl1 libglib2.0-0 2>/dev/null || true
fi

echo "[setup] LeRobot ${LEROBOT_TAG} を取得..."
rm -rf "$LEROBOT_DIR"
git clone --quiet --depth 1 --branch "$LEROBOT_TAG" \
    https://github.com/huggingface/lerobot.git "$LEROBOT_DIR"

SMOLVLM_SOURCE="${LEROBOT_DIR}/src/lerobot/policies/smolvla/smolvlm_with_expert.py"
TRAIN_SCRIPT="${LEROBOT_DIR}/src/lerobot/scripts/lerobot_train.py"

# bf16 非対応 GPU では float16 へ切り替え
if ! "$PY" -c 'import torch; raise SystemExit(0 if torch.cuda.is_bf16_supported() else 1)'; then
    echo "[setup] bf16 非対応 GPU のため smolvlm_with_expert.py を fp16 にパッチ..."
    sed -i 's/torch_dtype="bfloat16",/torch_dtype="float16",/' "$SMOLVLM_SOURCE"
fi

# Colab 向けログ抑制パッチ
sed -i 's/logging.info(pformat(cfg.to_dict()))/logging.debug(pformat(cfg.to_dict()))/' "$TRAIN_SCRIPT"
sed -i 's/disable=inside_slurm(),/disable=True,/' "$TRAIN_SCRIPT"

echo "[setup] ローカル環境向けパッチを適用..."
"$PY" "${SCRIPT_DIR}/apply_lerobot_patches.py" "$LEROBOT_DIR"

echo "[setup] LeRobot を editable install..."
"$PY" -m pip uninstall -y lerobot torchao 2>/dev/null || true
"$PY" -m pip install --upgrade pip

# LeRobot の依存で torch が cu130 等に上がると、CUDA 12.5 ドライバで GPU が使えなくなる。
_saved_torch="$("$PY" -c "import torch; print(torch.__version__)" 2>/dev/null || true)"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"

"$PY" -m pip install -e "${LEROBOT_DIR}[training,smolvla,peft]"
"$PY" -m pip uninstall -y torchao 2>/dev/null || true

_new_torch="$("$PY" -c "import torch; print(torch.__version__)")"
if [[ "$_new_torch" != "$_saved_torch" ]] || [[ "$_new_torch" != *"+cu124"* ]]; then
    echo "[setup] torch が ${_new_torch} に更新されました。CUDA 12.4 ビルド (cu124) へ戻します..."
    "$PY" -m pip install \
        "torch==2.6.0" \
        "torchvision==0.21.0" \
        "torchaudio==2.6.0" \
        --index-url "$TORCH_INDEX"
fi

if ! "$PY" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
    echo "[setup] エラー: CUDA が利用できません。"
    "$PY" -c "import torch; print(f'  torch={torch.__version__}')"
    echo "  ドライバと PyTorch の CUDA バージョンが一致しているか確認してください。"
    echo "  修復: bash examples/smolvla_libero_spatial_lora_local/repair_torch.sh"
    exit 1
fi

echo "[setup] torch=$("$PY" -c "import torch; print(torch.__version__)")  GPU=$("$PY" -c "import torch; print(torch.cuda.get_device_name(0))")"

echo "[setup] 完了。次: python examples/smolvla_libero_spatial_lora_local/train.py"
