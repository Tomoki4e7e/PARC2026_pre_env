#!/usr/bin/env bash
# 学習用 conda 環境を作成し PyTorch をインストールする。
#
# robot_ssd (NTFS) では Python 環境を置けないため:
#   - conda 環境本体 → ext4（~/.local/smolvla-conda-env）
#   - pip/HF キャッシュ・学習出力 → robot_ssd（work/）
#
# 使い方:
#   bash examples/smolvla_libero_spatial_lora_local/create_env.sh
#   bash examples/smolvla_libero_spatial_lora_local/create_env.sh --clean   # 容量確保
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env_smolvla.sh"

# ダウンロードキャッシュは必ず robot_ssd へ（既存 env var を上書き）
export PIP_CACHE_DIR="$SMOLVLA_WORK_DIR/pip-cache"
export TMPDIR="$SMOLVLA_WORK_DIR/tmp"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"

TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"
DO_CLEAN=0
for arg in "$@"; do
    case "$arg" in
        --clean) DO_CLEAN=1 ;;
    esac
done

if ! command -v conda >/dev/null 2>&1; then
    echo "[create_env] conda が見つかりません。"
    exit 1
fi

_eval_conda_hook() {
    if [ -n "${CONDA_EXE:-}" ] && [ -f "${CONDA_EXE%bin/conda}etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1091
        source "${CONDA_EXE%bin/conda}etc/profile.d/conda.sh"
    elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1091
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1091
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    else
        echo "[create_env] conda.sh が見つかりません。"
        exit 1
    fi
}
_eval_conda_hook

# ext4 側の空き容量を確認（PyTorch + CUDA ライブラリで ~8 GB 必要）
_env_ref="$SMOLVLA_CONDA_ENV"
if [ ! -e "$_env_ref" ]; then
    _env_ref="$(dirname "$_env_ref")"
fi
_env_avail_kb="$(df -k "$_env_ref" 2>/dev/null | tail -1 | awk '{print $4}')"
_env_avail_gb=$(( _env_avail_kb / 1024 / 1024 ))
echo "[create_env] ext4 空き: ${_env_avail_gb} GB（PyTorch インストールに ~8 GB 推奨）"

if [ "$_env_avail_gb" -lt 8 ]; then
    echo "[create_env] 警告: ext4 の空きが少ないです。"
    if [ "$DO_CLEAN" -eq 0 ]; then
        echo "[create_env] 次を試してください:"
        echo "  bash $0 --clean    # conda / pip キャッシュ削除で容量確保"
        echo ""
        echo "  または手動で:"
        echo "    conda clean -a -y"
        echo "    rm -rf ~/.cache/pip"
    fi
fi

if [ "$DO_CLEAN" -eq 1 ]; then
    echo "[create_env] ディスク容量確保中..."
    conda clean -a -y || true
    rm -rf "$HOME/.cache/pip" 2>/dev/null || true
    rm -rf "$PIP_CACHE_DIR"/* 2>/dev/null || true
    _env_avail_kb="$(df -k "$_env_ref" 2>/dev/null | tail -1 | awk '{print $4}')"
    _env_avail_gb=$(( _env_avail_kb / 1024 / 1024 ))
    echo "[create_env] 清理後 ext4 空き: ${_env_avail_gb} GB"
fi

if [ ! -d "$SMOLVLA_CONDA_ENV" ]; then
    echo "[create_env] conda 環境を ext4 上に作成: $SMOLVLA_CONDA_ENV"
    conda create --prefix "$SMOLVLA_CONDA_ENV" python=3.12 -y
else
    echo "[create_env] 既存環境を使用: $SMOLVLA_CONDA_ENV"
fi

conda activate "$SMOLVLA_CONDA_ENV"

echo "[create_env] PyTorch (CUDA) をインストール..."
echo "             packages → $SMOLVLA_CONDA_ENV (ext4)"
echo "             cache    → $PIP_CACHE_DIR (robot_ssd)"
echo "             tmp      → $TMPDIR (robot_ssd)"
python -m pip install --upgrade pip
python -m pip install torch --index-url "$TORCH_INDEX"

python - <<'PY'
import sys
import torch
print(f"Python: {sys.version.split()[0]}")
print(f"Torch:  {torch.__version__}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA が利用できません。")
print(f"GPU:    {torch.cuda.get_device_name(0)}")
PY

echo ""
echo "[create_env] 完了。"
echo ""
echo "次のコマンド:"
echo "  source examples/smolvla_libero_spatial_lora_local/env_smolvla.sh"
echo "  conda activate $SMOLVLA_CONDA_ENV"
echo "  bash examples/smolvla_libero_spatial_lora_local/setup_lerobot.sh"
