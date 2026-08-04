#!/usr/bin/env bash
# 学習用のキャッシュ・一時ファイルを robot_ssd（リポジトリ配下 work/）へ向ける。
#
# NTFS (fuseblk) 上では conda / venv の symlink が使えないため、
# Python 環境本体は ext4（ホーム等）に置き、大容量データだけ robot_ssd を使う。
#
# 使い方（毎回）:
#   source examples/smolvla_libero_spatial_lora_local/env_smolvla.sh
set -euo pipefail

_SMOLVLA_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export SMOLVLA_ROOT="$_SMOLVLA_ENV_DIR"
export SMOLVLA_WORK_DIR="${SMOLVLA_WORK_DIR:-$_SMOLVLA_ENV_DIR/work}"

# ---- 大容量データは robot_ssd (work/) へ ----
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$SMOLVLA_WORK_DIR/pip-cache}"
export TMPDIR="${TMPDIR:-$SMOLVLA_WORK_DIR/tmp}"
export HF_HOME="${HF_HOME:-$SMOLVLA_WORK_DIR/hf_cache}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$SMOLVLA_WORK_DIR/lerobot_cache}"
export TORCH_HOME="${TORCH_HOME:-$SMOLVLA_WORK_DIR/torch_home}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$SMOLVLA_WORK_DIR/xdg-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_DISABLE_SYMLINKS_WARNING="${HF_HUB_DISABLE_SYMLINKS_WARNING:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

mkdir -p \
    "$SMOLVLA_WORK_DIR" \
    "$PIP_CACHE_DIR" \
    "$TMPDIR" \
    "$HF_HOME" \
    "$HF_LEROBOT_HOME" \
    "$TORCH_HOME" \
    "$XDG_CACHE_HOME"

# ---- Python 環境は ext4 上（NTFS では symlink 不可）----
_smolvla_fs_type="$(df -T "$SMOLVLA_WORK_DIR" 2>/dev/null | tail -1 | awk '{print $2}')"
if [[ "$_smolvla_fs_type" == "fuseblk" || "$_smolvla_fs_type" == "ntfs" || "$_smolvla_fs_type" == "ntfs3" ]]; then
    export SMOLVLA_ON_NTFS=1
    export SMOLVLA_CONDA_ENV="${SMOLVLA_CONDA_ENV:-$HOME/.local/smolvla-conda-env}"
else
    export SMOLVLA_ON_NTFS=0
    export SMOLVLA_CONDA_ENV="${SMOLVLA_CONDA_ENV:-$SMOLVLA_WORK_DIR/conda-env}"
fi

_smolvla_disk_data="$(df -h "$SMOLVLA_WORK_DIR" 2>/dev/null | tail -1 | awk '{print $1 "  avail=" $4 "  use=" $5 "  mount=" $6}')"
_smolvla_env_ref="${SMOLVLA_CONDA_ENV}"
if [ ! -e "$_smolvla_env_ref" ]; then
    _smolvla_env_ref="$(dirname "$_smolvla_env_ref")"
fi
_smolvla_disk_env="$(df -h "$_smolvla_env_ref" 2>/dev/null | tail -1 | awk '{print $1 "  avail=" $4 "  use=" $5 "  mount=" $6}')"

echo "[smolvla] work (cache/data): $SMOLVLA_WORK_DIR"
echo "[smolvla]   $_smolvla_disk_data"
echo "[smolvla] conda env (ext4):     $SMOLVLA_CONDA_ENV"
echo "[smolvla]   $_smolvla_disk_env"
echo "[smolvla] pip cache:            $PIP_CACHE_DIR"
echo "[smolvla] tmp:                  $TMPDIR"
echo "[smolvla] HF cache:             $HF_HOME"
if [[ "${SMOLVLA_ON_NTFS:-0}" == "1" ]]; then
    echo "[smolvla] NOTE: robot_ssd は NTFS のため Python 環境は ext4（ホーム）に配置します。"
    echo "[smolvla]       pip/HF キャッシュと学習出力は robot_ssd を使用します。"
fi
