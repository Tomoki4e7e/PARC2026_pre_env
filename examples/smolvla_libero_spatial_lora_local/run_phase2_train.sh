#!/usr/bin/env bash
# Phase 2: 第1段階 multi-suite LoRA 学習（spatial + object + goal, 740 ep 想定）
#
# 汎化の土台となる 30 ベースタスクを学習し、merged モデルを submission へ配置する。
#
# 使い方:
#   bash examples/smolvla_libero_spatial_lora_local/run_phase2_train.sh
#   bash examples/smolvla_libero_spatial_lora_local/run_phase2_train.sh --skip-train   # マージ+配置のみ
#   bash examples/smolvla_libero_spatial_lora_local/run_phase2_train.sh --download-only
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

SKIP_TRAIN=0
DOWNLOAD_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --skip-train) SKIP_TRAIN=1 ;;
        --download-only) DOWNLOAD_ONLY=1 ;;
        *)
            echo "[phase2] ERROR: unknown argument: $arg"
            exit 1
            ;;
    esac
done

cd "$ROOT"

if [ -f "$SCRIPT_DIR/env_smolvla.sh" ]; then
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/env_smolvla.sh"
fi

if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate ~/.local/smolvla-conda-env
fi

echo "[phase2] 1/4 task 名突合"
python "$SCRIPT_DIR/list_dataset_tasks.py"

echo "[phase2] 2/4 データセット取得（multi-suite, ~740 ep）"
python "$SCRIPT_DIR/download_dataset.py"

if [ "$DOWNLOAD_ONLY" -eq 1 ]; then
    echo "[phase2] --download-only: ここで終了"
    exit 0
fi

if [ "$SKIP_TRAIN" -eq 0 ]; then
    echo "[phase2] 3/4 multi-suite LoRA 学習 + マージ（40k steps, 約数時間）"
    python "$SCRIPT_DIR/train.py" --stage1
else
    echo "[phase2] 3/4 マージのみ（--skip-train）"
    python "$SCRIPT_DIR/merge_lora.py" --stage1
fi

echo "[phase2] 4/4 submission_template へ multisuite モデルを配置"
bash "$SCRIPT_DIR/deploy_model.sh" multisuite

echo "[phase2] 完了"
echo ""
echo "次: ローカル評価で example 4 タスク + 前回結果と比較"
echo "  bash $SCRIPT_DIR/run_policy_server.sh --port 8000"
echo "  source $ROOT/env.sh && python -m pipeline --server-url http://localhost:8000 --track track1 --n-episodes 20"
