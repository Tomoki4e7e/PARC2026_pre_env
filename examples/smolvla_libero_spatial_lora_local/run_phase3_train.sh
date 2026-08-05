#!/usr/bin/env bash
# Phase 3: hold-out object/goal 弱点 LoRA（stage2 merged ベース, ~230 ep）
#
# hold-out n=20 で弱かった 5 ベースタスクを追加学習する:
#   object: alphabet soup, butter, ketchup
#   goal:   open middle drawer, turn on stove
#
# 使い方:
#   bash examples/smolvla_libero_spatial_lora_local/run_phase3_train.sh
#   bash examples/smolvla_libero_spatial_lora_local/run_phase3_train.sh --skip-train
#   bash examples/smolvla_libero_spatial_lora_local/run_phase3_train.sh --download-only
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
            echo "[phase3] ERROR: unknown argument: $arg"
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

if [ ! -f "$SCRIPT_DIR/work/smolvla_libero_plus_track1_stage2_merged/model.safetensors" ]; then
    echo "[phase3] ERROR: stage2 merged が必要です。先に --track1-stage2 学習を完了してください。"
    exit 1
fi

echo "[phase3] 1/4 task 名突合"
python "$SCRIPT_DIR/list_dataset_tasks.py" --phase3

echo "[phase3] 2/4 データセット取得（hold-out 弱点 5 タスク, ~230 ep）"
python "$SCRIPT_DIR/download_dataset.py" --phase3

if [ "$DOWNLOAD_ONLY" -eq 1 ]; then
    echo "[phase3] --download-only: ここで終了"
    exit 0
fi

if [ "$SKIP_TRAIN" -eq 0 ]; then
    echo "[phase3] 3/4 hold-out 弱点 LoRA 学習 + マージ（10k steps, stage2 ベース）"
    python "$SCRIPT_DIR/train.py" --phase3
else
    echo "[phase3] 3/4 マージのみ（--skip-train）"
    python "$SCRIPT_DIR/merge_lora.py" --phase3
fi

echo "[phase3] 4/4 submission_template へ phase3 モデルを配置"
bash "$SCRIPT_DIR/deploy_model.sh" phase3

echo "[phase3] 完了"
echo ""
echo "次: hold-out object/goal + in-set example 4 を n=20 で再評価"
echo "  bash $SCRIPT_DIR/run_policy_server.sh --port 8000"
echo "  source $ROOT/env.sh && python -m pipeline --server-url http://localhost:8000 --track track1 --n-episodes 20"
