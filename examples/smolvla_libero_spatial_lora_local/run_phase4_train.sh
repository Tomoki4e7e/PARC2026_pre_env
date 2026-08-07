#!/usr/bin/env bash
# Phase 4: LIBERO-90 多様タスク + 既知40 replay LoRA 学習
#
# 事前条件:
#   python examples/smolvla_libero_spatial_lora_local/prepare_diverse_retrain_data.py --prepare
#
# 使い方:
#   bash examples/smolvla_libero_spatial_lora_local/run_phase4_train.sh
#   bash examples/smolvla_libero_spatial_lora_local/run_phase4_train.sh --skip-merge
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

SKIP_MERGE=0
for arg in "$@"; do
    case "$arg" in
        --skip-merge) SKIP_MERGE=1 ;;
        *)
            echo "[phase4] ERROR: unknown argument: $arg"
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

if [ ! -f "$SCRIPT_DIR/work/diverse_retrain_manifest.json" ]; then
    echo "[phase4] ERROR: manifest がありません。先に prepare_diverse_retrain_data.py --prepare を実行してください。"
    exit 1
fi
if [ ! -f "$SCRIPT_DIR/work/datasets/libero_diverse_retrain/meta/info.json" ]; then
    echo "[phase4] ERROR: 集約データセットがありません。先に prepare_diverse_retrain_data.py --prepare を実行してください。"
    exit 1
fi

echo "[phase4] 1/2 camera key を SmolVLA 向けに確認・整列"
python "$SCRIPT_DIR/align_diverse_dataset_cameras.py"

echo "[phase4] 2/2 LoRA 学習（${PHASE4_STEPS:-v2} steps）"
TRAIN_ARGS=(--phase4)
if [ "$SKIP_MERGE" -eq 1 ]; then
    TRAIN_ARGS+=(--skip-merge)
fi
python "$SCRIPT_DIR/train.py" "${TRAIN_ARGS[@]}"

echo "[phase4] 完了"
echo ""
echo "次: proxy 基準で checkpoint 選定（推奨）"
echo "  python $SCRIPT_DIR/run_phase4_proxy_select.py --stage2-baseline"
echo "参考: OOD-90 early stopping"
echo "  python $SCRIPT_DIR/run_phase4_ood_early_stop.py"
