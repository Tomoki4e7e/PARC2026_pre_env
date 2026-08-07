#!/usr/bin/env bash
# マージ済みモデルを submission_template へ配置する。
#
# 使い方:
#   bash examples/smolvla_libero_spatial_lora_local/deploy_model.sh              # multisuite（Phase 2 既定）
#   bash examples/smolvla_libero_spatial_lora_local/deploy_model.sh multisuite
#   bash examples/smolvla_libero_spatial_lora_local/deploy_model.sh spatial
#   bash examples/smolvla_libero_spatial_lora_local/deploy_model.sh stage2
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORK="$SCRIPT_DIR/work"
SUB="$ROOT/submission_template"
DEST="$SUB/model_weights/smolvla_spatial_lora"
VARIANT="${1:-multisuite}"

case "$VARIANT" in
  base)
    # LoRA 追加学習前の素の lerobot/smolvla_libero_plus（汎化の基準点）
    REV="$(sed -n 's/^BASE_MODEL_REVISION = "\(.*\)"/\1/p' "$SCRIPT_DIR/config.py")"
    SRC="$WORK/hf_cache/hub/models--lerobot--smolvla_libero_plus/snapshots/$REV"
    ;;
  multisuite)
    SRC="$WORK/smolvla_libero_plus_multisuite_merged"
    ;;
  spatial)
    SRC="$WORK/smolvla_libero_plus_spatial_lora_merged"
    ;;
  stage2)
    SRC="$WORK/smolvla_libero_plus_track1_stage2_merged"
    ;;
  phase3)
    SRC="$WORK/smolvla_libero_plus_phase3_holdout_merged"
    ;;
  phase4)
    # v2（proxy 選定・normalizer 固定）を優先。無ければ from_stage2 / legacy へフォールバック。
    SRC="$WORK/smolvla_libero_diverse_phase4_v2_merged"
    if [ ! -f "$SRC/config.json" ]; then
      SRC="$WORK/smolvla_libero_diverse_phase4_from_stage2_merged"
    fi
    if [ ! -f "$SRC/config.json" ]; then
      SRC="$WORK/smolvla_libero_diverse_phase4_merged"
    fi
    ;;
  phase4_v2)
    SRC="$WORK/smolvla_libero_diverse_phase4_v2_merged"
    ;;
  phase4_legacy)
    SRC="$WORK/smolvla_libero_diverse_phase4_merged"
    ;;
  *)
    echo "[deploy] ERROR: unknown variant '$VARIANT'"
    echo "  use: base, multisuite, spatial, stage2, phase3, phase4, phase4_v2, phase4_legacy"
    exit 1
    ;;
esac

if [ ! -f "$SRC/config.json" ] || [ ! -f "$SRC/model.safetensors" ]; then
    echo "[deploy] ERROR: merged model not found: $SRC"
    echo "  Phase 2 (multisuite): bash $SCRIPT_DIR/run_phase2_train.sh"
    echo "  または: python $SCRIPT_DIR/merge_lora.py --stage1"
    exit 1
fi

echo "[deploy] variant=$VARIANT"
echo "[deploy]   src:  $SRC"
echo "[deploy]   dest: $DEST"

mkdir -p "$SUB/model_weights"
if [ -d "$DEST" ]; then
    # NTFS 等で rm -rf が失敗する環境向け
    find "$DEST" -mindepth 1 -delete 2>/dev/null || rm -rf "$DEST" 2>/dev/null || true
fi
if [ ! -d "$DEST" ]; then
    mkdir -p "$DEST"
fi
rsync -aL --delete --exclude 'train_config.json' "$SRC/" "$DEST/"
# policy 実装は submission_template 側を正とする。examples/ の古い写しで上書きすると、
# instruction safety 等の提出修正が失われるため、ここでは重みだけを切り替える。

echo "[deploy] 完了 ($(du -sh "$DEST" | cut -f1))"
echo ""
echo "ローカル評価:"
echo "  bash $SCRIPT_DIR/run_policy_server.sh --port 8000"
echo "  source $ROOT/env.sh && python -m pipeline --server-url http://localhost:8000 --track track1 --n-episodes 20"
