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
  augmentation)
    SRC="$WORK/smolvla_libero_plus_image_aug_merged"
    ;;
  perturbation_replay)
    SRC="$WORK/smolvla_track1_perturbation_replay_merged"
    ;;
  stove_collision)
    SRC="$WORK/smolvla_stove_corrective_merged"
    ;;
  augmentation_selected)
    SEL_JSON="$ROOT/results/augmentation_checkpoint_select/selection.json"
    if [ ! -f "$SEL_JSON" ]; then
      echo "[deploy] ERROR: Augmentation checkpoint/補間選定が未実行です: $SEL_JSON"
      exit 1
    fi
    SRC="$(python3 -c "import json; print(json.load(open(r'$SEL_JSON'))['selected_model_dir'])")"
    echo "[deploy] augmentation_selected → $(python3 -c "import json; print(json.load(open(r'$SEL_JSON'))['selected'])")"
    ;;
  phase3)
    SRC="$WORK/smolvla_libero_plus_phase3_holdout_merged"
    ;;
  phase4)
    # 統一選定の結果のみを使う（silent fallback 禁止）。
    # 未実行なら明示エラー。選定結果が stage2 なら stage2 を出す。
    SEL_JSON="$ROOT/results/phase4_unified_select/selection.json"
    if [ ! -f "$SEL_JSON" ]; then
      echo "[deploy] ERROR: Phase4 統一選定が未実行です。"
      echo "  先に: /home/cm-pc/.local/smolvla-conda-env/bin/python \\"
      echo "    $SCRIPT_DIR/run_phase4_unified_select.py"
      echo "  明示指定: phase4_v2 / phase4_from_stage2 / phase4_legacy / stage2"
      exit 1
    fi
    SRC="$(python3 -c "import json; print(json.load(open(r'$SEL_JSON'))['selected_model_dir'])")"
    echo "[deploy] phase4 → unified selection: $(python3 -c "import json; print(json.load(open(r'$SEL_JSON'))['selected'])")"
    ;;
  phase4_v2)
    SRC="$WORK/smolvla_libero_diverse_phase4_v2_merged"
    ;;
  phase4_from_stage2)
    SRC="$WORK/smolvla_libero_diverse_phase4_from_stage2_merged"
    ;;
  phase4_legacy)
    SRC="$WORK/smolvla_libero_diverse_phase4_merged"
    ;;
  phase4_selected)
    SRC="$WORK/smolvla_libero_phase4_selected_merged"
    ;;
  phase5)
    SRC="$WORK/smolvla_libero_plus_phase5_quality_merged"
    ;;
  *)
    echo "[deploy] ERROR: unknown variant '$VARIANT'"
    echo "  use: base, multisuite, spatial, stage2, augmentation, perturbation_replay, stove_collision,"
    echo "       augmentation_selected, phase3,"
    echo "       phase4 (unified selection), phase4_v2, phase4_from_stage2,"
    echo "       phase4_legacy, phase4_selected, phase5"
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
rsync -aL --delete \
    --exclude 'train_config.json' \
    --exclude 'interpolation.json' \
    "$SRC/" "$DEST/"
# policy 実装は submission_template 側を正とする。examples/ の古い写しで上書きすると、
# instruction safety 等の提出修正が失われるため、ここでは重みだけを切り替える。

echo "[deploy] 完了 ($(du -sh "$DEST" | cut -f1))"
echo ""
echo "ローカル評価:"
echo "  bash $SCRIPT_DIR/run_policy_server.sh --port 8000"
echo "  source $ROOT/env.sh && python -m pipeline --server-url http://localhost:8000 --track track1 --n-episodes 20"
