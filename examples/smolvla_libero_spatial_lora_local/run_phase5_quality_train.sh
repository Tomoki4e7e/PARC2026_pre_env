#!/usr/bin/env bash
# Phase5: 品質フィルタ済みデモでの追加 LoRA +（任意）複合 proxy 選定
#
# 使い方:
#   bash examples/smolvla_libero_spatial_lora_local/run_phase5_quality_train.sh
#   bash examples/smolvla_libero_spatial_lora_local/run_phase5_quality_train.sh --with-proxy-select
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env_smolvla.sh"

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${SMOLVLA_CONDA_ENV:-$HOME/.local/smolvla-conda-env}"
fi

WITH_PROXY=0
for arg in "$@"; do
  case "$arg" in
    --with-proxy-select) WITH_PROXY=1 ;;
    *)
      echo "unknown arg: $arg" >&2
      exit 1
      ;;
  esac
done

cd "$ROOT"
PYTHON="$(command -v python)"

echo "[phase5] 1/3 score episode quality (jerk/SPARC/orientation)"
"$PYTHON" "$SCRIPT_DIR/score_episode_quality.py" --phase5

echo "[phase5] 2/3 LoRA train (quality-filtered demos)"
"$PYTHON" "$SCRIPT_DIR/train.py" --phase5 --skip-merge

echo "[phase5] train finished. checkpoints under:"
echo "  $SCRIPT_DIR/work/outputs/smolvla_libero_plus_phase5_quality_lora/checkpoints"

if [ "$WITH_PROXY" -eq 1 ]; then
  echo "[phase5] 3/3 proxy select (success+collision+smoothness)"
  "$PYTHON" "$SCRIPT_DIR/run_phase4_proxy_select.py" --phase5 --stage2-baseline
else
  echo "[phase5] skip proxy select (pass --with-proxy-select to run)"
  echo "  python $SCRIPT_DIR/run_phase4_proxy_select.py --phase5"
fi
