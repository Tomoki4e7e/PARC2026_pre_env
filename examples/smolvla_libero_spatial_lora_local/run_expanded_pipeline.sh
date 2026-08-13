#!/usr/bin/env bash
# Reproducible expanded-data LoRA -> residual reannotation/retrain -> 3-slice eval.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env_smolvla.sh"
PYTHON="${SMOLVLA_CONDA_ENV}/bin/python"
WORK="$SCRIPT_DIR/work"
DATASET="$WORK/datasets/libero_expanded_multisuite_v1"
MODEL="$WORK/smolvla_expanded_multisuite_merged"
ANNOTATED="$WORK/rl_expanded_reannotated"
RESIDUAL="$WORK/residual_policy_expanded"

cd "$ROOT"

if [ ! -f "$DATASET/meta/info.json" ]; then
    "$PYTHON" "$SCRIPT_DIR/build_expanded_multisuite_dataset.py"
fi

"$PYTHON" "$SCRIPT_DIR/train.py" --expanded \
    --log-file "$WORK/expanded_multisuite_train.log"

"$PYTHON" "$SCRIPT_DIR/annotate_base_actions.py" \
    --input-dir "$WORK/rl_gate_540_deployed" \
    --input-dir "$WORK/rl_base_deployed_111/rl_independent_visual" \
    --input-dir "$WORK/rl_base_deployed_111/rl_collision_unsafe_visual" \
    --input-dir "$WORK/rl_base_deployed_111/rl_collision_safe_visual" \
    --output-dir "$ANNOTATED" --model-dir "$MODEL" \
    --confirm-local-generated-data

"$PYTHON" "$SCRIPT_DIR/train_residual_policy.py" \
    --input-dir "$ANNOTATED" --output-dir "$RESIDUAL" \
    --base-model-dir "$MODEL" --steps 5000 --seed 42

"$PYTHON" "$SCRIPT_DIR/eval_expanded_generalization.py" \
    --model-dir "$MODEL" --residual-dir "$RESIDUAL" \
    --n-episodes 2 --seed 42
