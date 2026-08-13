#!/usr/bin/env bash
# Expanded-base-first curriculum and a like-for-like comparison with legacy order.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env_smolvla.sh"
PYTHON="${SMOLVLA_CONDA_ENV}/bin/python"
WORK="$SCRIPT_DIR/work"
RESULTS="$ROOT/results/curriculum_order_ab"
EXPANDED_DATA="$WORK/datasets/libero_expanded_multisuite_v1"
EXPANDED_BASE="$WORK/smolvla_curriculum_expanded_base_merged"
SPATIAL_DATA="$WORK/datasets/libero_spatial_failure_corrective_v1"
SPATIAL_MODEL="$WORK/smolvla_spatial_failure_corrective_merged"
TRACK1_DATA="$WORK/datasets/libero_track1_with_expanded_replay_v1"
FINAL_MODEL="$WORK/smolvla_track1_with_expanded_replay_merged"
LEGACY_MODEL="$WORK/smolvla_track1_perturbation_replay_merged"

cd "$ROOT"
mkdir -p "$RESULTS"

if [ ! -f "$EXPANDED_DATA/meta/info.json" ]; then
    "$PYTHON" "$SCRIPT_DIR/build_expanded_multisuite_dataset.py"
fi

# Stage 1: initial SmolVLA -> expanded 580-episode base.
if [ ! -f "$EXPANDED_BASE/model.safetensors" ]; then
    "$PYTHON" "$SCRIPT_DIR/train.py" --curriculum-expanded-base \
        --log-file "$WORK/curriculum_expanded_base_train.log"
fi

# Both orders use the exact same deterministic 46-task manifest and protocol.
if [ ! -f "$RESULTS/legacy/summary.json" ]; then
    "$PYTHON" "$SCRIPT_DIR/eval_expanded_generalization.py" \
        --model-dir "$LEGACY_MODEL" --disable-residual --include-spatial-clean \
        --n-episodes 2 --seed 42 --port 8040 --result-root "$RESULTS/legacy"
fi
if [ ! -f "$RESULTS/expanded_base/summary.json" ]; then
    "$PYTHON" "$SCRIPT_DIR/eval_expanded_generalization.py" \
        --model-dir "$EXPANDED_BASE" --disable-residual --include-spatial-clean \
        --n-episodes 2 --seed 42 --port 8041 --result-root "$RESULTS/expanded_base"
fi

# Stage 2 is optional: its dataset contains only failed canonical Spatial tasks.
if [ ! -f "$WORK/spatial_failure_corrective_manifest.json" ]; then
    "$PYTHON" "$SCRIPT_DIR/build_curriculum_datasets.py" spatial \
        --eval-summary "$RESULTS/expanded_base/summary.json"
fi
SELECTED_BASE="$EXPANDED_BASE"
SPATIAL_ARGS=()
if grep -q '"spatial_corrective_required": true' "$WORK/spatial_failure_corrective_manifest.json"; then
    if [ ! -f "$SPATIAL_DATA/meta/info.json" ]; then
        echo "Spatial corrective manifest requires training but dataset is missing" >&2
        exit 1
    fi
    if [ ! -f "$SPATIAL_MODEL/model.safetensors" ]; then
        PARC_CURRICULUM_BASE_MODEL_DIR="$EXPANDED_BASE" \
            "$PYTHON" "$SCRIPT_DIR/train.py" --spatial-corrective \
            --log-file "$WORK/spatial_failure_corrective_train.log"
    fi
    if [ ! -f "$RESULTS/spatial_corrective/summary.json" ]; then
        "$PYTHON" "$SCRIPT_DIR/eval_expanded_generalization.py" \
            --model-dir "$SPATIAL_MODEL" --disable-residual --include-spatial-clean \
            --n-episodes 2 --seed 42 --port 8042 --result-root "$RESULTS/spatial_corrective"
    fi
    SPATIAL_ARGS=(--spatial-corrective "$RESULTS/spatial_corrective/summary.json")
fi

"$PYTHON" "$SCRIPT_DIR/compare_curriculum_order.py" \
    --legacy "$RESULTS/legacy/summary.json" \
    --expanded-base "$RESULTS/expanded_base/summary.json" \
    "${SPATIAL_ARGS[@]}" --output "$RESULTS/pre_track1_gate.json"
if grep -q '"selected_pre_track1": "spatial_corrective"' "$RESULTS/pre_track1_gate.json"; then
    SELECTED_BASE="$SPATIAL_MODEL"
fi

# Stage 3: 170 weak-task episodes + 43 broad replay episodes (20.2%).
if [ ! -f "$TRACK1_DATA/meta/info.json" ]; then
    "$PYTHON" "$SCRIPT_DIR/build_curriculum_datasets.py" track1
fi
if [ ! -f "$FINAL_MODEL/model.safetensors" ]; then
    PARC_CURRICULUM_BASE_MODEL_DIR="$SELECTED_BASE" \
        "$PYTHON" "$SCRIPT_DIR/train.py" --track1-replay \
        --log-file "$WORK/track1_with_expanded_replay_train.log"
fi
if [ ! -f "$RESULTS/new_final/summary.json" ]; then
    "$PYTHON" "$SCRIPT_DIR/eval_expanded_generalization.py" \
        --model-dir "$FINAL_MODEL" --disable-residual --include-spatial-clean \
        --n-episodes 2 --seed 42 --port 8043 --result-root "$RESULTS/new_final"
fi

"$PYTHON" "$SCRIPT_DIR/compare_curriculum_order.py" \
    --legacy "$RESULTS/legacy/summary.json" \
    --expanded-base "$RESULTS/expanded_base/summary.json" \
    "${SPATIAL_ARGS[@]}" --new-final "$RESULTS/new_final/summary.json" \
    --output "$RESULTS/comparison.json"

echo "Curriculum A/B complete: $RESULTS/comparison.json"
