#!/usr/bin/env bash
# Compare the restored legacy-order model with the scored and expanded-first models.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env_smolvla.sh"
# shellcheck disable=SC1091
source "$ROOT/env.sh"

EVAL_PYTHON="/home/cm-pc/.venvs/parc2026-PARC2026_pre/bin/python"
RESULT_ROOT="$ROOT/results/legacy_order_comparison"
PORT="${PARC_COMPARISON_PORT:-8060}"
N_EPISODES="${PARC_COMPARISON_EPISODES:-10}"
MAX_STEPS="${PARC_COMPARISON_MAX_STEPS:-300}"

MODELS=(
  "score_033443:$SCRIPT_DIR/work/smolvla_track1_perturbation_replay_merged"
  "legacy_order_expanded:$SCRIPT_DIR/work/smolvla_expanded_multisuite_merged"
  "expanded_first_current:$SCRIPT_DIR/work/smolvla_track1_with_expanded_replay_merged"
)

mkdir -p "$RESULT_ROOT"

wait_for_server() {
  local attempt
  for attempt in $(seq 1 180); do
    if "$EVAL_PYTHON" -c \
      "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${PORT}/health', timeout=1).read()" \
      >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  return 1
}

stop_server() {
  if [ -n "${SERVER_PID:-}" ]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    unset SERVER_PID
  fi
}
trap stop_server EXIT

cd "$ROOT"
for entry in "${MODELS[@]}"; do
  label="${entry%%:*}"
  model="${entry#*:}"
  output="$RESULT_ROOT/$label"
  mkdir -p "$output/weak4"
  if [ ! -f "$model/model.safetensors" ]; then
    echo "missing model: $model" >&2
    exit 1
  fi

  export PARC_MODEL_DIR="$model"
  export PARC_RESIDUAL_POLICY=0
  export PARC_COLLISION_CRITIC=0
  export POLICY_SERVER_LOG="$output/server.log"
  bash "$SCRIPT_DIR/run_policy_server.sh" --port "$PORT" >"$output/server.stdout.log" 2>&1 &
  SERVER_PID=$!
  wait_for_server

  "$EVAL_PYTHON" -m pipeline \
    --server-url "http://127.0.0.1:$PORT" --track track1 --benchmark libero_t1 \
    --n-episodes "$N_EPISODES" --max-steps "$MAX_STEPS" --seed 42 \
    --output-dir "$output/weak4"

  "$EVAL_PYTHON" "$SCRIPT_DIR/run_proxy_eval.py" \
    --tag "legacy_order_comparison/$label/proxy" \
    --server-url "http://127.0.0.1:$PORT" --max-steps "$MAX_STEPS" --seed 42

  stop_server
done

"${SMOLVLA_CONDA_ENV}/bin/python" "$SCRIPT_DIR/summarize_legacy_order_comparison.py" \
  --root "$RESULT_ROOT"
