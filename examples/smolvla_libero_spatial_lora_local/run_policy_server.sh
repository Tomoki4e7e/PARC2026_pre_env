#!/usr/bin/env bash
# SmolVLA 提出物のポリシーサーバーを安全に起動する。
#
# 重要: SmolVLA 環境で pip install -r requirements.txt を実行すると torch が
# 2.11.0 等に上がり CUDA ライブラリが壊れ、python policy_server.py 実行時に
# ターミナル/画面ごと落ちることがあります。このスクリプトは pip install を
# 行わず、torch の健全性を確認してから起動します。
#
# 使い方:
#   bash examples/smolvla_libero_spatial_lora_local/run_policy_server.sh
#   bash examples/smolvla_libero_spatial_lora_local/run_policy_server.sh --port 8000
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/env_smolvla.sh"

PYTHON="${SMOLVLA_CONDA_ENV}/bin/python"
if [ ! -x "$PYTHON" ]; then
    echo "[run_policy_server] ERROR: SmolVLA 環境が見つかりません: $PYTHON"
    echo "  初回: bash examples/smolvla_libero_spatial_lora_local/create_env.sh"
    echo "        bash examples/smolvla_libero_spatial_lora_local/setup_lerobot.sh"
    exit 1
fi

LOG_DIR="${SMOLVLA_WORK_DIR}/logs"
mkdir -p "$LOG_DIR"
LOG="${POLICY_SERVER_LOG:-$LOG_DIR/policy_server.log}"

_check_torch() {
    "$PYTHON" - <<'PY'
import sys
try:
    import torch
except Exception as exc:
    print(f"BROKEN: import failed: {exc}", file=sys.stderr)
    raise SystemExit(1)
version = torch.__version__
if not torch.cuda.is_available():
    print(
        f"BROKEN: torch {version} — CUDA 不可用\n"
        "  pip install -r requirements.txt で torch が上がった可能性があります。",
        file=sys.stderr,
    )
    raise SystemExit(1)
print(version)
PY
}

if ! torch_version="$(_check_torch 2>&1)"; then
    echo "[run_policy_server] torch が壊れています。repair_torch.sh を実行します..."
    echo "$torch_version"
    PYTHON="$PYTHON" bash "$SCRIPT_DIR/repair_torch.sh"
    torch_version="$(_check_torch)"
fi

PORT=8000
prev=""
for arg in "$@"; do
    if [ "$prev" = "--port" ]; then
        PORT="$arg"
    fi
    prev="$arg"
done

if ss -tln 2>/dev/null | grep -q ":${PORT} "; then
    echo "[run_policy_server] ERROR: port ${PORT} は既に使用中です。"
    echo "  既存プロセス: $(ss -tlnp 2>/dev/null | grep ":${PORT} " || true)"
    echo "  停止する場合: kill \$(ss -tlnp | grep ':${PORT} ' | sed -n 's/.*pid=\\([0-9]*\\).*/\\1/p')"
    echo "  別ポートで起動: bash $0 --port 8001"
    exit 1
fi

echo "[run_policy_server] Python: $PYTHON"
echo "[run_policy_server] torch:  $torch_version"
echo "[run_policy_server] port:   $PORT"
echo "[run_policy_server] ログ:   $LOG"
echo "[run_policy_server] 起動中... (Ctrl+C で停止)"

cd "$ROOT/submission_template"
exec "$PYTHON" policy_server.py "$@" 2>&1 | tee -a "$LOG"
