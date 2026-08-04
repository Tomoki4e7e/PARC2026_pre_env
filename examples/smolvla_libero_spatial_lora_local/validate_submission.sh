#!/usr/bin/env bash
# SmolVLA 提出 zip の validate_submission.py を正しい Python 環境で実行する。
#
# parc2026 環境 (Python 3.10) では lerobot==0.6.0 が入らずスモークテストが失敗する。
# SmolVLA conda 環境 (Python 3.12) を使うこと。
#
# ★ --install は付けないこと（requirements.txt の pip install で torch が壊れる）。
#
# 使い方:
#   bash examples/smolvla_libero_spatial_lora_local/validate_submission.sh submission.zip
#   bash examples/smolvla_libero_spatial_lora_local/validate_submission.sh submission_template/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/env_smolvla.sh"

PYTHON="${SMOLVLA_CONDA_ENV}/bin/python"
if [ ! -x "$PYTHON" ]; then
    echo "[validate] ERROR: SmolVLA 環境が見つかりません: $PYTHON"
    exit 1
fi

TARGET="${1:-submission.zip}"
if [ ! -e "$ROOT/$TARGET" ] && [ ! -e "$TARGET" ]; then
    echo "[validate] ERROR: 対象が見つかりません: $TARGET"
    exit 1
fi

# --install が渡されたら警告して除去
args=("$@")
filtered=()
for a in "${args[@]}"; do
    if [ "$a" = "--install" ]; then
        echo "[validate] WARN: --install は SmolVLA 環境では torch を壊すため無視します。"
        echo "           依存は setup_lerobot.sh 済みであることを前提に検証します。"
        continue
    fi
    filtered+=("$a")
done

echo "[validate] Python: $PYTHON"
echo "[validate] 対象:   ${filtered[0]}"
cd "$ROOT"
exec "$PYTHON" validate_submission.py "${filtered[@]}"
