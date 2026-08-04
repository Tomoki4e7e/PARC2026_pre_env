#!/usr/bin/env bash
# 提出 zip を Python 3.10 採点環境向けに bundle する。
#
# lerobot==0.6.0 は PyPI で Python>=3.12 必須のため、LeRobot ソースを vendor/ に同梱する。
#
# 使い方:
#   bash examples/smolvla_libero_spatial_lora_local/bundle_submission.sh
#   bash examples/smolvla_libero_spatial_lora_local/bundle_submission.sh /path/to/output.zip
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SUB="$ROOT/submission_template"
LEROBOT_SRC="$SCRIPT_DIR/work/lerobot/src/lerobot"
VENDOR="$SUB/vendor/lerobot"
OUTPUT="${1:-$ROOT/submission.zip}"

if [ ! -d "$LEROBOT_SRC" ]; then
    echo "[bundle] ERROR: LeRobot ソースがありません。先に setup_lerobot.sh を実行してください。"
    exit 1
fi

echo "[bundle] vendor/lerobot を同期..."
rm -rf "$SUB/vendor"
mkdir -p "$SUB/vendor"
rsync -a --delete \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.git' \
    "$LEROBOT_SRC/" "$VENDOR/"

python3 "$SCRIPT_DIR/patch_lerobot_py310.py" "$VENDOR"

rm -rf "$SUB/__pycache__" "$SUB/vendor/**/__pycache__" 2>/dev/null || true
find "$SUB/vendor" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true

echo "[bundle] zip 作成: $OUTPUT"
rm -f "$OUTPUT"
(
    cd "$SUB"
    zip -r "$OUTPUT" \
        policy_server.py \
        smolvla_mypolicy.py \
        lerobot_bootstrap.py \
        requirements.txt \
        vendor/ \
        model_weights/ \
        -x '*/__pycache__/*' -x '*.pyc'
)

echo "[bundle] 完了: $(du -h "$OUTPUT" | cut -f1)  $OUTPUT"
echo ""
echo "検証 (静的):"
echo "  python validate_submission.py $OUTPUT --static"
echo "検証 (スモーク, SmolVLA 環境):"
echo "  bash examples/smolvla_libero_spatial_lora_local/validate_submission.sh $OUTPUT"
echo "検証 (parc2026 / Python 3.10 相当):"
echo "  source env.sh && python validate_submission.py $OUTPUT --install"
