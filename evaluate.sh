#!/bin/bash
# ============================================================
# Omnicampus 評価スクリプト
#
# Docker コンテナ上でこのスクリプト1つが呼ばれる。
# 参加者の submission.zip を展開し、ポリシーサーバーを起動、
# 評価パイプラインを実行して結果を JSON で出力する。
#
# 使い方 (Omnicampus):
#   ./evaluate.sh /path/to/submission.zip
#
# 使い方 (ローカルテスト):
#   ./evaluate.sh /path/to/submission.zip [--n-episodes 2] [--max-steps 10]
# ============================================================

set -euo pipefail

# --- 設定 ---
SUBMISSION_ZIP="${1:?Usage: $0 <submission.zip> [extra pipeline args...]}"
shift  # 残りの引数はパイプラインに渡す

PIPELINE_DIR="$(cd "$(dirname "$0")" && pwd)"
SUBMISSION_DIR="/tmp/submission_$$"
SERVER_PORT="${SERVER_PORT:-8000}"
SERVER_TIMEOUT=120  # サーバー起動待ちの上限（秒）
SERVER_PID=""

# --- クリーンアップ ---
cleanup() {
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[evaluate] ポリシーサーバーを停止 (PID=$SERVER_PID)"
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    rm -rf "$SUBMISSION_DIR"
}
trap cleanup EXIT

# --- ステップ 1: 提出物の展開 ---
echo "[evaluate] 提出物を展開: $SUBMISSION_ZIP"
mkdir -p "$SUBMISSION_DIR"
unzip -q "$SUBMISSION_ZIP" -d "$SUBMISSION_DIR"

# 必須ファイルの確認
if [ ! -f "$SUBMISSION_DIR/policy_server.py" ]; then
    echo "[evaluate] エラー: policy_server.py が見つかりません"
    exit 1
fi

# --- ステップ 2: 追加依存のインストール ---
if [ -f "$SUBMISSION_DIR/requirements.txt" ]; then
    echo "[evaluate] 追加依存をインストール"
    pip install -q -r "$SUBMISSION_DIR/requirements.txt"
fi

# --- ステップ 3: ポリシーサーバーの起動 ---
echo "[evaluate] ポリシーサーバーを起動 (port=$SERVER_PORT)"
cd "$SUBMISSION_DIR"
python policy_server.py --port "$SERVER_PORT" &
SERVER_PID=$!
cd "$PIPELINE_DIR"

# ヘルスチェックで起動を待つ
echo "[evaluate] サーバーの起動を待機..."
elapsed=0
while [ $elapsed -lt $SERVER_TIMEOUT ]; do
    if curl -s "http://localhost:$SERVER_PORT/health" > /dev/null 2>&1; then
        echo "[evaluate] サーバー起動確認 (${elapsed}秒)"
        break
    fi
    # サーバープロセスが死んでいないか確認
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[evaluate] エラー: ポリシーサーバーが起動に失敗しました"
        wait "$SERVER_PID" 2>/dev/null || true
        exit 1
    fi
    sleep 1
    elapsed=$((elapsed + 1))
done

if [ $elapsed -ge $SERVER_TIMEOUT ]; then
    echo "[evaluate] エラー: サーバーが ${SERVER_TIMEOUT}秒以内に起動しませんでした"
    exit 1
fi

# --- ステップ 4: 評価パイプライン実行 ---
echo "[evaluate] 評価パイプラインを実行"
python -m pipeline \
    --server-url "http://localhost:$SERVER_PORT" \
    --track track1 \
    "$@"

echo "[evaluate] 完了"
