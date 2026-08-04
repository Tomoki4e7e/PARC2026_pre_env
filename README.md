# PARC 2026 — 予選配布環境

PARC 2026予選のための配布環境である。
本リポジトリを用いることで、参加者が実装したポリシーを example タスク上で
実行し、提出前にローカル環境で採点および動作確認を行うことができる。

本環境で実施できる作業は次のとおりである。
- 自身のポリシーを HTTP サーバーとして起動し、Track 1 の example タスクで評価する
- 提出物（zip）を、本番と同一の手順でエンドツーエンドに検証する
- 提出物の妥当性および動作を、提出前に自動チェックする
  （必須ファイルの有無、サーバーが起動して所定の応答を返すこと、
  各リクエストが制限時間内に完了することを確認する）

最初に参照すべきファイルは次のとおりである。
- **本マシン（NTFS + conda + GPU）向け起動手順:** [parc2026/README.md](parc2026/README.md)
- 提出物の作成方法・動作する最小実装: [submission_template/](submission_template/)
  （`policy_server.py` の `MyPolicy` は編集前でもそのまま動作する）
- 提出前チェック: [validate_submission.py](validate_submission.py)
- 学習の参考例: [examples/](examples/)（提出には必須ではない）

評価パイプラインおよび提出物チェックスクリプトは、本番採点のTrack 1と同じ評価処理・制約を
再現する。ただし、本番評価とは以下の点で異なる。

- 同梱されているのは公開されている example タスクのみである。本番の採点は、
  **公開されていないタスクを含む別のタスクセット**で実施される
- 出力されるのは成功率および軌道メトリクスの生値である。リーダーボードの順位を決定する
  スコア算出設定は含まれない
- 推論タイムアウト（[下記](#タイムアウト仕様)）および成功判定（[下記](#成功判定)）は
  本番と同一である

## 1. セットアップ

Python 3.10、git、unzip が必要である。[setup.sh](setup.sh) は本番の採点環境と同一の構築
（venv、ピン止めした依存、LIBERO-plus の取得とパッチ、アセットのダウンロードと配線）を
一括して実行する。setup.sh が取得・インストールする第三者製ソフトウェアとその
ライセンスは [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) を参照すること。

```bash
bash setup.sh     # 初回のみ（アセット取得を含めて 10〜20 分）
source env.sh     # 評価を実行するシェルで毎回実行する
```

> setup.sh は `~/.libero/config.yaml` を上書きする（既存の設定は `.bak` に退避される）。
> 既に LIBERO を使用しており元の設定に戻す場合は、`~/.libero/config.yaml.bak`
> を書き戻すこと。

### Docker を使用する場合（既存環境への影響を避ける場合はこちらを推奨する）

```bash
docker build -t parc2026 .
docker run -it --rm parc2026                     # 対話シェル（以降のコマンドをそのまま実行できる）
docker run --rm -v $PWD/my_submission.zip:/sub.zip parc2026 \
    python evaluate.py /sub.zip --n-episodes 2   # 提出 zip の一括評価
```

本番の採点コンテナと同一のベース（ubuntu 22.04 + osmesa レンダリング）を使用し、環境構築は
ローカルと同一の [setup.sh](setup.sh) がビルド時に実行される。

## 2. 評価を回す

```bash
# 1) 自身のポリシーサーバーを起動する（別ターミナル。テンプレートは編集前でも
#    ランダム action を返すので、まずそのまま起動して疎通確認できる）
python submission_template/policy_server.py --port 8000

# 2) 評価を実行する
python -m pipeline --server-url http://localhost:8000 --track track1 --n-episodes 2 --max-steps 600

# タスクを指定して評価する（example タスク名を指定する。存在しない名前は候補一覧つきでエラーとなる）
python -m pipeline --server-url http://localhost:8000 --track track1 --tasks <task_id>

# 提出 zip をエンドツーエンドで検証する（zip 展開 → 依存インストール → 評価まで自動実行）
python evaluate.py my_submission.zip --n-episodes 2
```

結果は `results/<submission_id>.json` に出力される。成功率、ステップ数、軌道メトリクス
（経路長、jerk、SPARC 等）の詳細が含まれる。

## 3. 提出前のチェック

提出物の妥当性（必須ファイル、zip 構造、エンドポイント）と、実際に起動して
動作すること（/health→/reset→/act が正常に応答し、応答が制限時間内であること）を検査する。

```bash
python validate_submission.py my_submission.zip            # 静的検査 + 起動スモークテスト
python validate_submission.py my_submission.zip --static   # 静的検査のみ（起動しない）
```

---

## 提出フォーマット

提出物は **HTTP ポリシーサーバー一式の zip** である。サーバーは次の 3 エンドポイントを
実装する（[テンプレート](submission_template/)を編集することで自動的に満たされる）。

| エンドポイント | 役割 |
|---|---|
| `GET /health` | 起動確認（200 を返すまで評価側がポーリングする） |
| `POST /reset` | エピソード開始（`instruction`, `seed` を JSON で受け取る） |
| `POST /act` | 観測（msgpack）→ action を返す。**float32 shape (7,)** `[dx, dy, dz, droll, dpitch, dyaw, gripper]` |

## 成功判定

本番の採点と同一の基準である。エピソードが成功と扱われるのは、**タスクのゴール条件を
満たし、かつ衝突が発生していない**場合のみである。

衝突は「操作対象以外の物体を動かしたか」で判定する。タスクが操作対象とする物体
（BDDL の `:obj_of_interest`）を除く全物体について、初期位置からの変位（xyz 各軸の
絶対値の和）を各ステップで監視し、その最大値が **1 mm** を超えた物体が 1 つでもあれば、
そのエピソードは失敗となる。

- 対象物体を掴んで動かすことは当然に許容される。判定対象は「それ以外の物体」である
- 変位は環境が落ち着いた時点（エピソード開始直前）の位置を基準とする
- 動かしてしまった物体を元の位置へ戻しても、変位の最大値で判定するため失敗のままである

## タイムアウト仕様

本番のTrack 1採点と同一の制約である。

**`/act`・`/reset` の 1 リクエストが 10 秒を超えた場合、そのトラックは失敗（error 扱い）
となり 0 点となる。** これは平均でも累積でもなく、1 回でも超過するとそのトラック全体が
失敗となる制約である。モデルの推論が 10 秒以内に収まることを必ず確認すること。

| 対象 | 上限 | 超えると |
|---|---|---|
| 推論: `/act`（および `/reset`）1 リクエスト | **10 秒** | そのトラックは error 扱いの 0 点 |
| サーバー起動（モデルロードを含む） | 既定 **120 秒**（`SERVER_TIMEOUT` で変更可） | 評価不能として終了 |

- タイムアウトは **HTTP リクエスト単位**である。平均・累積・エピソード単位の制限はない。
- アクションチャンクをサーバー内にキャッシュするモデルの場合、推論が実行される「重い」
  リクエストのみが上限の対象となる（実質的な制約は「チャンク 1 回分の推論 ≤ 10 秒」である）。
- [validate_submission.py](validate_submission.py) のスモークテストは、同一の 10 秒基準で
  レイテンシを警告する。提出前に必ず一度実行することを推奨する。

## ディレクトリ構成

| パス | 役割 |
|---|---|
| [pipeline/](pipeline/) | Track 1 評価パイプライン |
| [compe/t1/](compe/t1/) | Track 1 の example タスク定義 |
| [submission_template/](submission_template/) | 提出テンプレート（`policy_server.py` の `MyPolicy` のみ編集。編集前でも動作する） |
| [evaluate.py](evaluate.py) | Track 1 の zip 一括評価 |
| [validate_submission.py](validate_submission.py) | 提出物チェックスクリプト |
| [examples/](examples/) | 学習の参考例（SmolVLA の LoRA 追加学習ノートブック）。提出には必須ではない |
| [tests/](tests/) | ハーネスの単体テスト |
| [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) | setup.sh が取得する第三者製ソフトのライセンス表記 |
