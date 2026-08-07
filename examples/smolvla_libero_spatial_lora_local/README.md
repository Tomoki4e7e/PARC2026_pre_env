# SmolVLA Spatial LoRA 学習（ローカル GPU 版）

Colab ノートブック [`smolvla_libero_spatial_lora.ipynb`](../smolvla_libero_spatial_lora.ipynb) と
**同一の学習条件**を、ローカル GPU 上で実行するスクリプト群です。

| ファイル | 役割 |
|---|---|
| [`setup_lerobot.sh`](setup_lerobot.sh) | LeRobot v0.6.0 + 学習依存のセットアップ |
| [`create_env.sh`](create_env.sh) | conda 環境 + PyTorch を robot_ssd 上に作成 |
| [`env_smolvla.sh`](env_smolvla.sh) | キャッシュ・一時ファイルを `work/` へ向ける |
| [`config.py`](config.py) | ハイパーパラメータ（ノートブックと同一） |
| [`train.py`](train.py) | LoRA 追加学習 |
| [`merge_lora.py`](merge_lora.py) | LoRA をベース重みへマージ |
| [`smolvla_mypolicy.py`](smolvla_mypolicy.py) | `MyPolicy` への組み込み参考実装 |
| [`download_dataset.py`](download_dataset.py) | 学習用データセット（50 エピソード）の事前ダウンロード |
| [`repair_torch.sh`](repair_torch.sh) | torch の CUDA 非互換を修復 |
| [`apply_lerobot_patches.py`](apply_lerobot_patches.py) | NTFS 向け LeRobot パッチ（`setup_lerobot.sh` から呼び出し） |

## 実行手順（最新）

リポジトリ直下（`/mnt/robot_ssd/PARC2026_pre`）で実行する。パスは環境に合わせて読み替える。

### A. 初回のみ（環境構築）

```bash
cd /mnt/robot_ssd/PARC2026_pre

# ext4 の空きが少ない場合
bash examples/smolvla_libero_spatial_lora_local/create_env.sh --clean

# conda 環境 + PyTorch（~/.local/smolvla-conda-env）
bash examples/smolvla_libero_spatial_lora_local/create_env.sh

# LeRobot v0.6.0 + ローカル向けパッチ
bash examples/smolvla_libero_spatial_lora_local/setup_lerobot.sh
```

### B. 作業のたびに（学習・推論）

```bash
cd /mnt/robot_ssd/PARC2026_pre
source examples/smolvla_libero_spatial_lora_local/env_smolvla.sh
conda activate ~/.local/smolvla-conda-env
```

CUDA 確認（`2.6.0+cu124 True` であること）:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

異常時（`cuda: False`）:

```bash
bash examples/smolvla_libero_spatial_lora_local/repair_torch.sh
```

### C. 学習

```bash
# （推奨）データセットを先に取得
python examples/smolvla_libero_spatial_lora_local/download_dataset.py

# LoRA 学習 + マージ（約 1〜2 時間）
python examples/smolvla_libero_spatial_lora_local/train.py
```

進捗確認:

```bash
tail -f examples/smolvla_libero_spatial_lora_local/work/train.log
```

学習末尾で symlink エラーが出ても、`checkpoints/003000/` があればマージのみ実行:

```bash
python examples/smolvla_libero_spatial_lora_local/merge_lora.py
```

出力: `work/smolvla_libero_plus_spatial_lora_merged/`

### D. 提出物（MyPolicy）への組み込み

```bash
mkdir -p submission_template/model_weights
cp -r examples/smolvla_libero_spatial_lora_local/work/smolvla_libero_plus_spatial_lora_merged \
      submission_template/model_weights/smolvla_spatial_lora
cp examples/smolvla_libero_spatial_lora_local/smolvla_mypolicy.py submission_template/
```

`submission_template/policy_server.py` の `MyPolicy` を編集し、
`submission_template/requirements.txt` に推論依存（後述 §3.3）を追記する。

### E. ローカル評価

**ターミナル 1**（学習環境 or GPU あり）:

```bash
cd /mnt/robot_ssd/PARC2026_pre
# ★ pip install -r requirements.txt は実行しない（torch が壊れて画面/ターミナルが落ちる）
bash examples/smolvla_libero_spatial_lora_local/run_policy_server.sh --port 8000
```

torch が壊れた場合は自動修復を試みます。手動修復:

```bash
source examples/smolvla_libero_spatial_lora_local/env_smolvla.sh
conda activate ~/.local/smolvla-conda-env
bash examples/smolvla_libero_spatial_lora_local/repair_torch.sh
```

**ターミナル 2**（PARC 評価環境）:

```bash
cd /mnt/robot_ssd/PARC2026_pre
source env.sh
python -m pipeline --server-url http://localhost:8000 --track track1 --n-episodes 2
```

提出前チェック:

```bash
cd /mnt/robot_ssd/PARC2026_pre
# 採点環境と同じ Python 3.10 + pip install で検証（推奨）
source env.sh
python validate_submission.py submission.zip --install

# ローカル GPU スモーク（SmolVLA 環境、pip install なし）
bash examples/smolvla_libero_spatial_lora_local/validate_submission.sh submission.zip
```

### F. zip 提出物の作成（必須）

採点環境は **Python 3.10** のため `lerobot==0.6.0` は pip 不可。
`bundle_submission.sh` が LeRobot ソースを `vendor/lerobot/` に同梱した zip を作成する。

```bash
cd /mnt/robot_ssd/PARC2026_pre
bash examples/smolvla_libero_spatial_lora_local/bundle_submission.sh
# → submission.zip（約 686 MB）

cp submission.zip ~/PARC2026_submission/submission.zip   # 提出用に robot_ssd 外へ
```

---

- **Python 3.12 以上**（LeRobot v0.6.0 推奨）
- **NVIDIA GPU + CUDA 対応 PyTorch**
- `git`, `ffmpeg`, `conda`（Miniconda / Anaconda）
- ディスク空き: モデル + データセット + PyTorch で **30 GB 以上** 推奨

> **ストレージについて:** このリポジトリは `/mnt/robot_ssd`（NTFS）上にあります。
> NTFS では conda / venv の **symlink が使えない** ため、次のように分担します。
>
> | 置き場所 | 内容 |
> |---|---|
> | **robot_ssd** (`work/`) | pip キャッシュ、HF キャッシュ、LeRobot ソース、学習出力 |
> | **ext4**（ホーム等） | conda 環境本体（PyTorch 等 ~8 GB） |
>
> ホーム（`/`）の空きが足りない場合は `create_env.sh --clean` で conda / pip キャッシュを削除してください。

> 評価（`python -m pipeline`）はリポジトリの `setup.sh` + `env.sh` 環境（Python 3.10）で行ってください。
> 学習環境と評価環境は **分離** して問題ありません。

## 1. 環境構築

**推奨:** robot_ssd 上に conda 環境を作成する（ホーム容量不足を回避）

```bash
cd /mnt/robot_ssd/PARC2026_pre

# ext4 の空きが少ない場合（~8 GB 必要）: キャッシュ削除
bash examples/smolvla_libero_spatial_lora_local/create_env.sh --clean

# 初回のみ: conda 環境（ext4）+ PyTorch
bash examples/smolvla_libero_spatial_lora_local/create_env.sh

# 以降、作業するたびに有効化
source examples/smolvla_libero_spatial_lora_local/env_smolvla.sh
conda activate ~/.local/smolvla-conda-env

# LeRobot セットアップ（初回のみ）
bash examples/smolvla_libero_spatial_lora_local/setup_lerobot.sh
```

`work/` 以下（robot_ssd = NTFS）に置かれるデータ:

| パス | 内容 |
|---|---|
| `work/pip-cache/` | pip ダウンロードキャッシュ |
| `work/hf_cache/` | Hugging Face モデル・データセット |
| `work/lerobot/` | LeRobot ソース |
| `work/outputs/` | LoRA チェックポイント |
| `work/smolvla_libero_plus_spatial_lora_merged/` | マージ済みモデル |

conda 環境本体（ext4）:

| パス | 内容 |
|---|---|
| `~/.local/smolvla-conda-env/` | Python 3.12 + PyTorch + LeRobot |

<details>
<summary>ホームに conda 環境を置く場合（空き 30 GB 以上ある場合のみ）</summary>

```bash
conda create -n smolvla python=3.12 -y
conda activate smolvla
source examples/smolvla_libero_spatial_lora_local/env_smolvla.sh   # キャッシュだけ robot_ssd へ
pip install torch --index-url https://download.pytorch.org/whl/cu124
bash examples/smolvla_libero_spatial_lora_local/setup_lerobot.sh
```

</details>

## 2. 学習の実行

```bash
python examples/smolvla_libero_spatial_lora_local/train.py
```

既定条件（ノートブックと同一）:

- Spatial 10 タスク × 各 5 エピソード = 50 エピソード
- 3,000 steps、batch size 1、LoRA r=16
- 初期重み: `lerobot/smolvla_libero_plus`

出力先:

- LoRA チェックポイント: `work/outputs/smolvla_libero_plus_spatial_lora/`
- マージ済みモデル: `work/smolvla_libero_plus_spatial_lora_merged/`

マージのみ再実行する場合:

```bash
python examples/smolvla_libero_spatial_lora_local/merge_lora.py
```

## 3. MyPolicy への組み込み

学習で得た **マージ済みモデル** を提出物に載せ、`MyPolicy` から推論します。

### 3.1 ディレクトリ構成

```
submission.zip
├── policy_server.py       # MyPolicy を編集
├── smolvla_mypolicy.py    # 参考実装をコピー（任意）
├── requirements.txt       # 推論依存を追記
└── model_weights/
    └── smolvla_spatial_lora/   # merge_lora.py の出力をコピー
        ├── config.json
        ├── model.safetensors
        ├── policy_preprocessor.json
        ├── policy_preprocessor*.safetensors
        ├── policy_postprocessor.json
        └── policy_postprocessor*.safetensors
```

モデルのコピー例:

```bash
mkdir -p submission_template/model_weights
cp -r examples/smolvla_libero_spatial_lora_local/work/smolvla_libero_plus_spatial_lora_merged \
      submission_template/model_weights/smolvla_spatial_lora

cp examples/smolvla_libero_spatial_lora_local/smolvla_mypolicy.py \
   submission_template/
```

### 3.2 policy_server.py の編集

`MyPolicy` を次のように差し替えます（`smolvla_mypolicy.py` を同梱する場合）:

```python
class MyPolicy(BasePolicy):
    def __init__(self):
        from smolvla_mypolicy import SmolVLAMyPolicy
        self._impl = SmolVLAMyPolicy("model_weights/smolvla_spatial_lora")

    def get_action(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        return self._impl.get_action(obs)

    def reset(self, instruction: str = "") -> None:
        self._impl.reset(instruction=instruction)
```

`smolvla_mypolicy.py` を使わず、`policy_server.py` に直接書いても構いません。
要点は次の 3 点です。

1. **観測の変換** — PARC の観測キーを SmolVLA 形式へマッピング
   - `agentview_image` → `observation.images.image`
   - `robot0_eye_in_hand_image` → `observation.images.image2`
   - `robot0_eef_pos` + axis-angle(`robot0_eef_quat`) + `robot0_gripper_qpos` → `observation.state` (8 次元)
   - `reset()` で受け取った `instruction` → `task`
2. **前処理 / 後処理** — `make_pre_post_processors()` でモデル同梱の processor を使用
3. **エピソードリセット** — `reset()` で `policy.reset()` を呼び、action chunk キャッシュをクリア

### 3.3 requirements.txt

`submission_template/requirements.txt` に推論用依存を追記します（`git+` や `--index-url` は使用不可）:

```text
torch>=2.5
lerobot==0.6.0
safetensors>=0.4
transformers>=4.45
accelerate>=0.33
peft>=0.13
pillow>=10.0
```

> 採点環境は外部通信を遮断するため、モデル重みは zip に同梱してください。
> VLM 重み（SmolVLM2）は `model.safetensors` にマージ済みのため、追加ダウンロードは不要です。

### 3.4 ローカル評価

```bash
# ターミナル 1: ポリシーサーバー（GPU 推奨）
cd submission_template
pip install -r requirements.txt
python policy_server.py --port 8000

# ターミナル 2: PARC 評価（setup.sh 環境）
source env.sh
python -m pipeline --server-url http://localhost:8000 --track track1 --n-episodes 2
```

提出前チェック:

```bash
python validate_submission.py submission_template/ --install
```

## LIBERO-90 多様タスク再学習データ

固定 OOD 12タスクを除外した40タスク（新規物体・受け皿・関節・複合操作を
各10タスク）と、既知40タスクの replay を準備する。

```bash
source examples/smolvla_libero_spatial_lora_local/env_smolvla.sh
conda activate ~/.local/smolvla-conda-env

# metadata のみ取得して分割・schema・episode 数を検証
python examples/smolvla_libero_spatial_lora_local/prepare_diverse_retrain_data.py

# 全データを取得し、学習用ローカルデータセットへ集約
python examples/smolvla_libero_spatial_lora_local/prepare_diverse_retrain_data.py --prepare
```

データ源は `nvidia/LIBERO_LeRobot_v3` の固定 revision
`e5907374380b8f96511957e6ba5582be52a1e179`（OpenMDW-1.1）。
選択結果は `work/diverse_retrain_manifest.json` に保存される。
構成は多様タスク 800 ep + replay 280 ep（replay 25.9%）。OOD task は manifest
上で明示され、学習 episode との非重複を実行時に検証する。

`--prepare` の出力は `work/datasets/libero_diverse_retrain`（5,614 ep / 約4.2GB）。
suite ごとに `names` 表記や scalar の `fps` が異なるため、集約前に feature
metadata を正規化する（dtype と shape は変更しない）。集約後はカメラキーを
`front` / `wrist` に整列し、SmolVLA preprocessor と一致させる。

### Phase 4 学習 + OOD early stopping

```bash
# 学習（20k steps, 2.5k ごとに checkpoint）
bash examples/smolvla_libero_spatial_lora_local/run_phase4_train.sh --skip-merge

# checkpoint ごとに OOD-90 固定12タスクで評価し最良を採用
python examples/smolvla_libero_spatial_lora_local/run_phase4_ood_early_stop.py
```

## 注意事項

| 項目 | ノートブック / 本スクリプト | PARC 本番採点 |
|---|---|---|
| 評価タスク | LIBERO-plus Spatial 10 タスク | Track 1（非公開タスク含む） |
| 画像解像度（学習） | データセット既定（256 前後） | 評価時 128×128（processor がリサイズ） |
| 推論タイムアウト | なし | **1 リクエスト 10 秒**（超過で 0 点） |

SmolVLA は action chunking を行うため、チャンク生成時のみ重い推論が走り、
それ以外の `/act` はキャッシュ返却になります。初回推論が 10 秒以内に収まることを
必ず確認してください。

## トラブルシューティング

### 学習末尾の symlink エラー（`Function not implemented: ... checkpoints/last`）

**原因:** チェックポイント出力先（robot_ssd = NTFS）では symlink が使えません。

**重要:** 学習自体（3000 step）は **正常完了** している場合があります。
`checkpoints/003000/` が存在すれば、マージだけ実行してください:

```bash
python examples/smolvla_libero_spatial_lora_local/merge_lora.py
```

再学習する場合は `setup_lerobot.sh` を再実行すると NTFS 向けパッチが当たり、
末尾エラーは警告のみで終了します。

### torchcodec / libffi エラー（`undefined symbol: ffi_type_uint32`）

**原因:** `torchcodec` がシステムの FFmpeg / libffi と競合します。

**対処:** ローカル学習では動画デコードに `pyav` を使用します（`train.py` 更新済み）。
PyAV は LeRobot インストール時に同梱されています。再実行するだけで構いません。

### Hugging Face ダウンロードエラー（`xet-read-token` / `ConnectionError`）

**原因:** Hugging Face の XET バックエンド経由のダウンロードがネットワーク不安定で失敗することがあります。

**対処（スクリプト更新済み）:** `HF_HUB_DISABLE_XET=1` で通常 HTTP ダウンロードに切り替えます。
`env_smolvla.sh` を source したうえで再実行してください。

データセットだけ先に取得する場合:
```bash
source examples/smolvla_libero_spatial_lora_local/env_smolvla.sh
conda activate ~/.local/smolvla-conda-env
python examples/smolvla_libero_spatial_lora_local/download_dataset.py
```

成功後に学習:
```bash
python examples/smolvla_libero_spatial_lora_local/train.py
```

### `policy_server.py` 実行時にターミナル/画面が落ちる

**原因:** SmolVLA 環境で `pip install -r requirements.txt` を実行すると、
lerobot の依存解決により torch が `2.11.0` 等に上がり、CUDA 12.5 ドライバと
非互換の CUDA ライブラリが混在します。`import torch` やモデルロード時に
セグフォルトが起き、ターミナルや画面（GPU ドライバ）ごと落ちることがあります。

**確認:**
```bash
conda activate ~/.local/smolvla-conda-env
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 正常例: 2.6.0+cu124 True
# 異常例: ImportError (libtorch_cuda.so: undefined symbol ...) または 2.11.0 False
```

**修復・起動（推奨）:**
```bash
cd /mnt/robot_ssd/PARC2026_pre
bash examples/smolvla_libero_spatial_lora_local/run_policy_server.sh --port 8000
```

**注意:** SmolVLA 環境では `pip install -r submission_template/requirements.txt` を
実行しないでください。依存は `setup_lerobot.sh` 済みです。

### `train.py` 実行時にターミナルが落ちる / CUDA エラー

**原因:** `setup_lerobot.sh` 実行時に LeRobot の依存関係が torch を `2.11.0+cu130` 等へ
自動更新し、現在の NVIDIA ドライバ（CUDA 12.5）と非互換になることがあります。

**確認:**
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 正常例: 2.6.0+cu124 True
# 異常例: 2.11.0+cu130 False
```

**修復:**
```bash
source examples/smolvla_libero_spatial_lora_local/env_smolvla.sh
conda activate ~/.local/smolvla-conda-env
bash examples/smolvla_libero_spatial_lora_local/repair_torch.sh
```

ログは `work/train.log` にも出力されます。ターミナルが落ちても内容を確認できます:
```bash
tail -f examples/smolvla_libero_spatial_lora_local/work/train.log
```

### 初回実行が長い

初回は Hugging Face からベースモデル（~900 MB）とデータセットをダウンロードするため、
**10〜30 分**かかることがあります。進捗が止まっているように見えても正常な場合があります。

## ハイパーパラメータの変更

[`config.py`](config.py) を編集してください。主な項目:

- `STEPS`, `BATCH_SIZE`, `LEARNING_RATE` — 学習量
- `LORA_R`, `LORA_ALPHA` — LoRA 容量
- `TRAIN_EPISODES_PER_TASK` — 使用デモ数
