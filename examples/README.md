# 参考例（examples）

| ファイル | 内容 |
|---|---|
| [smolvla_libero_spatial_lora.ipynb](smolvla_libero_spatial_lora.ipynb) | SmolVLA を LIBERO-plus Spatial で LoRA 追加学習する Google Colab ノートブック |
| [smolvla_libero_spatial_lora_local/](smolvla_libero_spatial_lora_local/) | 上記と同一条件の **ローカル GPU 学習スクリプト** + MyPolicy 組み込み例 |

## smolvla_libero_spatial_lora_local（ローカル GPU）

Colab ノートブックと同一条件の LoRA 学習をローカル GPU で実行する。
**実行手順の詳細は [smolvla_libero_spatial_lora_local/README.md](smolvla_libero_spatial_lora_local/README.md) の「実行手順（最新）」を参照。**

```bash
cd /mnt/robot_ssd/PARC2026_pre

# 初回のみ
bash examples/smolvla_libero_spatial_lora_local/create_env.sh
bash examples/smolvla_libero_spatial_lora_local/setup_lerobot.sh

# 作業のたびに
source examples/smolvla_libero_spatial_lora_local/env_smolvla.sh
conda activate ~/.local/smolvla-conda-env

# 学習
python examples/smolvla_libero_spatial_lora_local/download_dataset.py   # 推奨
python examples/smolvla_libero_spatial_lora_local/train.py

# 提出物へ組み込み
mkdir -p submission_template/model_weights
cp -r examples/smolvla_libero_spatial_lora_local/work/smolvla_libero_plus_spatial_lora_merged \
      submission_template/model_weights/smolvla_spatial_lora
cp examples/smolvla_libero_spatial_lora_local/smolvla_mypolicy.py submission_template/
```

## smolvla_libero_spatial_lora.ipynb

`lerobot/smolvla_libero_plus` を初期重みとし、LIBERO-plus Spatial の 10 タスクを
LoRA で追加学習する。学習後は LoRA を元の重みへマージし、追加学習の前後を
同一条件で比較する。

### 使い方

1. Google Colab で開き、ランタイムのタイプを GPU（T4 で足りる）に変更する
2. 上から順に実行する。所要時間は T4 で数時間程度である
3. マージ済みモデル一式（zip）と、追加学習前後の成功率の比較（CSV）が出力される

学習条件は 10 タスク × 各 5 エピソード（計 50 エピソード）、3,000 steps、
バッチサイズ 1 で、Colab で完走することを優先した最小構成である。
性能を伸ばす場合はここを出発点に、自身の環境で条件を組み直すとよい。

### 提出物にするまでの作業

出力されるのは LeRobot 形式のモデル重みであり、これ単体では提出できない。
[submission_template/](../submission_template/) の `MyPolicy` にモデルを組み込み、
ポリシーサーバーの形にする。観測と action の仕様は
[submission_template/policy_server.py](../submission_template/policy_server.py)
の docstring にある。

推論は 1 リクエストあたり 10 秒以内に収める必要がある
（[ルートの README](../README.md#タイムアウト仕様)）。

### ノートブック内の評価と、本番の採点の違い

ノートブック内の評価は学習の効果を手早く確認するためのもので、採点とは条件が異なる。
出てくる成功率は本番スコアの目安にはならない。

| 項目 | ノートブック | 本番の採点 |
|---|---|---|
| 評価タスク | LIBERO-plus Spatial の 10 タスク | Track 1（`compe/t1/` のタスクセット） |
| 実行方法 | LeRobot の `lerobot-eval` | `python -m pipeline` + 提出したポリシーサーバー |
| 観測の解像度 | 256×256 | 128×128 |
| 1 タスクあたりの試行数 | 3（`EVAL_EPISODES_PER_TASK` で変更可） | 非公開（配布キットの既定は 20） |

試行数が 3 のままだと 1 エピソードの成否で成功率が約 33 ポイント動くため、
追加学習の前後を比べる場合は `EVAL_EPISODES_PER_TASK` を増やすこと。

### 実行環境

ノートブックの環境構築は Colab 向けで、[setup.sh](../setup.sh) とは独立している。
依存パッケージのバージョンが一致しない箇所があるため、評価と提出前チェックは
リポジトリ側の環境（`setup.sh` + `env.sh`）で行うこと。

ノートブックが利用する第三者製ソフトウェア・モデル・データセットのライセンスは、
各配布元の表記を参照すること。
