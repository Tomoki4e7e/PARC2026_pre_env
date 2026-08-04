# サードパーティ ライセンス表記

本配布環境（PARC 2026 予選 自己評価キット）は、`setup.sh` の実行時に第三者が
権利を有するソフトウェア・データセットを取得・インストールして利用する。これらは
本リポジトリには同梱しておらず、それぞれの提供元から取得される。各ソフトウェアの
著作権および利用条件は、それぞれのライセンスに従う。

以下は `setup.sh` が取得・インストールする主要な第三者製ソフトウェアとその
ライセンスの一覧である。正確・最新の条文は各配布元を参照すること。

---

## リポジトリ（`git clone` で取得）

| ソフトウェア | 取得元 | ライセンス |
|---|---|---|
| LIBERO | https://github.com/Lifelong-Robot-Learning/LIBERO | MIT License（Copyright (c) 2023 Lifelong Robot Learning） |
| LIBERO-plus | https://github.com/sylvestf/LIBERO-plus | **ライセンス未記載**（下記の注意を参照） |

### LIBERO-plus のライセンスに関する注意

`sylvestf/LIBERO-plus` のリポジトリには本表記作成時点で LICENSE ファイルおよび
README 上のライセンス記載が確認できない。明示的なライセンスが付与されていない
著作物は、既定では著作権者に全権利が留保される点に留意すること。利用条件が
不明確なため、再配布や商用利用の可否等は、必要に応じて原著作者に確認すること。
なお LIBERO-plus は LIBERO（MIT）を基盤とする派生物であり、LIBERO 由来部分には
LIBERO の MIT License が及ぶ。

---

## データセット（Hugging Face Hub から取得）

| データセット | 取得元 | ライセンス |
|---|---|---|
| LIBERO-plus assets（`assets.zip`） | https://huggingface.co/datasets/Sylvest/LIBERO-plus | LIBERO-plus に準ずる（上記の注意を参照） |

---

## Python パッケージ（`pip install` で取得）

`setup.sh` がピン止めしてインストールする主要パッケージとそのライセンス。
バージョンは配布時点のもの。推移的依存（各パッケージがさらに引く依存）は
含めていない。

| パッケージ | バージョン | ライセンス |
|---|---|---|
| torch | 2.11.0+cpu | BSD-3-Clause |
| mujoco | 3.7.0 | Apache-2.0 |
| robosuite | 1.4.0 | MIT License |
| numpy | 1.26.4 | BSD-3-Clause |
| gym | 0.25.2 | MIT License |
| bddl | 3.6.0 | MIT License |
| cloudpickle | 3.1.2 | BSD-3-Clause |
| easydict | 1.13 | LGPL-3.0 |
| hydra-core | 1.3.2 | MIT License |
| einops | 0.8.2 | MIT License |
| opencv-python-headless | 4.11.0.86 | Apache-2.0 |
| scipy | 1.15.3 | BSD-3-Clause |
| pyyaml | 6.0.3 | MIT License |
| h5py | 3.16.0 | BSD-3-Clause |
| Pillow | 12.3.0 | MIT-CMU (HPND) |
| termcolor | 3.3.0 | MIT License |
| tqdm | 4.70.0 | MPL-2.0 AND MIT |
| matplotlib | 3.10.9 | PSF-based (matplotlib license) |
| requests | 2.34.2 | Apache-2.0 |
| msgpack | 1.2.1 | Apache-2.0 |
| fastapi | 0.140.13 | MIT License |
| uvicorn | 0.51.0 | BSD-3-Clause |
| huggingface_hub | 1.25.1 | Apache-2.0 |
| wand | 0.7.2 | MIT License |
| scikit-image | 0.25.2 | BSD-3-Clause |
| pytest | 9.1.1 | MIT License |

> 注: `easydict` は LGPL-3.0、`tqdm` は MPL-2.0 AND MIT。これらは他パッケージの
> 寛容型ライセンス（MIT / BSD / Apache-2.0）とは条件が異なるため、配布形態を
> 変える場合は各条項を確認すること。

---

各ライセンスの全文は、`setup.sh` 実行後に取得された各パッケージ／リポジトリの
配布物内（例: 各 `git clone` 先の `LICENSE`、`venv/lib/.../site-packages/<pkg>/`
配下のライセンスファイルや `*.dist-info/`）に含まれる。
