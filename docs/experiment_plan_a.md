# 実験計画：案A「可動部ハンドル把持タスク × LIBERO-plus摂動」による
# 「段階依存の開口部把持能力」検証実験

## 0. 背景・目的

本実験の最終目標は、双腕ロボットによる靴下装着タスクのうち、
「履けている段階に応じて形が変わる靴下の開口部を見つけ、把持する」
という部分能力を、LIBERO/LIBERO-plus + SmolVLAベースのBase Policy
+ Flow Matching残差ポリシー（残差強化学習）の構成で事前検証することである。

LIBERO/LIBERO-plusには変形物体（布）のタスクは存在しないため、完全な代替には
なり得ない。そこで本実験は、靴下把持タスクが要求するコア能力を以下の2点に
分解し、それを再現できる既存タスクで代理検証することを目的とする。

1. **状態依存の把持点再局在化**：把持対象（引き出し／扉のハンドル）の
   見かけの位置・角度・遮蔽が毎回変わる中で、視覚から正しい把持点・
   把持姿勢を都度選び直す能力。
2. **軌道暗記からの脱却**：Base Policy が学習データの典型軌道を
   出力するだけでは失敗する状況を作り、Residual Policy（Flow Matching）
   が視覚状態に応じた補正を行っているかを検証する。

対象研究の位置づけ上、本実験の成否が靴下タスク本体の成否を保証するもの
ではない点に留意する（詳細は §8 限界を参照）。

---

## 1. 検証する仮説

- **H1**: Base Policy（SmolVLA）単体は、Objects Layout / Camera
  Viewpoint 摂動が大きくなるにつれて把持成功率が単調に低下する
  （＝暗記軌道への依存を示す）。
- **H2**: Base Policy + Flow Matching残差ポリシー（残差強化学習）は、
  同摂動下でもBase Policy単体より把持成功率の低下が緩やかである
  （＝視覚状態依存の補正ができている）。
- **H3**: 残差ポリシーによる補正量（残差アクションのノルム等）は、
  摂動強度と正の相関を持つ（＝残差ポリシーが状態に応じて適応的に
  働いていることの間接証拠）。

---

## 2. 使用タスク

### 2.1 タスク選定
LIBERO-Goal／LIBERO-90 に含まれる可動部操作タスクのうち、ハンドル形状・
サイズが異なる以下3タスクを主対象とする（多様な把持対象形状での汎化性も
同時に見るため）。

| タスク名（原題） | 対象物 | 選定理由 |
|---|---|---|
| `open the middle drawer of the cabinet` | 引き出しハンドル（横長・水平） | 標準的なLIBERO-Goalタスク、水平方向の把持姿勢が必要 |
| `open the top drawer and put the bowl inside` | 引き出しハンドル | 到達距離が異なる別インスタンスでの汎化確認用 |
| `open microwave` (LIBERO-90) | 電子レンジ扉ハンドル（縦長） | ハンドル形状が引き出しと異なり、把持方向（縦）が異なる代替対象 |

### 2.2 成功基準の再定義（重要）
本実験の関心は「引き出しを最後まで開けられるか」ではなく「ハンドルを
正しく把持できるか」である。そのため、各タスクについて **独自の成功基準
（Grasp Success）を新たに定義**する。

- **Grasp Success（本実験の主指標）**：グリッパーがハンドル可動域内に
  接触し、閉じた状態でハンドルの並進追従（ハンドルの動きに対しグリッパー
  位置の追従誤差が閾値以下）が一定時間（例：0.5秒）継続すること。
- **Task Success（参考指標）**：LIBERO標準のタスク完了基準（引き出しが
  一定量開く等）。Base/Residualの比較においては補助的に記録する。
- `open the top drawer and put the bowl inside`は靴下着衣支援に必要な
  handle把持との対応だけを検証し、bowl配置TSRは学習・モデル選択・
  Phase 1 gateの対象外とする。

Grasp Success判定はMuJoCoの接触情報（contact force, grip site distance）
から自動判定するスクリプトを別途実装する（§6参照）。

---

## 3. 摂動設定（LIBERO-plus）

LIBERO-plus（Fei et al., 2025, arXiv:2510.13626）が定義する7摂動軸・21
サブコンポーネントのうち、本実験の目的（視覚状態依存の把持点再局在化）
に直結する以下3軸を主要摂動として使用する。

| 摂動軸 | 内容 | 採用理由 |
|---|---|---|
| **Objects Layout** | 対象物（ハンドル含む家具）の位置・向きのランダム化、confounding objects配置 | 把持点の絶対位置を毎回変えることで暗記軌道を無効化する主軸 |
| **Camera Viewpoint** | カメラ視点の変化 | ハンドルの見かけの形状・角度を変化させ、靴下開口部の見た目変化を模擬 |
| **Robot Initial States** | ロボット初期姿勢のランダム化 | 到達軌道の毎回の作り直しを要求し、軌道暗記の効果を最小化 |

Light Conditions / Background Textures / Sensor Noise / Language
Instructions の4軸は本実験の主眼（幾何学的な把持点再局在化）とは直接
関係が薄いため、**副次実験（Ablation, §5.3）**として軽く触れるに留める。

### 3.1 摂動強度レベル
LIBERO-plusの提供する強度パラメータに従い、各摂動軸につき3段階を設定する。

| レベル | Objects Layout | Camera Viewpoint | Robot Initial States |
|---|---|---|---|
| L0（摂動なし） | 標準配置 | 標準視点 | 標準初期姿勢 |
| L1（軽度） | ±5cm, ±15° | ±10° | 関節角±5% |
| L2（強度） | ±15cm, ±45° | ±25° | 関節角±15% |

摂動レベルはLIBERO-plus公式リポジトリ（github.com/sylvestf/LIBERO-plus）
のパラメータ範囲に準拠して設定し、実装時に必要であれば数値を調整する。

---

## 4. 比較対象（モデル条件）

| 条件名 | 構成 | 目的 |
|---|---|---|
| **Base-only** | SmolVLAのみ | H1の検証：暗記軌道依存の確認（ベースライン） |
| **Base + Residual (FM)** | SmolVLA Base + Flow Matching残差ポリシー（残差強化学習で学習） | H2・H3の検証：提案手法本体 |


Base Policyは意図的に「学習データに忠実な動き」を出すよう、データ拡張・
Domain Randomizationなしの標準LIBEROデータのみで学習する。これにより
Base単体の摂動脆弱性を意図的に残し、Residual Policyの寄与を明確化する。

---

## 5. 実験手順

### 5.1 Phase 1: Base Policy学習
1. 対象drawer 2タスクにつき、LIBERO標準デモと検証済みsim demoでBase Policyを模倣学習。
2. 摂動なし（L0）環境でのstrict GSRが各drawer taskで90%以上に
   達することを確認してから次フェーズへ進む。

### 5.2 Phase 2: 摂動下でのBase Policy単体評価（H1）
1. 3タスク × 3摂動軸 × 3強度レベル（L0/L1/L2）で評価。
2. 各条件につき最低50 episode（LIBERO-plus推奨のサンプル数に準拠）。
3. Objects Layout, Camera Viewpoint, Robot Initial Statesは単独摂動で
   まず評価し、Grasp SuccessとTask Successの摂動レベル別低下曲線を
   取得する。

### 5.3 Phase 3: Residual Policy学習
1. Base Policyを固定し、摂動ありのシミュレーション環境上でオンライン
   残差強化学習を実施。行動空間は Base出力への加算残差
   （a = a_base + a_residual）とし、a_residualをFlow Matchingで
   パラメータ化する。
2. 学習時の摂動分布：L1相当のランダム化を学習用ドメインランダマイズ
   として採用し、L2は未見の汎化評価専用として学習には使用しない
   （汎化性能を厳密に見るため）。
3. 報酬設計：Grasp Success到達で高報酬、接触前の到達距離減少に
   対するshaping reward、過大な接触力へのペナルティを付与。

### 5.4 Phase 4: 摂動下でのBase+Residual評価（H2・H3）
1. Phase 2と同一の評価プロトコル（3タスク×3摂動軸×3強度、各50 episode）
   をBase+Residualで実施。
2. 特にL2（未学習強度）での成功率をBase-onlyと比較し、汎化による
   上乗せ効果を主要な結果とする。
3. 残差アクションのノルム・方向を摂動レベル別にログし、H3（適応的な
   補正の間接証拠）を検証する。

### 5.5 Phase 5（Ablation）: 副次摂動軸・複合摂動
1. Light Conditions, Background Textures, Sensor Noiseを単独で軽く評価
   （各L1相当、50 episode）し、幾何摂動以外への頑健性も参考値として
   記録する。
2. Objects Layout + Camera Viewpointの複合摂動（LIBERO-plusが指摘する
   “negative compositional generalization”の再現確認）を実施し、
   単独摂動の性能低下から単純加算で予測できない追加低下がないか確認する。

---

## 6. 評価指標・ログ項目

| 指標 | 定義 | 用途 |
|---|---|---|
| Grasp Success Rate (GSR) | §2.2定義のGrasp Success達成episode割合 | 主指標 |
| Task Success Rate (TSR) | LIBERO標準タスク完了率 | 参考指標 |
| Time-to-Grasp | 開始からGrasp Success判定までのステップ数 | 効率性の確認 |
| Residual Action Norm | 各タイムステップの \|\|a_residual\|\| | H3検証・補正量の可視化 |
| Grasp Point Error | 実際の把持点と理想把持点（ハンドル中心線上の最近点）とのユークリッド距離 | 把持点再局在化精度の定量評価 |
| Failure Mode内訳 | 「対象を見失う」「誤方向から接近」「接触後保持失敗」等をルールベースで分類 | 失敗要因分析 |

全指標は摂動軸×レベル×タスクごとに平均・標準偏差を算出し、L0を基準とした
相対低下率（%）でBase-onlyとBase+Residualを比較する。

---

## 7. 想定される結果の見方

- Base-onlyのGSRがL0→L2で大きく低下し、かつBase+ResidualのGSRの低下が
  緩やかであれば、H1・H2は支持される。
- 両条件のGSR低下カーブがほぼ同じであれば、Residual Policyが視覚状態に
  応じた補正ではなく単なるノイズ耐性の付与に留まっている可能性があり、
  Residual Policyの設計（観測入力・報酬設計）の見直しが必要と判断する。
- Grasp Point Errorが摂動レベルに応じてBase-onlyでは増大し、
  Base+Residualでは抑制されていれば、これは「開口部（把持対象）の
  見た目の変化に応じて把持点を再局在化できているか」という、靴下タスクで
  本質的に問いたい能力に最も近い定量的証拠となる。

---

## 8. 限界（Limitations）

- 本実験は剛体・可動部（ハンドル）を用いた代替であり、布の伸縮・自己遮蔽・
  不定形な開口といった靴下タスク特有の困難さは再現できていない。
- 「段階（装着の進み具合）に応じた形状変化」は、本実験では「摂動レベル
  （L0/L1/L2）」や「視点変化」で疑似的に表現しているに過ぎず、連続的な
  変形のダイナミクスは含まれない。
- あくまで「視覚状態に応じた把持点再局在化」という能力の必要条件の検証
  であり、十分条件ではない。本実験で良好な結果が得られても、実タスク
  （靴下把持）での成功を保証するものではない。
- より高い忠実度が必要な場合は、SoftGym（Lin et al., CoRL 2020）や
  GarmentLab等の変形物体シミュレータでの追加検証を推奨する（別紙提案済み）。

---

## 9. 実装メモ・使用ツール

- シミュレータ／評価環境：LIBERO-plus公式リポジトリ
  （https://github.com/sylvestf/LIBERO-plus）をベースに、
  Grasp Success自動判定スクリプトを追加実装。
- ポリシー実行基盤：既存のSmolVLAを固定Base Policyとして流用し、
  独立したFlow Matching残差モジュールの出力を7次元actionへ加算する。
- 残差強化学習の実装は、固定Base Policy出力に対する加算残差学習という
  標準的な定式化（Johannink et al., 2019; Davchev et al., 2021）に準拠。

---

## 10. 参考文献

- Fei, S., Wang, S., Shi, J., et al. (2025). *LIBERO-Plus: In-depth
  Robustness Analysis of Vision-Language-Action Models*. arXiv:2510.13626.
  https://github.com/sylvestf/LIBERO-plus
- Liu, B., et al. (2023). *LIBERO: Benchmarking Knowledge Transfer for
  Lifelong Robot Learning*. NeurIPS 2023 Datasets and Benchmarks.
- Johannink, T., et al. (2019). *Residual Reinforcement Learning for
  Robot Control*. ICRA 2019.
- Davchev, T., et al. (2021). *Residual Reinforcement Learning from
  Demonstrations*. arXiv:2106.08050.
- （Flow Matching残差方策の設計参考）RFS: *Reinforcement Learning with
  Residual Flow Steering for Dexterous Manipulation*. arXiv:2602.01789.
- （変形物体タスクへの拡張検討時の参考）Zhang, F., & Demiris, Y. (2022).
  *Learning garment manipulation policies toward robot-assisted
  dressing*. Science Robotics, 7(65), eabm6010.

---

## 11. 実装時に確定した条件と再現コマンド

- Objects Layoutは実在するLIBERO-plus BDDLを使用し、L1をdifficulty
  1–2、L2をdifficulty 4–5とした。
- Camera ViewpointとRobot Initial Statesは分類先BDDLが配布物にないため、
  seed固定の実行時摂動とした。CameraはL1 ±10° / L2 ±25°、Robotは
  joint range比でL1 ±5% / L2 ±15%とし、joint limitと初期接触を検査する。
- `open microwave`はL0 shape-generalization評価だけに使用し、Base LoRAと
  摂動学習から除外した。
- 主指標のGrasp Successは、finger-handle接触、gripper aperture 0.055以下、
  handle joint変位0.003以上、EEF-handle相対位置変化0.02m以下を0.5秒継続、
  の全条件で判定する。
- 公開データの実在episodeは対象drawer 2タスク合計79件であり、
  計画値100件（各50件）を満たさなかった。重複samplingは行わず全79件を使用した。

再現コマンド:

```bash
source examples/smolvla_libero_spatial_lora_local/env_smolvla.sh
"$SMOLVLA_CONDA_ENV/bin/python" \
  examples/smolvla_libero_spatial_lora_local/prepare_experiment_a_data.py
"$SMOLVLA_CONDA_ENV/bin/python" \
  examples/smolvla_libero_spatial_lora_local/train.py \
  --experiment-a-base --steps 10000

source env.sh
python -m pytest -q tests/residual_rl
python -m experiments.libero_residual_rl manifest \
  --episodes-per-condition 5 --include-microwave
```

Base推論はSmolVLA環境の`base_server.py`とLIBERO環境の
`RemoteBasePolicy`に分離し、オンライン残差RL時にも両環境の依存関係を
混在させない。実行結果は`docs/experiment_plan_a_results.md`を参照。
