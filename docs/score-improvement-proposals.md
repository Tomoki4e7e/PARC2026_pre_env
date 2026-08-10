# Physical AI プロジェクト スコア向上のための施策提案

> [pipeline-analysis.md](pipeline-analysis.md) のコード解析を踏まえ、公開情報の範囲で「勝ち筋」になりそうな施策を優先度順にまとめる。①〜③は特にリターンが大きいと見ている。

## 前提の整理

以下は開発者本人による確認済みの前提であり、施策の優先度判断に反映している。

- 推論に必要な大サイズの重みファイル（`model.safetensors`等）はgit管理の都合上リポジトリから省かれている。
- 最終スコアの具体的な算出方法は非公開であり、現状の `overall_score = mean(success_rate)` は暫定的な仮設定である。
- 本番の採点環境自体はこのプロジェクト内には存在しない。
- 本番でTrack1に摂動（ノイズ）は課されない。

スコア算出式が非公開・暫定単純平均である以上、**「どの重み付けになっても効くもの」を優先**するのが合理的である。具体的には②③（衝突回避・確実な出荷物の選定）と①（言語ミスマッチの解消）は、成功率単体でも、trajectory品質が将来加味されても、両方に効く。逆にハイパーパラメータの微調整（LoRA rank等）は既に4段階分試行済みで収穫逓減している可能性が高く、優先度は下げてよい。

---

## ⓪ 失敗記録: Phase5 提出が本番スコア 0.000（2026-08-09）

**症状**: ポリシーサーバーは正常起動、`/act` は全て 200、合計 140.1 秒で完走した上で
`track1: 総合スコア 0.000`。パッケージ側は無罪だった（`config.json` の実質差分は
学習ハイパラのみ、正規化統計は stage2 と md5 一致、validate は PASS）。

**原因**: 選定に使った hold-out 9 タスクのうち **5 タスクが Phase5 の学習データそのもの**
だった（`pick up the alphabet soup` / `ketchup` / `butter` / `open the middle drawer` /
`turn on the stove`）。汚染タスクでのみスコアが伸び、合成 proxy は phase5 0.524 >
stage2 0.482 となったため、汎化を失ったモデルを最良と誤選定した。

| hold-out suite | 学習との重複 | stage2 | phase5 |
|---|---|---|---|
| spatial | なし（clean） | 0.400 | **0.333** |
| object | 3/3 学習済み | 0.067 | 0.267 |
| goal | 2/3 学習済み | 0.200 | 0.333 |

clean な suite だけが悪化しており、典型的な暗記の兆候だった。本番は非公開タスクなので
成功が 0 に落ち、Total Score は成功・衝突をゲートにした積のため全体が 0.000 になった。

**副因**: 品質フィルタで選んだ 360 ep は品質スコアがほぼ同一帯（min 0.659 / mean 0.670）に
集中し、軌道の多様性が落ちていた。滑らかさで絞ると挙動のバリエーションが削られる。

**対策（実装済み）**: `proxy_holdout.py` を追加し、hold-out を「その段階の学習タスクと排他」
に自動構成する（`--holdout-mode auto_clean` が既定、汚染時は停止）。加えて
`clean_holdout_not_worse` と `ood_not_worse`（OOD-90）をゲート化し、1 つでも落ちたら
stage2 を維持する。`stage1` / `phase4` は 3 suite の全ベースタスクを学習済みで
suite 内 hold-out が成立しないことも判明したため、OOD-90 が唯一の汎化シグナルである。

---

## ① instruction正規化のデフォルトを`alias`から`generic`へ（実施済み・2026-08-09）

`submission_template/smolvla_mypolicy.py` の既定を **`generic`** に変更した。

### A/B 検証（stage2・同一 seed）

| mode | inset (example4) | holdout spatial | language proxy |
|---|---|---|---|
| alias（旧既定） | 0.625 | **0.400** | 0.722 |
| **generic（新既定）** | 0.625 | **0.867** | 0.722 |
| match | 0.625 | 0.867 | 0.722 |

- **generic の本命効果**: instruction 末尾の摂動サフィックス（`table N` / `light N`）を除去し、学習時の task 文字列に揃える。hold-out で +0.467。
- **match は不採用**: 上記セットでは generic と同点だが、未知 instruction を既知 40 タスクへ強制写像する（OOD false-accept が高い）。
- **本番 Track1**: CSV 由来の instruction には摂動サフィックスが無い想定のため、本番での上振れ幅は hold-out ほど大きくない可能性がある。それでも alias との差分は「等価 or 改善」で悪化は観測されなかった。

結果の生データ: `results/instruction_mode_ab_stage2/summary.json`

---

## ② Phase4チェックポイント選定を一本化（実施済み・2026-08-09）

`run_phase4_proxy_select.py`（→`phase4_v2`）と`run_phase4_ood_early_stop.py`（→旧`phase4`）の
二重系統を廃止し、**`run_phase4_unified_select.py` を唯一の出荷決定点**にした。

### 変更点

| 項目 | 旧 | 新 |
|---|---|---|
| 出荷決定 | proxy / OOD が別々 + deploy silent fallback | `run_phase4_unified_select.py` のみ |
| `deploy_model.sh phase4` | v2→from_stage2→legacy を黙ってフォールバック | **統一選定 JSON 必須**（未実行はエラー） |
| 明示 variant | `phase4_v2` / `phase4_legacy` | + `phase4_from_stage2` / `phase4_selected` |
| ゲート | バラバラ | inset / clean-holdout(可なら) / OOD-90 を stage2 比較 |

Phase4 は libero_plus 3 suite をほぼ全学習しているため **suite 内 clean hold-out は N/A**。
その場合は inset + OOD-90 で stage2 と比較する。

結果: `results/phase4_unified_select/selection.json`

### 横並び結果（現行推論: generic + chunk off）

| candidate | inset | clean holdout | OOD-90 | proxy | ゲート |
|---|---|---|---|---|---|
| **stage2（採用）** | **0.725** | **0.778** | 0.000 | **0.913** | baseline |
| phase4_v2 | 0.000 | N/A | 0.000 | -0.034 | inset / proxy 落ち |
| phase4_from_stage2 | 0.000 | N/A | 0.000 | 0.090 | inset / proxy 落ち |
| phase4_legacy | 0.000 | N/A | 0.000 | 0.073 | inset / proxy 落ち |

**出荷決定: stage2 を維持。** `deploy_model.sh phase4` も選定 JSON 経由で stage2 を出す。

---

## ③ アクションチャンク境界のジャーク低減（実装・評価済み・2026-08-09）

全タスク共通ランプを実装し、stage2 + `generic` で A/B した。

### 攻撃的ランプ（初回）

| mode | inset | holdout avg | hold coll | proxy |
|---|---|---|---|---|
| **off（採用）** | **0.725** | **0.778** | 0.089 | **0.913** |
| legacy（旧・タスク辞書） | 0.625 | 0.778 | 0.089 | 0.870 |
| ramp `0.6→0.8→1.0` | 0.675 | 0.756 | 0.111 | 0.874 |

### milder 再試行

| mode | inset | holdout avg | hold coll | proxy | ゲート |
|---|---|---|---|---|---|
| **off** | **0.725** | 0.778 | 0.089 | **0.913** | baseline |
| ramp `0.85→0.95→1.0` | 0.650 | 0.778 | 0.089 | 0.884 | 落ち（inset/proxy） |
| ramp `0.9→1.0` | 0.675 | 0.778 | 0.089 | 0.895 | 落ち（inset/proxy） |

- hold-out 成功率は milder でも `off` と同点だが、**inset が一律に下がる**（衝突はわずかに減る程度）。
- **結論不変: 既定は `CHUNK_SCALE_MODE=off`**。ランプ実装は残置（`PARC_CHUNK_SCALE_MODE=ramp` + `PARC_CHUNK_SCALE_RAMP` で再試行可）。

生データ: `results/chunk_scale_ab_stage2/summary.json`, `results/chunk_scale_ab_mild/summary.json`

---

## ④ Temporal ensemble（ACT 式・推論のみ）（実施済み・2026-08-09）

ACT [arXiv:2304.13705](https://huggingface.co/papers/2304.13705) Algorithm 2 の
指数重み付き temporal ensembling を `smolvla_mypolicy.py` に実装した。
**追加学習は不要**（同一 stage2 重みの推論切替のみ）。

### 実装

| 環境変数 | 採用値 | 意味 |
|---|---|---|
| `PARC_TEMPORAL_ENSEMBLE` | `1`（既定） | 有効化 |
| `PARC_TE_STRIDE` | `5` | 何 step ごとに再推論するか |
| `PARC_TE_COEFF` | `0.01` | ACT 既定。古い予測をやや重視 |
| `PARC_TE_CHUNK` | （未設定=50） | ensemble に使う chunk 長 |

無効化: `PARC_TEMPORAL_ENSEMBLE=0`。A/B スクリプト:
`examples/smolvla_libero_spatial_lora_local/run_temporal_ensemble_ab.py`

### A/B（stage2 + generic + chunk off・同一 seed）

| variant | inset | holdout avg | hold coll | hold jerk | proxy | ゲート |
|---|---|---|---|---|---|---|
| off（旧既定） | 0.725 | 0.778 | 0.089 | 3.932 | 0.913 | baseline |
| **te_s5_c0.01（採用）** | **0.850** | **0.933** | **0.022** | **2.603** | **1.102** | 全通過 |
| te_s10_c0.01 | 0.750 | 0.933 | 0.022 | 2.890 | 1.055 | 全通過 |

- 成功率・衝突・滑らかさ（jerk）が同時に改善。proxy +0.19。
- 評価時間は inset 117s → 245s（再推論増）。1 リクエスト推論 ~150ms で 10s 制限内。
- **出荷既定を `te_s5_c0.01` に変更**（stage2 重みは変更なし）。

生データ: `results/temporal_ensemble_ab_stage2/summary.json`

---

## 中コスト施策（学習のやり直しを伴う）

| 施策 | 根拠 | 備考 |
|---|---|---|
| **品質フィルタ LoRA（Phase5）** | デモを jerk/SPARC/EEF回転で上位選定し BC。checkpoint は success+collision+滑らかさの複合 proxy | `score_episode_quality.py` → `train.py --phase5` → `run_phase4_proxy_select.py --phase5` |
| 画像augmentation追加 | `train.py`の`build_train_command()`は`--dataset.video_backend`と`--dataset.use_imagenet_stats=false`のみで、color jitter/random cropの類は一切渡していないことを確認済み | LIBERO派生タスクは光源・視点差に弱いことが多く、augmentationは定番かつローリスク |
| Model soup（重み平均） | Stage2/Phase3/Phase4はいずれもStage2マージ済みモデルから分岐した兄弟モデル | 3つのマージ済み`model.safetensors`を単純平均するだけで試せる。推論コストは増えない |
| Phase4のnormalizer統計をpreserve版とrecompute版で両方proxy評価 | `PHASE4_PRESERVE_BASE_NORMALIZER=True`はStage2互換性のための選択で、性能面で検証されたわけではない | `merge_lora.py`のフラグを切り替えて比較するだけ |
| episodes/task の増量 | `TRAIN_EPISODES_PER_TASK=20`など明示的に上限指定 | 特に弱点タスクで頭打ちなら増やす価値あり |

### 画像 augmentation 実装（2026-08-10）

`train.py --augmentation` を追加し、stage2 をベースにした独立 LoRA 候補として実装した。
旧 stage2 の重みや出力先は上書きしない。

- photometric: brightness、contrast、saturation、hue
- nonlinear illumination: gamma `0.75–1.35`
- texture: coarse/fine noise、strength `0.01–0.045`
- 1フレームにつき最大4変換を重み付き選択し、ランダム順で複合適用
- 4,000 steps、LR `5e-5 → 5e-6`、LoRA rank/alpha `16/16`
- 比較: `run_augmentation_select.py` で旧 stage2 と inset / clean hold-out / OOD-90 を同条件評価

### GPU 不検出の原因と解消

初回実行は Codex の通常 sandbox から起動したため、ホストに存在する
`/dev/nvidia0` / `nvidiactl` / `nvidia-uvm*` が名前空間内へ公開されていなかった。
その結果、kernel module と RTX 3060 Ti は正常でも `nvidia-smi` と
`torch.cuda.is_available()` が失敗した。ホスト側では driver `555.42.06`、
torch `2.6.0+cu124`、CUDA 12.4 の整合性と全 device node を確認できた。
GPU device を許可した実行環境で起動することで解消し、追加の driver 更新は不要だった。

### 追加学習・stage2 比較（完了・2026-08-10）

実行コマンド:

```bash
source examples/smolvla_libero_spatial_lora_local/env_smolvla.sh
/home/cm-pc/.local/smolvla-conda-env/bin/python \
  examples/smolvla_libero_spatial_lora_local/train.py --augmentation \
  --log-file examples/smolvla_libero_spatial_lora_local/work/image_augmentation_train.log
/home/cm-pc/.local/smolvla-conda-env/bin/python \
  examples/smolvla_libero_spatial_lora_local/run_augmentation_select.py
```

4,000 step を完走し、最終 loss `0.229`。マージ済み重み SHA-256 は
`5a84efeb1a99569a3890306d65b4189198ed03a54cbf282c139dbb2c9a768314`。
現行推論（generic + chunk off + temporal ensemble）で旧 stage2 と同条件評価した。

| candidate | inset success | inset collision | clean holdout | hold collision | OOD-90 | proxy |
|---|---:|---:|---:|---:|---:|---:|
| **stage2（維持）** | **0.850** | **0.050** | **0.933** | **0.022** | 0.000 | **1.102** |
| image augmentation | 0.825 | 0.075 | 0.911 | 0.044 | 0.000 | 1.069 |

新モデルは `inset_not_worse`、`clean_holdout_not_worse`、`proxy_beats_stage2` の
3ゲートを落とした。色・照明・texture 頑健化の狙いに対し、通常分布での成功率と衝突率が
悪化したため不採用。選定スクリプトは stage2 を提出テンプレートへ復元済み。

生データ: `results/image_augmentation_select/selection.json`

### Augmentation checkpoint + stage2 補間（採用・2026-08-10）

4k最終モデルの差分が強すぎる可能性を検証するため、既存の1k / 2k / 3k / 4k
checkpointと、`stage2 + α × (aug4k - stage2)`（α=0.25 / 0.50 / 0.75）を比較した。
短縮screening（inset 5 / hold-out 3）で全8候補を評価し、通過したα=0.25 / 0.75を
stage2と本評価（inset 10 / hold-out 5 / OOD 3）した。

| candidate | inset | collision | clean holdout | hold collision | OOD-90 | proxy | 全ゲート |
|---|---:|---:|---:|---:|---:|---:|---|
| stage2 | 0.850 | 0.050 | 0.933 | 0.022 | 0.000 | 1.102 | baseline |
| **blend α=0.25（採用）** | **0.850** | **0.050** | **0.956** | **0.022** | 0.000 | **1.115** | PASS |
| blend α=0.75 | 0.850 | 0.050 | 0.956 | 0.022 | 0.000 | 1.114 | PASS |

α=0.25は通常4タスクの成功・衝突を維持し、clean object hold-outを0.867→0.933へ
改善した。spatial 0.933、goal 1.000は維持。proxy最大のα=0.25を出荷候補に採用した。
checkpoint単体は1k/2kでinsetが0.800へ低下し、3k/4kも本評価へ進める改善がなかった。

再配置:

```bash
bash examples/smolvla_libero_spatial_lora_local/deploy_model.sh augmentation_selected
```

生データ:

- screening: `results/augmentation_checkpoint_select/screening.json`
- final: `results/augmentation_checkpoint_select/selection.json`

---

## まとめ：優先順位

1. **① instruction正規化 → `generic` 既定化（実施済み）**
2. **③ chunk スケール → 攻撃的 / milder ramp とも否決、既定 `off`（実施済み）**
3. **② Phase4選定一本化 → 全候補が stage2 に敗北、stage2 維持（実施済み）**
4. **④ temporal ensemble → `te_s5_c0.01` 採用（実施済み）**
5. 残る中コスト施策（augmentation・model soup 等）

提出物は **stage2 + `generic` + chunk `off` + temporal ensemble `s5/c0.01`**。
