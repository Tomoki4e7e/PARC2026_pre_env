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

## ① instruction正規化のデフォルトを`alias`から底上げする（低コスト・高リターン）

`submission_template/smolvla_mypolicy.py:256` は

```python
INSTRUCTION_MODE = os.environ.get("PARC_INSTRUCTION_MODE", "alias").strip().lower()
```

であり、`Dockerfile`・`requirements.txt`・`bundle_submission.sh`のどこにも `PARC_INSTRUCTION_MODE` を設定する箇所がないことを確認した。つまり**本番の採点サーバーは常に`alias`モード（完全一致辞書のみ、未知の文言はそのまま通す）で動く**。

非公開タスクの instruction 文言が `instruction_aliases.py`（4スイートの既知タスクから生成）に載っている保証はなく、載っていなければモデルは学習時と異なる言い回しをそのまま与えられる。`eval_paraphrase_ood.py`で既にOOD誤マッチ率≤2%になるよう閾値（`PARAPHRASE_MODEL_MIN_SCORE=0.10`）を調整済みなので、これを活かして

- 最低限 `generic`（alias＋規則書き換え＋variantサフィックス除去）へ格上げ
- proxy評価で`match`モードの方が実際にスコアが高いか比較した上で採用

を検討する価値が高い。**コード変更ゼロ、env var 1個で試せる**ため最初に潰すべき仮説。

---

## ② Phase4チェックポイント選定を一本化して「本当に一番良いもの」を出荷する

`run_phase4_proxy_select.py`（→`phase4_v2`としてデプロイ）と`run_phase4_ood_early_stop.py`（→`phase4`としてデプロイ）が別々の指標・別々のvariant名で走り、`deploy_model.sh`のフォールバック順（`phase4_v2 → phase4_from_stage2 → phase4`）次第で片方が黙って上書きされる。実際に出荷されている重みが「意図した最良チェックポイント」である保証がない状態なので、

1. 両選定の候補チェックポイントを列挙
2. 「example4 + 9task hold-out」と「真OOD 12task」の両方で横並び評価
3. 最も両立している1つを明示的に選んでデプロイ

という手順に統合するだけで、既存の学習をやり直さずにスコアが上振れする可能性がある。

---

## ③ アクションチャンク境界のジャーク低減を全タスクに一般化する

`CHUNK_FIRST_STEP_ACTION_SCALE`（`smolvla_mypolicy.py:263-271`）は**特定のinstruction文字列にしか効かない**手作業のスケーリング表。非公開タスクはこの辞書にヒットしないため恩恵を受けない。成功判定は「ゴール達成 かつ 対象外物体の変位が1mm以内」なので、チャンク切替直後のスナップ的な動きは**タスク非依存で衝突リスクを底上げしている**はずである。対策として

- 全タスク共通で新チャンク先頭N stepに減衰スケール（例: 0.6→0.8→1.0のランプ）をかける
- あるいはACT系の時間アンサンブル（重複するチャンク予測を指数加重平均）に置き換える

のどちらかにすれば、instruction辞書のカバレッジに依存せず衝突起因の失敗を減らせる。`total_score.py`のjerk/SPARC計算は既に存在するので、これを使ってオフラインで効果を定量化できる。

---

## 中コスト施策（学習のやり直しを伴う）

| 施策 | 根拠 | 備考 |
|---|---|---|
| 画像augmentation追加 | `train.py`の`build_train_command()`は`--dataset.video_backend`と`--dataset.use_imagenet_stats=false`のみで、color jitter/random cropの類は一切渡していないことを確認済み | LIBERO派生タスクは光源・視点差に弱いことが多く、augmentationは定番かつローリスク |
| Model soup（重み平均） | Stage2/Phase3/Phase4はいずれもStage2マージ済みモデルから分岐した兄弟モデル | 3つのマージ済み`model.safetensors`を単純平均するだけで試せる。推論コストは増えない |
| Phase4のnormalizer統計をpreserve版とrecompute版で両方proxy評価 | `PHASE4_PRESERVE_BASE_NORMALIZER=True`はStage2互換性のための選択で、性能面で検証されたわけではない | `merge_lora.py`のフラグを切り替えて比較するだけ |
| episodes/task の増量 | `TRAIN_EPISODES_PER_TASK=20`など明示的に上限指定 | 特に弱点タスクで頭打ちなら増やす価値あり |

---

## まとめ：優先順位

1. **① instruction正規化モードの見直し**（env var変更のみで試せる）
2. **② Phase4選定ロジックの統合**（再学習不要、選定・デプロイ手順の整理のみ）
3. **③ チャンク境界ジャークの一般化**（推論コード側の小改修）
4. 中コスト施策（augmentation・model soup・normalizer比較・episode増量）は①〜③の効果を見てから着手

どれから着手するか、あるいは①のenv var切り替えを実際にproxy評価で試すところから一緒にやるか、決めてもらえれば実装に入る。
