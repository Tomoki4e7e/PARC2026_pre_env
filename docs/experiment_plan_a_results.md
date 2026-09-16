# LIBERO Flow残差RL 実験結果

実行日: 2026-09-15

## 結論

Phase 1の事前登録ゲート（全対象タスクのL0 GSR 90%以上）を通過しなかった。
このため、L1オンライン残差RL、小規模paired比較、各50 episode本評価は
実行していない。ゲートを無視した後段評価による選択バイアスを避け、
本実験はBase Policy不成立という否定結果で停止した。

## 実装・検証

- `experiments/libero_residual_rl/`へ、摂動catalog、seed固定Camera/Robot摂動、
  Grasp Success状態機械、安全制限付きresidual環境、CNN+GRU条件encoder、
  4×7次元Conditional Flow actor、twin critic、TD3型online trainer、
  paired evaluator、bootstrap/McNemar/Spearman集計を実装した。
- Base-onlyは同じ環境wrapperでresidualを厳密にゼロ化する。
- MuJoCo真値は報酬と評価だけに使用し、policy observationへ渡さない。
- 単体テスト10件、3タスクの実環境reset/step、3摂動軸の適用を通過した。
- synthetic replayによる30 updateスモークは全値finiteで、critic lossの
  最初5 update平均0.02439から最後5 update平均0.001279への低下を確認した。
  checkpoint保存・再ロードも通過した。
- Base推論最大時間は0.407秒で、10秒制約を満たした。

## Phase 1結果

固定revision `lerobot/smolvla_libero_plus@7bb70aa...` の5 episode予備評価:

- `open the middle drawer...`: GSR 0%、TSR 80%
- `open the top drawer and put the bowl inside`: GSR 0%、TSR 40%
- `open microwave`: GSR 0%、TSR 0%

その後、主要drawer 2タスクの公開demo全79 episodeを用い、LoRA
（rank 16、alpha 16、10,000 steps、画像encoderを含む）を実行した。
学習lossは2.183から0.326へ低下した。計画値は各50 demoだったが、固定revision
に存在する一意episodeは合計79件だけだったため、重複samplingはしていない。

LoRA後のL0 5 episodeゲート:

- `open the middle drawer...`: GSR 60%、TSR 60%
- `open the top drawer and put the bowl inside`: GSR 0%、TSR 0%
- hold-out `open microwave`: GSR 0%、TSR 0%

中段drawerの把持は改善したが、全タスク90%条件から大きく不足した。
3,000-step時点と10,000-step時点の結果も同じであり、学習継続だけでは
改善しなかった。

## 仮説判定

- H1: 判定不能。L0 Baseゲート未通過のため、L1/L2の単調低下を測定していない。
- H2: 判定不能。事前登録条件に従いResidual Policyを実環境学習していない。
- H3: 判定不能。実環境Residual actionと摂動強度の対応データがない。

今回の結果はFlow残差RLの不支持ではなく、比較の前提となるBase Policyが
成立しなかったことを示す。次回は上段drawer+bowlの段階別demo品質、
gripper閉動作、instruction/schema整合を先に検証し、Base L0ゲートを
満たしたcheckpointを固定してから残差実験を再開する必要がある。

## 成果物と再現性

- 生ログ: `experiments/libero_residual_rl/artifacts/evaluation/`
- trainer smoke: `experiments/libero_residual_rl/artifacts/smoke/`
- 機械可読ゲート結果:
  `experiments/libero_residual_rl/artifacts/results/gate_summary.json`
- dataset manifest:
  `examples/smolvla_libero_spatial_lora_local/work/datasets/experiment_a_source/experiment_a_manifest.json`
- merged model SHA-256:
  `c2abb3adf121cfdad4f0b6bf52d15707830ae2f2df54b55a759a3facef1f879f`
- dataset manifest SHA-256:
  `10bf060294578e0ded8e67cee6dd04c65a718de16fc8e5954e4b046776e354ee`
- 実行時git commit: `f3cc3c8086dd6ed2029a63d7d7400334170a850e`

研究用metric・MuJoCo真値・残差コードは`submission_template`へ配線しておらず、
PARC提出Policyには混入しない。

## Phase 1ゲート回復試行

no-motion probeではLIBERO Panda controllerのgripper符号は
`+1=close / -1=open`だった。公開LIBERO-Goal v3は状態apertureとの照合から
`0=close / 1=open`と確認し、学習用変換を`g_env = 1 - 2*g_dataset`へ固定した。
推論checkpointは`action_schema.json`を必須とし、暗黙の符号推定を廃止した。
GSRにはclose commandを追加し、open-hookが成功にならないことをtestした。

公開demo auditはmiddle 43 episode / 6,019 frameにclose区間がなく、topは
36 episode / 7,157 frameの全件に1回目のclose、2回目のcloseは4件だけだった。
研究専用MuJoCo supervisorで閉把持demoを各50件生成し、別processのaction
replayで厳密GSR 50/50ずつを確認した。未使用seedの失敗rolloutからさらに
DAgger corrective demoを各20件、同じ方法で検証して追加した。oracle真値と
supervisorは学習データ生成だけに使用し、評価policyへは渡していない。

公開79件、初回sim 100件、DAgger 40件を128 px、front/wrist、8D state、
signed 7D actionへ統一した。top-only / middle-only / balanced 500-step、
balanced 3,000-step、DAgger後balanced 3,000/10,000-stepを学習した。
5-episode選定評価の最良は以下で、90% gateには達しなかった。

- schema修正済み元checkpoint: middle GSR 1/5、top GSR 3/5
- DAgger前3,000-step（20-step gripper hold）: middle 0/5、top 2/5
- DAgger後10,000-step: middle GSR 0/5 / TSR 3/5、top GSR 2/5 / TSR 4/5
- Base normalizer差替え: middle GSR 0/5 / TSR 2/5、top GSR 0/5 / TSR 0/5

主失敗は`hold_failed`で、タスク自体は達成しても閉把持を0.5秒維持せず
open-hookでdrawerを開く挙動が残った。metricやseedは変更していない。
5-episode候補選定gateを通過しなかったため、独立20 episode最終評価と
L1残差RLは実行していない。Phase 1は引き続き未通過である。

回復試行の成果物:

- demo audit: `experiments/libero_residual_rl/artifacts/results/demo_audit.json`
- verified sim manifest:
  `examples/smolvla_libero_spatial_lora_local/work/datasets/experiment_a_oracle_manifest.json`
- recovery dataset manifest SHA-256:
  `3925e6c40de07d09acd344457226ab3c15fa2bb1c36234f1c46f69da22173493`
- 10,000-step merged model SHA-256:
  `af42eba30300ca1cfa3e9c3cf6dab01d12ea1a542ff47e151b2bbf3615984b78`
