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

## Learned-gripper head回復試行

全7D LoRAではclose frameが軌道全体lossへ埋もれたため、v4 SmolVLAの
arm 6軸を固定し、4-frameのfront+wrist、8D state、base actionから
gripper closeだけを推定するCNN+GRU headを追加した。MuJoCo contact等は
教師生成と評価だけに使用し、headの推論入力には含めていない。unit testで
arm 6軸がbitwise不変であることを確認した。

別seedでstrict-grasp成功軌道をmiddle/top各200件収集し、全400件を別process
でaction replayしてGSRを再確認した。合計42,854 frameで、close教師は
middle 8,650、top 7,926 frameだった。raw Base action、supervised action、
contact、closed、joint displacement、phaseを保存し、320/80 episodeへ
分離して学習した。

held-out分類ではtwo-camera headが最良だった。

- vector: TP 2,554 / FP 87 / FN 127
- front+vector: TP 2,626 / FP 41 / FN 55
- wrist: TP 2,601 / FP 47 / FN 80
- two-camera: TP 2,607 / FP 39 / FN 74

同一未使用seed各5 episodeのclosed-loop比較:

- vector: middle GSR 3/5、top GSR 4/5
- front+vector: middle GSR 4/5・TSR 4/5、top GSR 5/5・TSR 0/5
- wrist: middle GSR 3/5、top GSR 5/5
- two-camera: middle GSR 4/5、top GSR 5/5

one-round DAggerとして各task 20成功軌道と各20件までのhard negativeを追加し、
candidate失敗rolloutも追加したが、結果はmiddle 4/5、top 5/5で変わらなかった。
middleの残る1件はcontact 0の`target_missed`であり、gripper-only headでは
arm approachを修正できなかった。候補gateの両task 5/5条件を満たさないため、
独立20 episode最終gateは実行していない。Phase 1は未通過のままである。

dataset buildも固定10 episodeで実測した。LeRobot featureを正しく`video`に
した上で、SVT-AV1は18.58 frame/s、CPU H.264 streamingは19.62 frame/sだった
（先頭frame PSNRはそれぞれ38.75 dB、36.43 dB）。
NVENCはPyAVのencoder一覧には存在したが、実encode時に
`avcodec_open2(h264_nvenc): Function not implemented`で失敗した。この環境では
CPU H.264 streamingを採用し、最大の高速化は公開79件の再decodeを避け、
新規shardだけを`resume` appendする構成とした。

追加成果物:

- dataset benchmark:
  `experiments/libero_residual_rl/artifacts/results/dataset_build_benchmark.json`
- labeled dataset manifest SHA-256:
  `81629ef30b4dd4eb3affa7e725fd13bc2048db8ffca9e2f0ccd856396030010a`
- selected two-camera head SHA-256:
  `fc25a36c443cbb5f5d2bcbc1a5518ee457e51ca3ef4fde97839d582aa3938a01`
- DAgger head SHA-256:
  `0e900e0d1bcd5ec85c9432610bdeabaa33953a4f880ef9546907ba8dcc58a5b0`

## Front arm-residual回復試行

middleの残存`target_missed`へ対応するため、Frozen SmolVLAのraw actionを条件に
front画像4 frameと8D stateだけから6D arm residualを出すCNN+GRU headを追加した。
追加headはwrist画像、handle座標、contact、rewardを推論入力に使用しない。
topはhandleのstrict GSRだけを評価し、bowl配置TSRは算出・gate判定しなかった。

既存400件のgripper教師はarm 6軸がBaseと同一で残差教師がゼロだったため、
新しくbounded arm-correction supervisorを用いた。これはデータ生成時だけ
MuJoCo handle位置を使い、Baseの回転actionを維持しながらtranslation correctionを
付加する。trainはmiddle 200/top 100、独立validationは各50件を収集し、全400件を
別processでstrict GSR replay検証した。trainは29,828 frame（approach教師20,880）、
validationは10,019 frame（approach教師6,821）だった。

primary headの最終held-out MAEは0.00379、approach accuracyは0.935だったが、
未使用candidate seed各5件のclosed-loop GSRはmiddle 2/5、top 2/5となった。
offline誤差とclosed-loop性能が乖離し、状態分布ずれにより`hold_failed`と
`target_missed`が増加した。

計画どおりDAggerを1回だけ実施し、combined policyの訪問stateから
middle 80/top 50件を追加してrefinement headを学習した。使用済みcandidate seedは
廃棄し、新規seedで再評価したがmiddle 2/5、top 1/5だった。別validation seedで
arm scale 0.10/0.05を比較した結果はいずれもmiddle 4/5、top 4/5で、両task 5/5の
候補gateを満たさなかった。metricやseedを変更せず、独立20 episode最終gateは
実行していない。したがってPhase 1は未通過であり、L1残差RLも再開していない。

成果物:

- seed protocol SHA-256:
  `82f56c0577523cbfbaddc2ba2a0a1fa037cd9fbfd3d57a82b97c5a2d47770cdb`
- primary arm head SHA-256:
  `9247a4a9215c85e08602e1bd55f4c7a8e9098f851f7936a01d2963f5c549260d`
- DAgger refinement head SHA-256:
  `0e544db977e2d3e7b2d5e582401491a8288a6550efbf3e16214fcb47d4842ed1`
- train/validation/DAgger manifest SHA-256:
  `7b3ae273e72594b6aa1f3d52bf2183e2c3d0648e2ebb830a54b71bc8614151c4`,
  `d31ccc77d482a87a619c066037e970dd6b3d8cb4429050d271d81b4a0ceea568`,
  `9f6935a8533abf5cc0b31112c6f87bb0844031864396d4b91b2af0e9407a14ec`

## v5 LoRA BC回復試行

arm residual headを最終構成から外し、v4 merged SmolVLAへcurated
arm-correction軌道を追加LoRAした。train 300件とDAgger-train 104件を監査し、
overlong 21件、top early-close 4件を除外した。採用379件へ、arm 6軸が
v4 Baseと一致するclean anchorをmiddle/top各50件追加した。最終datasetは
479 episode・42,331 frame（middle 303、top 176）で、validation/candidate/final
seedとの重複は0だった。H.264 streaming構築は1,451.8秒、29.16 frame/s、
復号画像の先頭frame PSNRは36.42 dBだった。

v4 mergedをBaseにrank/alpha 16、vision frozen、LR 5e-5→5e-6で
12,000 step学習し、2,000 step間隔の全6 checkpointを固定validation seed各5件で
Base-onlyとfrozen front gripper-onlyの両方について比較した。gripper-onlyが全点で
Base-onlyを上回ったが、最良はstep 2,000/6,000のmiddle 3/5・top 4/5だった。
同じseedのv4+gripper baselineはmiddle 4/5・top 4/5であり、追加LoRAによる
再現可能な改善は確認できなかった。v4 normalizerは学習・merge後とも同一hashを維持し、
merge後にLoRA keyが残っていないこと、8D state、2-camera Base入力、
signed gripper schemaを確認した。最大推論時間は0.43秒未満だった。

step 2,000 + frozen front gripper headを未使用candidate seedで一度だけ評価した結果は
middle 4/5（`hold_failed` 1）、top 3/5（`target_missed` 1、
`wrong_approach` 1）で、両task 5/5条件を満たさなかった。v5 Base上で
gripper教師を追加200件収集しfront headを1回だけ再学習したが、固定validationは
middle 3/5・top 4/5で改善しなかったため不採用とした。candidate失敗後にseedやmetricを
変更せず、独立20 episode final gateは実行していない。従ってPhase 1は未通過であり、
L1 Flow Matching残差RLは再開していない。top bowl TSRは全選択・gateから除外した。

成果物:

- v5 dataset manifest SHA-256:
  `9137711ee27f984e60b55a609f6fbb7d20db5ce2bd19d366545edd8e4b4d506a`
- selected step 2,000 adapter SHA-256:
  `7ecd1108912bf72e14e99b9bc7a0952cd8e6bb48f525ff8664b574efc538955f`
- selected merged model SHA-256:
  `51b4cdbe53d166866e2ff09c410ac1671c4db5f9b462968442948cf96f70e1b0`
- frozen / retrained front gripper head SHA-256:
  `92f09c0b2ea568d73c57201330c37fa2a4645d24d80a79bd090418591954241b`,
  `db95ae0a6a3c61b7128a6790592cf0259f2046e713038d33ee213034353e5173`
- seed protocol SHA-256:
  `714d681e840348092c74e544f397dd76a675d9e72552d56119f0aea4ac73a297`
- v4/v5 normalizer SHA-256:
  `d27527837aa9ead2c444dc642a37b26a577e7dc540084944af8a055b59e27643`

## Action horizon回復試行

全405 evaluation episodeを再集計すると、失敗222件のうち
`hold_failed`が115件（51.8%）、`target_missed`が95件（42.8%）だった。
評価器の境界例は1件だけであり、主因はpolicy側と判断した。SmolVLAは
`n_action_steps=50`、20 Hzで2.5秒をopen-loop実行していたため、50-action予測は
維持しつつ、実行horizonをtask別に変更できるようにした。

固定validation seedで、middleはhorizon 5、topはhorizon 10が最良だった。
さらに学習済みtwo-camera gripper headのclose判定中だけarm 6軸を0.5倍へ減衰し、
接触後のovershootを抑えた。この信号は画像、8D state、Base actionから学習済みで、
MuJoCo contactやhandle位置を推論時には使用しない。

未使用development seed各20 episode:

- middle: GSR 19/20（95%）
- top: GSR 18/20（90%）

この構成を固定して、事前登録済みの未使用final seed各20 episodeを一度だけ評価した。

- middle: GSR 16/20（80%）。`target_missed` 1、`hold_failed` 3
- top: GSR 18/20（90%）。`target_missed` 1、`wrong_approach` 1
- 最大Base推論時間: middle 0.196秒、top 0.164秒

development gateでは両task 90%へ到達したが、独立finalでmiddleが未達だったため、
Phase 1は通過扱いにしない。final結果を用いた再選定・追加学習も行わない。
今後の再学習向けには、arm-correction teacherが現在のhandle位置からpregraspを
毎step更新し、strict GSR達成時点で軌道をtrimしてrelease教師を混入させず、
200 frame超の有効軌道を保持するようデータ生成を修正した。

成果物:

- summary:
  `experiments/libero_residual_rl/artifacts/results/phase1_replan_gate_summary.json`
- development:
  `experiments/libero_residual_rl/artifacts/evaluation/v4_gripper_damped05_development20.jsonl`
- final:
  `experiments/libero_residual_rl/artifacts/evaluation/v4_gripper_damped05_final20.jsonl`
- model / gripper head SHA-256:
  `af42eba30300ca1cfa3e9c3cf6dab01d12ea1a542ff47e151b2bbf3615984b78`,
  `fc25a36c443cbb5f5d2bcbc1a5518ee457e51ca3ef4fde97839d582aa3938a01`
