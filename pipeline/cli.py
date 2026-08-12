import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from .config import EvalConfig, Track
from .pipeline import EvaluationPipeline
from .rollout import PolicyInterface


class RandomPolicy:

    def __init__(self, action_dim: int = 7):
        self.action_dim = action_dim

    def get_action(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        return np.random.uniform(-1, 1, size=self.action_dim).astype(np.float32)

    def reset(self, instruction: str = "", seed: int | None = None) -> None:
        if seed is not None:
            np.random.seed(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VLAコンペティション評価パイプライン"
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default=None,
        help="ポリシーサーバーのURL (例: http://localhost:8000)",
    )
    parser.add_argument(
        "--submission",
        type=Path,
        help="提出物のディレクトリパス（非推奨: --server-url を使用）",
    )
    parser.add_argument(
        "--track",
        nargs="+",
        choices=["track1"],
        default=["track1"],
        help="評価するトラック",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        help="ベンチマーク名を直接指定（テスト用）",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=20,
        help="タスクあたりの評価エピソード数",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=300,
        help="エピソードあたりの最大ステップ数",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="評価するタスク数の上限（指定しなければ全タスク）",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        metavar="TASK_ID",
        help="タスク名を指定して評価する（スペース区切りで複数可）。"
             "存在しない名前はエラーになり、利用可能なタスク名を表示する",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="ポリシーサーバーの推論タイムアウト（秒、/act 1リクエストあたり）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="乱数シード",
    )
    parser.add_argument(
        "--init-state-offset",
        type=int,
        default=0,
        help="ローカル収集用の初期状態ordinal開始位置（既定の評価は0）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="ランダムポリシーでドライラン",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="結果出力ディレクトリ",
    )
    parser.add_argument(
        "--local-rl-record-dir",
        type=Path,
        default=None,
        help="ローカルシミュレータのRL遷移をNPZ保存するディレクトリ",
    )
    parser.add_argument(
        "--allow-local-data-recording",
        action="store_true",
        help="ローカル生成データであることを確認して記録を明示許可する",
    )
    parser.add_argument(
        "--video-dir", type=Path, default=None,
        help="学習済みポリシーのローカルrolloutをMP4保存するディレクトリ",
    )
    parser.add_argument(
        "--video-camera", choices=("agent", "wrist", "both"), default="agent",
        help="動画カメラ（既定: agent）",
    )
    parser.add_argument(
        "--video-fps", type=float, default=20.0,
        help="動画のFPS（既定: 20）",
    )
    parser.add_argument(
        "--video-frame-skip", type=int, default=1,
        help="N simulator stepsごとに1フレーム記録（既定: 1）",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="詳細ログを出力",
    )
    args = parser.parse_args()
    if args.local_rl_record_dir is not None and not args.allow_local_data_recording:
        parser.error(
            "--local-rl-record-dir には --allow-local-data-recording が必要です。"
            "運営評価の観測は保存・学習利用できません。"
        )
    if args.init_state_offset < 0:
        parser.error("--init-state-offset must be >= 0")
    if args.video_fps <= 0 or args.video_frame_skip <= 0:
        parser.error("--video-fps と --video-frame-skip は正の値が必要です")
    return args


def main() -> None:
    args = parse_args()


    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


    config = EvalConfig(
        n_eval_episodes=args.n_episodes,
        max_steps_per_episode=args.max_steps,
        seed=args.seed,
        init_state_offset=args.init_state_offset,
        rl_record_dir=args.local_rl_record_dir,
        allow_local_data_recording=args.allow_local_data_recording,
        video_dir=args.video_dir,
        video_camera=args.video_camera,
        video_fps=args.video_fps,
        video_frame_skip=args.video_frame_skip,
    )
    if args.output_dir is not None:
        config.output_dir = args.output_dir

    if args.max_tasks is not None:
        config.max_tasks = args.max_tasks
    if args.tasks:
        config.task_ids = args.tasks
    if args.benchmark:
        config.benchmark_name = args.benchmark


    track_map = {
        "track1": Track.TRACK1,
    }
    tracks = [track_map[t] for t in args.track]


    pipeline = EvaluationPipeline(eval_config=config)


    if args.dry_run:
        logging.info("ドライラン: ランダムポリシーを使用")
        policy: PolicyInterface = RandomPolicy()
        submission_id = "dry_run"
        skip_validation = True

    elif args.server_url:
        from .remote_policy import RemotePolicyClient

        logging.info("リモートポリシーサーバーに接続: %s", args.server_url)
        client = RemotePolicyClient(server_url=args.server_url, timeout_sec=args.timeout)
        client.wait_for_server()
        policy = client
        submission_id = f"server_{args.server_url.split(':')[-1]}"
        skip_validation = True

    elif args.submission:
        logging.error(
            "提出物からのポリシーロードは未実装。"
            "--dry-run または --server-url を使用してください。"
        )
        sys.exit(1)
    else:
        logging.error(
            "--dry-run, --server-url, --submission のいずれかを指定してください。"
        )
        sys.exit(1)


    submission_dir = args.submission
    if skip_validation:
        submission_dir = Path("/tmp/pipeline_submission")
        submission_dir.mkdir(parents=True, exist_ok=True)
        (submission_dir / "config.yaml").write_text("# auto-generated\n")
        (submission_dir / "policy.py").write_text("# auto-generated\n")

    result = pipeline.run(
        submission_dir=submission_dir,
        policy=policy,
        tracks=tracks,
        submission_id=submission_id,
        benchmark_override=args.benchmark,
        skip_validation=skip_validation,
    )


    print("\n" + "=" * 60)
    print(f"提出ID: {result.submission_id}")
    print(f"合計時間: {result.total_elapsed_sec:.1f}秒")
    for ts in result.track_scores:
        print(f"\n  {ts.track.value}: 総合スコア {ts.overall_score:.3f}")
        for task_score in ts.task_scores:
            print(f"    {task_score.task_name}: {task_score.success_rate:.1%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
