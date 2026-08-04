import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import evaluate as ev


def _write_result(dirpath: Path, submission_id: str, tracks: list[dict]) -> None:
    (dirpath / f"{submission_id}.json").write_text(
        json.dumps({"submission_id": submission_id, "tracks": tracks}),
        encoding="utf-8",
    )


def _track(name: str, score: float, **metrics) -> dict:
    return {
        "track": name,
        "benchmark": "libero_t1",
        "overall_score": score,
        "overall_metrics": {"mean_success_rate": score, **metrics},
        "tasks": [],
    }


def test_track1_score_is_reported(tmp_path):
    _write_result(tmp_path, "server_8000", [_track("track1", 0.75)])

    r = ev.build_omnicampus_result(str(tmp_path), "server_8000")

    assert r["public_score"] == pytest.approx(0.75)
    assert r["base_passed"] is True
    assert "Score: 0.75" in r["course_top_message"]


def test_genuine_zero_score_is_not_treated_as_infra_failure(tmp_path):
    _write_result(tmp_path, "server_8000", [_track("track1", 0.0)])

    r = ev.build_omnicampus_result(str(tmp_path), "server_8000")

    assert r["public_score"] == 0.0
    assert r["base_passed"] is False
    assert "Success rate: 0%" in r["course_top_message"]


def test_errored_track_raises_instead_of_reporting_zero(tmp_path):
    _write_result(tmp_path, "server_8000", [_track("track1", 0.0, error=1.0)])

    with pytest.raises(ev.EvaluationInfraError) as exc:
        ev.build_omnicampus_result(str(tmp_path), "server_8000")

    assert "track1" in str(exc.value)


def test_missing_result_file_returns_error_result(tmp_path):
    r = ev.build_omnicampus_result(str(tmp_path), "server_8000")

    assert r["public_score"] == 0.0
    assert "結果ファイルが見つかりません" in r["course_top_message"]


def test_empty_tracks_returns_error_result(tmp_path):
    _write_result(tmp_path, "server_8000", [])

    r = ev.build_omnicampus_result(str(tmp_path), "server_8000")

    assert "トラック結果がありません" in r["course_top_message"]
