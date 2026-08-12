import importlib.util
from pathlib import Path


def _load_server():
    path = Path(__file__).resolve().parents[1] / "submission_template" / "policy_server.py"
    spec = importlib.util.spec_from_file_location("test_policy_server", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_health_reports_loaded_model_provenance():
    server = _load_server()

    class Impl:
        model_path = "/models/selected"
        model_sha256 = "abc123"
        _residual_policy = None
        _collision_critic = object()

    class Policy:
        _impl = Impl()

    server.set_policy(Policy())
    assert server.health() == {
        "status": "ok",
        "policy": "Impl",
        "model_path": "/models/selected",
        "model_sha256": "abc123",
        "residual_enabled": False,
        "collision_critic_enabled": True,
    }
