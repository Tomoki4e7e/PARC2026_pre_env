import sys
import zipfile
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import validate_submission as vs

_TEMPLATE = _ROOT / "submission_template"


def _codes(report):
    return {code for _, code, _ in report.items}


def _error_codes(report):
    return {code for s, code, _ in report.items if s == vs.ERROR}


def _write_submission(dirpath: Path, policy_src: str | None = None,
                      requirements: str | None = None) -> Path:
    dirpath.mkdir(parents=True, exist_ok=True)
    src = policy_src if policy_src is not None else (_TEMPLATE / "policy_server.py").read_text()
    (dirpath / "policy_server.py").write_text(src)
    req = requirements if requirements is not None else "numpy>=1.19\n"
    (dirpath / "requirements.txt").write_text(req)
    return dirpath


def _zip_dir(dirpath: Path, zip_path: Path, arc_prefix: str = "") -> Path:
    with zipfile.ZipFile(zip_path, "w") as zf:
        for f in sorted(dirpath.rglob("*")):
            if f.is_file():
                zf.write(f, arc_prefix + str(f.relative_to(dirpath)))
    return zip_path


def test_good_zip_passes(tmp_path):
    d = _write_submission(tmp_path / "sub")
    z = _zip_dir(d, tmp_path / "good.zip")
    report = vs.validate_zip(z)
    assert report.ok, report
    assert not report.errors


def test_missing_policy_server(tmp_path):
    d = tmp_path / "sub"
    d.mkdir()
    (d / "requirements.txt").write_text("numpy\n")
    (d / "notes.txt").write_text("hello")
    z = _zip_dir(d, tmp_path / "m.zip")
    report = vs.validate_zip(z)
    assert not report.ok
    assert "struct.no_policy_server" in _error_codes(report)


def test_missing_requirements_is_warning(tmp_path):
    d = tmp_path / "sub"
    d.mkdir()
    (d / "policy_server.py").write_text((_TEMPLATE / "policy_server.py").read_text())
    z = _zip_dir(d, tmp_path / "r.zip")
    report = vs.validate_zip(z)

    assert report.ok, report
    assert "struct.missing_requirements" in _codes(report)


def test_syntax_error_detected(tmp_path):
    d = _write_submission(tmp_path / "sub", policy_src="def broken(:\n  pass\n")
    z = _zip_dir(d, tmp_path / "s.zip")
    report = vs.validate_zip(z)
    assert not report.ok
    assert "policy.syntax" in _error_codes(report)


def test_nested_root_autodescend(tmp_path):
    d = _write_submission(tmp_path / "sub")
    z = _zip_dir(d, tmp_path / "n.zip", arc_prefix="mysub/")
    report = vs.validate_zip(z)
    assert report.ok, report
    assert "struct.nested_root" in _codes(report)


def test_zip_slip_traversal_rejected(tmp_path):
    z = tmp_path / "slip.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("policy_server.py", "print(1)")
        zf.writestr("requirements.txt", "numpy\n")
        zf.writestr("../../etc/pwned.txt", "x")
    report = vs.validate_zip(z)
    assert not report.ok
    assert "zip.slip_traversal" in _error_codes(report)


def test_junk_warning(tmp_path):
    d = _write_submission(tmp_path / "sub")
    (d / ".git").mkdir()
    (d / ".git" / "config").write_text("x")
    z = _zip_dir(d, tmp_path / "j.zip")
    report = vs.validate_zip(z)
    assert "struct.junk" in _codes(report)


@pytest.mark.parametrize("line", [
    "git+https://github.com/foo/bar.git",
    "-e .",
    "--index-url https://evil.example/simple",
    "--extra-index-url https://evil.example/simple",
    "torch @ https://download.example/torch.whl",
])
def test_requirements_external_source_rejected(line):
    report = vs.Report()
    vs.check_requirements(f"numpy>=1.19\n{line}\n", report)
    assert not report.ok, line


def test_requirements_valid_passes():
    report = vs.Report()
    vs.check_requirements("numpy>=1.19\ntorch==2.1.0\n# comment\nfastapi\n", report)
    assert report.ok, report


def test_safe_extract_rejects_traversal(tmp_path):
    z = tmp_path / "slip.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("policy_server.py", "print(1)")
        zf.writestr("../evil.txt", "x")
    with pytest.raises(ValueError):
        vs.safe_extract(z, tmp_path / "out")


def test_safe_extract_nested_returns_root(tmp_path):
    d = _write_submission(tmp_path / "sub")
    z = _zip_dir(d, tmp_path / "n.zip", arc_prefix="mysub/")
    root = vs.safe_extract(z, tmp_path / "out")
    assert (root / "policy_server.py").is_file()


def _has_dynamic_deps() -> bool:
    try:
        import numpy, msgpack, requests, fastapi, uvicorn
        return True
    except Exception:
        return False


dynamic = pytest.mark.skipif(not _has_dynamic_deps(),
                             reason="動的スモークの依存が無い")


@dynamic
def test_smoke_template_passes(tmp_path):
    d = _write_submission(tmp_path / "sub")
    report = vs.smoke_test(d, port=8941, health_timeout=30)

    assert report.ok, report
    assert "smoke.act_ok" in _codes(report)


@dynamic
def test_smoke_nan_action_rejected(tmp_path):
    src = (_TEMPLATE / "policy_server.py").read_text().replace(
        "return np.random.uniform(-1, 1, size=7).astype(np.float32)",
        "return np.array([float('nan')]+[0.0]*6, dtype=np.float32)")
    d = _write_submission(tmp_path / "sub", policy_src=src)
    report = vs.smoke_test(d, port=8942, health_timeout=30)
    assert not report.ok
    assert "smoke.act_nan" in _error_codes(report)


@dynamic
def test_smoke_bad_shape_rejected(tmp_path):
    src = (_TEMPLATE / "policy_server.py").read_text().replace(
        "return np.random.uniform(-1, 1, size=7).astype(np.float32)",
        "return np.zeros(4, dtype=np.float32)")
    d = _write_submission(tmp_path / "sub", policy_src=src)
    report = vs.smoke_test(d, port=8943, health_timeout=30)
    assert not report.ok
    assert "smoke.act_shape" in _error_codes(report)


@pytest.mark.parametrize("line", [
    "evilpkg @ file:./evilpkg",
    "foo @ file:/tmp/evilpkg",
    "foo @ git://evil/repo",
    "foo @ ftp://evil/pkg.tgz",
    "foo @ ssh://evil/repo",
    "-r other.txt",
    "-c constraints.txt",
    "./localpkg",
])
def test_req_external_source_variants_rejected(line):
    report = vs.Report()
    vs.check_requirements(f"numpy>=1.19\n{line}\n", report)
    assert not report.ok, f"バイパス許容: {line!r}\n{report}"


def test_req_continuation_no_space_join():

    report = vs.Report()
    vs.check_requirements("num\\\npy==1.0.0\n", report)
    assert report.ok, report


@pytest.mark.parametrize("text", [
    "﻿numpy==1.0.0\n",
    "numpy==1.0.0 --hash=sha256:abcd\n",
    "--no-index\nnumpy==1.0.0\n",
    "--pre\nnumpy==1.0.0\n",
    "--only-binary=:all:\nnumpy==1.0.0\n",
])
def test_req_legit_not_rejected(text):
    report = vs.Report()
    vs.check_requirements(text, report)
    assert report.ok, f"誤検知: {text!r}\n{report}"


def test_bom_policy_server_passes(tmp_path):
    src = "﻿" + (_TEMPLATE / "policy_server.py").read_text()
    d = _write_submission(tmp_path / "sub", policy_src=src)
    z = _zip_dir(d, tmp_path / "bom.zip")
    report = vs.validate_zip(z)
    assert report.ok, report
    assert "policy.syntax" not in _error_codes(report)


def test_binary_policy_server_rejected(tmp_path):
    d = tmp_path / "sub"
    d.mkdir()
    (d / "policy_server.py").write_bytes(b"import os\x00\x00\nprint(1)\n")
    (d / "requirements.txt").write_text("numpy\n")
    z = _zip_dir(d, tmp_path / "bin.zip")
    report = vs.validate_zip(z)
    assert not report.ok, report
    assert "policy.syntax" in _error_codes(report)


@pytest.mark.parametrize("name", ["myvenv/util.py", "solvenv/core.py",
                                  "assets/.DS_Store_notes.txt", "my.git/config"])
def test_junk_no_substring_false_positive(name):
    report = vs.Report()
    vs._check_junk([name, "policy_server.py"], report)
    assert "struct.junk" not in _codes(report), f"誤爆: {name}\n{report}"


def test_junk_real_still_detected():
    report = vs.Report()
    vs._check_junk([".git/config", "policy_server.py"], report)
    assert "struct.junk" in _codes(report)


def test_encrypted_zip_no_crash(tmp_path):
    import subprocess
    d = _write_submission(tmp_path / "sub")
    z = tmp_path / "enc.zip"

    r = subprocess.run(["zip", "-P", "secret", "-j", str(z),
                        str(d / "policy_server.py"), str(d / "requirements.txt")],
                       capture_output=True)
    if r.returncode != 0:
        pytest.skip("zip コマンドが使えない")
    report = vs.validate_zip(z)
    assert not report.ok
    assert any(c.startswith("zip.") for c in _error_codes(report)), report


def test_validate_dir_nested_descend(tmp_path):
    d = _write_submission(tmp_path / "wrap" / "mysub")
    report = vs.validate_dir(tmp_path / "wrap")
    assert report.ok, report


@dynamic
def test_smoke_reaps_grandchildren(tmp_path):
    psutil = pytest.importorskip("psutil")
    marker = "GRANDCHILD_CANARY_PYTEST"
    src = f'''import subprocess, sys, argparse
subprocess.Popen([sys.executable, "-c",
    "import time\\nwhile True: time.sleep(1)  # {marker}"])
p = argparse.ArgumentParser(); p.add_argument("--port", type=int, default=8000)
a = p.parse_args()
import uvicorn
from fastapi import FastAPI
app = FastAPI()

@app.get("/health")
def h(): return {{"status": "ok"}}
uvicorn.run(app, host="0.0.0.0", port=a.port, log_level="warning")
'''
    d = _write_submission(tmp_path / "sub", policy_src=src,
                          requirements="fastapi\nuvicorn\n")

    def _canaries():
        found = []
        for p in psutil.process_iter(["cmdline"]):
            try:
                if any(marker in c for c in (p.info["cmdline"] or [])):
                    found.append(p.pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return found

    before = set(_canaries())
    vs.smoke_test(d, port=8945, health_timeout=20, act_timeout=5)
    import time
    time.sleep(1.0)
    leaked = set(_canaries()) - before

    for pid in leaked:
        try:
            psutil.Process(pid).kill()
        except psutil.NoSuchProcess:
            pass
    assert not leaked, f"孫プロセスが残留: {leaked}"


@dynamic
def test_smoke_float64_hint(tmp_path):
    src = (_TEMPLATE / "policy_server.py").read_text()

    src = src.replace('{"data": action.astype(np.float32).tobytes()}',
                      '{"data": action.astype(np.float64).tobytes()}')
    d = _write_submission(tmp_path / "sub", policy_src=src)
    report = vs.smoke_test(d, port=8944, health_timeout=30)
    assert not report.ok
    msgs = " ".join(m for _, _, m in report.items)
    assert "float64" in msgs, report
