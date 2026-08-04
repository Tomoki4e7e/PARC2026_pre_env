import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import SubmissionConstraints

logger = logging.getLogger(__name__)


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    import validate_submission as _vs
except Exception as e:
    _vs = None
    logger.warning("validate_submission モジュールを読み込めません: %s", e)


@dataclass
class ValidationResult:
    is_valid: bool
    errors: list[str]
    warnings: list[str]

    def __str__(self) -> str:
        status = "PASS" if self.is_valid else "FAIL"
        lines = [f"バリデーション結果: {status}"]
        for e in self.errors:
            lines.append(f"  [ERROR] {e}")
        for w in self.warnings:
            lines.append(f"  [WARN]  {w}")
        return "\n".join(lines)


class SubmissionValidator:

    def __init__(self, constraints: SubmissionConstraints | None = None):
        self.constraints = constraints or SubmissionConstraints()

    def validate(self, submission_dir: Path) -> ValidationResult:
        if _vs is None:

            errors = []
            if not Path(submission_dir).exists():
                errors.append(f"提出ディレクトリが存在しない: {submission_dir}")
            elif not (Path(submission_dir) / "policy_server.py").is_file():
                errors.append("policy_server.py が見つからない")
            return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=[])

        report = _vs.validate_dir(submission_dir)
        return ValidationResult(
            is_valid=report.ok,
            errors=[f"{code}: {msg}" for _, code, msg in report.items if _ == _vs.ERROR],
            warnings=[f"{code}: {msg}" for _, code, msg in report.items if _ == _vs.WARN],
        )
