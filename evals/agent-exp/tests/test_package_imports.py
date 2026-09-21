"""Bảo vệ import của các package semantic trước module phẳng cũ."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "evals/agent-exp/scripts"
SUBPEXP = SCRIPTS / "diagnostic_subexp"
ENTRYPOINTS = (
    SCRIPTS / "run_experiments.py",
    SCRIPTS / "run_fixed_diagnostics_ab.py",
    SCRIPTS / "run_prompt_ablation.py",
    SUBPEXP / "candidate_pair_41_3_position" / "cli.py",
    SUBPEXP / "article_42_observation_block_order" / "cli.py",
    SUBPEXP / "doctoral_defense_prompt_ablation" / "cli.py",
)


def _env(fake_dir: Path) -> dict[str, str]:
    return {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTHONPATH": os.pathsep.join((str(fake_dir), str(SCRIPTS), str(ROOT))),
    }


def test_semantic_imports_ignore_legacy_collision_modules(tmp_path: Path):
    """Package import không resolve các tên module phẳng giả."""
    for name in (
        "diagnostic_contracts.py",
        "diagnostic_artifacts.py",
        "diagnostic_metrics.py",
        "diagnostic_policy.py",
        "prompt_ablation_contracts.py",
    ):
        (tmp_path / name).write_text("raise AssertionError('legacy flat import')\n")
    code = (
        "import diagnostic_subexp.shared.contracts; "
        "import diagnostic_subexp.candidate_pair_41_3_position.cli; "
        "import diagnostic_subexp.article_42_observation_block_order.cli; "
        "import diagnostic_subexp.doctoral_defense_prompt_ablation.cli"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, env=_env(tmp_path), capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_public_entrypoint_help_works_from_root_and_unrelated_directory(tmp_path: Path):
    """Mọi entrypoint công khai tự bootstrap path độc lập với cwd."""
    for entrypoint in ENTRYPOINTS:
        for cwd in (ROOT, tmp_path):
            result = subprocess.run(
                [sys.executable, str(entrypoint), "--help"],
                cwd=cwd,
                env=_env(tmp_path),
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, (entrypoint, cwd, result.stderr)
