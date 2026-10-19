"""Windows launch checks and optional Git provenance; never starts learning."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


class StartupPreflightError(RuntimeError):
    """A common startup prerequisite failed before any IA was launched."""

    def __init__(self, message, report=None):
        super().__init__(message)
        self.report = report or {}


def git_metadata(repo):
    """Git is optional provenance, outside checkpoint compatibility material."""
    executable = shutil.which("git")
    result = {"git_commit": None, "git_metadata_status": "unavailable",
              "git_executable": executable, "git_metadata_error": None}
    if executable is None:
        result["git_metadata_error"] = "git executable is not on PATH; source/data SHA256 provenance remains available"
        return result
    try:
        probe = subprocess.run([str(Path(executable).resolve()), "rev-parse", "HEAD"],
            cwd=str(Path(repo).resolve()), capture_output=True, text=True,
            encoding="utf-8", errors="replace", check=False, timeout=5,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_OPTIONAL_LOCKS": "0"},
            shell=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        result["git_metadata_error"] = f"{type(error).__name__}: {error}"
        return result
    if probe.returncode == 0 and probe.stdout.strip():
        result.update(git_commit=probe.stdout.strip(), git_metadata_status="available")
    else:
        result["git_metadata_error"] = probe.stderr.strip() or f"git exited with code {probe.returncode}"
    return result


def _writable_directory(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    # Only this uniquely named probe file is removed; existing artifacts remain.
    with tempfile.NamedTemporaryFile(prefix=".launch_probe_", dir=path, delete=True) as handle:
        handle.write(b"launch path check")
        handle.flush()


def shared_launch_preflight(repo, output_root, config_dir=None):
    """Validate common paths and an absolute-Python import child, without learning."""
    repo = Path(repo).resolve()
    output = Path(output_root)
    if not output.is_absolute(): output = repo / output
    output = output.resolve()
    python = Path(sys.executable).resolve()
    report = {"status": "checking", "python_executable": str(python), "cwd": str(repo),
              "output_root": str(output), "child_shell": False,
              "environment": {"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
              "learning_started": False}
    try:
        if not repo.is_dir(): raise FileNotFoundError(f"Repository cwd does not exist: {repo}")
        required = [python, repo / "sustaincity_iagent/data/train_data.xlsx",
                    repo / "sustaincity_iagent/data/alternative_product_registry.json"]
        sources = ("alternative_training.py", "alternative_launch.py", "alternative_env.py",
                   "alternative_factory.py", "alternative_metrics.py", "alternative_rewards.py",
                   "alternative_evaluation.py")
        required += [repo / "sustaincity_iagent" / name for name in sources]
        if config_dir is not None:
            config_dir = Path(config_dir)
            if not config_dir.is_absolute(): config_dir = repo / config_dir
            config_dir = config_dir.resolve()
            if not config_dir.is_dir(): raise FileNotFoundError(f"Configuration directory does not exist: {config_dir}")
            required.append(config_dir / "batch_manifest.json")
            report["config_dir"] = str(config_dir)
        for path in required:
            if not path.is_file(): raise FileNotFoundError(f"Required startup file does not exist: {path}")
            with path.open("rb") as handle: handle.read(1)
        report["required_files"] = [str(path) for path in required]
        _writable_directory(output)
        _writable_directory(repo / "experiments/alternative_primary_objectives/logs/training_locks")
        script = (
            "import json,os,sys; from stable_baselines3 import PPO; "
            "from sustaincity_iagent.alternative_training import REPO_ROOT; "
            "print(json.dumps({'python_executable':sys.executable,'cwd':os.getcwd(),"
            "'repository':str(REPO_ROOT),'PPO_import':'passed','learning_started':False}))"
        )
        command = [str(python), "-B", "-X", "utf8", "-c", script]
        report["child_command"] = command
        child = subprocess.run(command, cwd=str(repo), shell=False, capture_output=True,
            text=True, encoding="utf-8", errors="replace", timeout=60,
            env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"})
        report.update(child_returncode=child.returncode, child_stdout=child.stdout,
                      child_stderr=child.stderr)
        if child.returncode != 0:
            raise RuntimeError(f"Project Python import subprocess failed with exit code {child.returncode}:\n{child.stderr}")
        report["child_probe"] = json.loads(child.stdout.strip().splitlines()[-1])
        if Path(report["child_probe"]["repository"]).resolve() != repo:
            raise RuntimeError("Child imported a different repository; refusing an ambiguous package installation")
        report.update(status="passed", **git_metadata(repo))
        return report
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise StartupPreflightError(f"Shared launch preflight failed: {error}", report) from error
