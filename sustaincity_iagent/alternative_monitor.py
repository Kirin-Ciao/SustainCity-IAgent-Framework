"""Read-only training monitor with bounded retry and visible reader errors.

Training artifacts are never repaired, rewritten or classified as failed by
this monitor. Last successfully read rows survive transient Windows locks.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys
import time

from .alternative_training import REPO_ROOT, display_status


class MonitorReadError(RuntimeError):
    def __init__(self, path, cause, attempts, category):
        self.path = Path(path)
        self.cause = cause
        self.attempts = attempts
        self.category = category
        super().__init__(f"{self.path}: {type(cause).__name__}: {cause}")

    def issue(self, run_id=None):
        return {"run_id": run_id, "path": str(self.path), "category": self.category,
                "severity": "warning" if self.category == "temporarily_unavailable" else "error",
                "attempts": self.attempts, "error_type": type(self.cause).__name__,
                "message": str(self.cause)}


def _reject_nonfinite_json(value):
    raise ValueError(f"Nonfinite JSON literal is invalid monitor data: {value}")


def read_json_retry(path, *, attempts=3, retry_delay=.02, sleeper=None):
    """Retry short access, disappearance and incomplete-JSON races only."""
    if not isinstance(attempts, int) or not 1 <= attempts <= 10:
        raise ValueError("Read attempts must be an integer between 1 and 10")
    if not math.isfinite(retry_delay) or not 0 <= retry_delay <= 1:
        raise ValueError("Retry delay must be finite and between 0 and 1 second")
    path = Path(path)
    sleeper = sleeper or time.sleep
    for index in range(attempts):
        try:
            value = json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=_reject_nonfinite_json)
            if not isinstance(value, dict):
                raise ValueError("Monitor JSON root must be an object")
            return value
        except (PermissionError, FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError) as error:
            if index + 1 < attempts:
                sleeper(retry_delay)
                continue
            category = "temporarily_unavailable" if isinstance(error, (PermissionError, FileNotFoundError)) else "invalid_json"
            raise MonitorReadError(path, error, attempts, category) from error
        except (OSError, ValueError) as error:
            raise MonitorReadError(path, error, index + 1, "invalid_data" if isinstance(error, ValueError) else "read_error") from error


def _finite_number(value, name, *, integer=False):
    if value is None: return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if integer and (value != int(value) or value < 0):
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _merge(run, status, progress, now):
    if not isinstance(status.get("status"), str) or not status["status"]:
        raise ValueError("status.json must contain a nonempty status string")
    steps = status.get("actual_timesteps", status.get("timesteps"))
    target = status.get("target_timesteps", status.get("target"))
    _finite_number(steps, "timesteps", integer=True)
    _finite_number(target, "target", integer=True)
    updated = _finite_number(progress.get("updated_at_epoch"), "updated_at_epoch")
    for name in ("mean_episode_reward", "approx_kl", "policy_gradient_loss", "elapsed_seconds",
                 "fps", "eta_seconds", "percent_complete", "value_loss", "entropy_loss", "explained_variance"):
        _finite_number(progress.get(name), name)
    checkpoint = status.get("latest_checkpoint") or status.get("checkpoint") or progress.get("latest_checkpoint")
    if checkpoint is not None and not isinstance(checkpoint, str):
        raise ValueError("latest_checkpoint must be a path string or null")
    age = max(0., now - updated) if updated is not None else None
    return {**progress, "run_id": run.name, "status": status["status"],
            "timesteps": steps, "target": target,
            "latest_checkpoint": checkpoint,
            "error": status.get("error"), "scientific_acceptance": status.get("scientific_acceptance"),
            "last_update_age_seconds": age,
            "stale_or_interrupted": bool(age is not None and age > 30 and status["status"] in ("running", "evaluating")),
            "monitor_read_state": "fresh", "monitor_last_success_epoch": now}


class ReadOnlyTrainingMonitor:
    """One watcher owns an in-memory cache; its training source is read only."""
    def __init__(self, output_root, *, attempts=3, retry_delay=.02, sleeper=None):
        self.output_root = Path(output_root)
        self.attempts = attempts
        self.retry_delay = retry_delay
        self.sleeper = sleeper
        self.last_known = {}

    def _read(self, path):
        return read_json_retry(path, attempts=self.attempts, retry_delay=self.retry_delay, sleeper=self.sleeper)

    def _fallback(self, run, issue, now):
        if run.name in self.last_known:
            row = copy.deepcopy(self.last_known[run.name])
            row["monitor_read_state"] = "last_known"
            updated = row.get("updated_at_epoch")
            row["last_update_age_seconds"] = max(0., now - updated) if updated is not None else None
            row["stale_or_interrupted"] = bool(row["last_update_age_seconds"] is not None and row["last_update_age_seconds"] > 30 and row["status"] in ("running", "evaluating"))
        else:
            row = {"run_id": run.name, "status": "monitor_unavailable", "timesteps": None,
                   "target": None, "latest_checkpoint": None, "monitor_read_state": "unavailable",
                   "monitor_last_success_epoch": None}
        row["monitor_issue"] = issue
        return row

    def snapshot(self):
        """Return rows and explicit reader issues; never mark a run as failed."""
        now = time.time()
        rows, issues = [], []
        try:
            runs = sorted(path for path in self.output_root.iterdir() if path.is_dir())
        except OSError as error:
            issue = MonitorReadError(self.output_root, error, 1,
                "temporarily_unavailable" if isinstance(error, (PermissionError, FileNotFoundError)) else "read_error").issue()
            issues.append(issue)
            return [self._fallback(self.output_root / name, issue, now) for name in sorted(self.last_known)], issues
        for run in runs:
            try:
                status_only_issue = None
                status = self._read(run / "status.json")
                try:
                    progress = self._read(run / "progress.json")
                except MonitorReadError as error:
                    # The status file is created immediately before the first
                    # progress file. Status-only first observation is legitimate.
                    if isinstance(error.cause, FileNotFoundError) and run.name not in self.last_known:
                        progress = {}
                        status_only_issue = error.issue(run.name)
                        status_only_issue["category"] = "progress_not_yet_available"
                    else:
                        raise
                try:
                    row = _merge(run, status, progress, now)
                    if status_only_issue:
                        issues.append(status_only_issue)
                        row["monitor_read_state"] = "status_only"
                        row["monitor_issue"] = status_only_issue
                except ValueError as error:
                    raise MonitorReadError(run / "status.json", error, 1, "invalid_data") from error
            except MonitorReadError as error:
                issue = error.issue(run.name)
                issues.append(issue)
                rows.append(self._fallback(run, issue, now))
                continue
            self.last_known[run.name] = copy.deepcopy(row)
            rows.append(row)
        return rows, issues


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("status",), nargs="?", default="status")
    parser.add_argument("--output-root", default="experiments/alternative_primary_objectives/runs")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=2.)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--read-attempts", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=.02)
    args = parser.parse_args(argv)
    if not math.isfinite(args.interval) or args.interval < .2:
        raise ValueError("Monitor interval must be finite and >=0.2 seconds")
    if args.iterations is not None and args.iterations <= 0:
        raise ValueError("Monitor iterations must be positive")
    if not 1 <= args.read_attempts <= 10 or not math.isfinite(args.retry_delay) or not 0 <= args.retry_delay <= 1:
        raise ValueError("Read attempts must be 1--10 and retry delay finite in 0--1 seconds")
    root = Path(args.output_root)
    if not root.is_absolute(): root = REPO_ROOT / root
    monitor = ReadOnlyTrainingMonitor(root, attempts=args.read_attempts, retry_delay=args.retry_delay)
    count, last_exit_code, previous_issues = 0, 0, set()
    try:
        while True:
            rows, issues = monitor.snapshot()
            if args.json: print(json.dumps(rows, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
            else: display_status(rows)
            current_issues = set()
            for issue in issues:
                identity = json.dumps(issue, sort_keys=True)
                current_issues.add(identity)
                if identity not in previous_issues:
                    print(f"Monitor {issue['severity']}: {issue['path']} | {issue['error_type']} after {issue['attempts']} read attempt(s): {issue['message']} | showing last known data where available; training status is unchanged.", file=sys.stderr, flush=True)
            if previous_issues and not current_issues:
                print("Monitor sources are readable again.", file=sys.stderr, flush=True)
            previous_issues = current_issues
            last_exit_code = 2 if any(issue["severity"] == "error" for issue in issues) else 0
            count += 1
            if not args.watch or args.iterations is not None and count >= args.iterations: break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    return last_exit_code


if __name__ == "__main__":
    raise SystemExit(main())
