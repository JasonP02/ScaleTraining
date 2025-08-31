#!/usr/bin/env python3
"""
Summarize Hydra multirun or local W&B runs.

Primary behavior:
  - Scan a Hydra multirun directory for per-job result.json files written by
    scaletraining.entrypoints.train and print a compact comparison.

Fallback behavior (when result.json files are not present):
  - Use local W&B run folders (wandb/run-*/files) to collect train_per_token_loss
    from wandb-summary.json and configuration values (primary_optimizer,
    rope_implementation) from config.yaml. If a multirun directory is provided,
    the last N W&B runs are used, where N equals the number of jobs in that
    multirun; otherwise it uses the last 1 run.

Usage:
  - From repo root, after running a sweep with -m:
      python scripts/summarize_multirun.py multirun/<date>/<time>
  - If no path is provided, the script will pick the most recent multirun.
"""
from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class JobResult:
    job_dir: Path
    final_train_loss: Optional[float]
    primary_optimizer: Optional[str]
    rope_implementation: Optional[str]


def find_latest_multirun(root: Path) -> Optional[Path]:
    multirun = root / "multirun"
    if not multirun.is_dir():
        return None
    def is_date_dir(p: Path) -> bool:
        name = p.name
        return (
            len(name) == 10
            and name[4] == "-"
            and name[7] == "-"
            and name[:4].isdigit()
            and name[5:7].isdigit()
            and name[8:10].isdigit()
        )
    dated = sorted((p for p in multirun.iterdir() if p.is_dir() and is_date_dir(p)), reverse=True)
    for d in dated:
        times = sorted((p for p in d.iterdir() if p.is_dir()), reverse=True)
        if times:
            return times[0]
    return None


def load_job_result(job_dir: Path) -> Optional[JobResult]:
    f = job_dir / "result.json"
    if not f.exists():
        return None
    try:
        data = json.loads(f.read_text())
        return JobResult(
            job_dir=job_dir,
            final_train_loss=data.get("final_train_loss"),
            primary_optimizer=data.get("primary_optimizer"),
            rope_implementation=data.get("rope_implementation"),
        )
    except Exception:
        return None


def summarize(base: Path) -> List[JobResult]:
    jobs: List[JobResult] = []
    for job_dir in sorted((p for p in base.iterdir() if p.is_dir()), key=lambda p: p.name):
        r = load_job_result(job_dir)
        if r is not None:
            jobs.append(r)
    return jobs


def print_table(results: List[JobResult]) -> None:
    if not results:
        print("No results found.")
        return
    results = sorted(
        results,
        key=lambda r: (float("inf") if r.final_train_loss is None else r.final_train_loss),
    )
    print("job\toptimizer\trope\tfinal_loss")
    for r in results:
        print(
            f"{r.job_dir.name}\t{r.primary_optimizer or ''}\t{r.rope_implementation or ''}\t"
            f"{'' if r.final_train_loss is None else f'{r.final_train_loss:.4f}'}"
        )


def write_csv(results: List[JobResult], path: Path) -> None:
    try:
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["job", "primary_optimizer", "rope_implementation", "final_train_loss"])
            for r in results:
                w.writerow([r.job_dir.name, r.primary_optimizer, r.rope_implementation, r.final_train_loss])
        print(f"Wrote {path}")
    except Exception as e:
        print(f"Could not write CSV: {e}")


def parse_yaml_value_after_key(lines: List[str], key: str) -> Optional[str]:
    """Extract a simple "value: ..." following a top-level "<key>:" without PyYAML.

    Returns the stripped value string or None if not found.
    """
    try:
        for i, line in enumerate(lines):
            if line.strip() == f"{key}:":
                for j in range(i + 1, min(i + 8, len(lines))):
                    s = lines[j].strip()
                    if s.startswith("value:"):
                        return s.split(":", 1)[1].strip()
                break
    except Exception:
        pass
    return None


def load_wandb_runs(root: Path) -> List[Tuple[Path, Optional[float], Optional[str], Optional[str], Optional[datetime]]]:
    """Collect local W&B runs from wandb/run-*/files.

    Returns: list of (run_path, loss, optimizer, rope, started_at).
    """
    wandb_root = root / "wandb"
    runs: List[Tuple[Path, Optional[float], Optional[str], Optional[str], Optional[datetime]]] = []
    if not wandb_root.is_dir():
        return runs
    for run_dir in (p for p in wandb_root.iterdir() if p.is_dir() and p.name.startswith("run-")):
        files = run_dir / "files"
        if not files.is_dir():
            continue
        summary_path = files / "wandb-summary.json"
        cfg_path = files / "config.yaml"
        meta_path = files / "wandb-metadata.json"
        if not summary_path.exists():
            continue
        loss: Optional[float] = None
        optimizer: Optional[str] = None
        rope: Optional[str] = None
        started_at: Optional[datetime] = None
        try:
            sdata = json.loads(summary_path.read_text())
            if isinstance(sdata.get("train_per_token_loss"), (int, float)):
                loss = float(sdata["train_per_token_loss"])
        except Exception:
            pass
        try:
            if cfg_path.exists():
                lines = cfg_path.read_text(encoding="utf-8").splitlines()
                optimizer = parse_yaml_value_after_key(lines, "primary_optimizer") or optimizer
                rope = parse_yaml_value_after_key(lines, "rope_implementation") or rope
        except Exception:
            pass
        try:
            if meta_path.exists():
                m = json.loads(meta_path.read_text())
                ts = m.get("startedAt")
                if isinstance(ts, str):
                    # Example: 2025-08-31T20:32:19.696892Z
                    started_at = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%fZ")
        except Exception:
            pass
        runs.append((run_dir, loss, optimizer, rope, started_at))
    runs.sort(key=lambda t: (datetime.min if t[4] is None else t[4]))
    return runs


def summarize_from_wandb(root: Path, jobs_expected: int) -> List[JobResult]:
    """Fallback summarization using local W&B run folders.

    Takes the most recent ``jobs_expected`` runs and maps them into JobResult entries.
    """
    wb = load_wandb_runs(root)
    if not wb:
        return []
    tail = wb[-jobs_expected:] if jobs_expected > 0 else wb[-1:]
    results: List[JobResult] = []
    for run_dir, loss, optimizer, rope, _ in tail:
        results.append(JobResult(job_dir=run_dir, final_train_loss=loss, primary_optimizer=optimizer, rope_implementation=rope))
    return results


def main(argv: List[str]) -> int:
    root = Path(__file__).resolve().parents[1]
    if len(argv) > 1:
        base = Path(argv[1])
    else:
        latest = find_latest_multirun(root)
        if latest is None:
            # No multirun path; attempt to summarize from latest W&B run as a last resort
            wb_results = summarize_from_wandb(root, jobs_expected=1)
            if not wb_results:
                print("No multirun found and no W&B runs detected.")
                return 1
            print_table(wb_results)
            return 0
        base = latest
    if not base.is_dir():
        print(f"Not a directory: {base}")
        return 1
    results = summarize(base)
    if not results:
        try:
            jobs_expected = sum(1 for p in base.iterdir() if p.is_dir())
        except Exception:
            jobs_expected = 0
        results = summarize_from_wandb(root, jobs_expected)
    print_table(results)
    if results:
        csv_target = base / "summary.csv" if base.exists() else (root / "summary.csv")
        write_csv(results, csv_target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
