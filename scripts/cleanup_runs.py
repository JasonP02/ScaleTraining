#!/usr/bin/env python3
"""Utility to prune old training runs and artifacts."""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple


PROTECTED_SUBSTRINGS = ("token", "data")
DEFAULT_TARGETS = ("outputs", "multirun", "wandb")


@dataclass
class Candidate:
    path: Path
    mtime: float


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dirs",
        nargs="+",
        default=list(DEFAULT_TARGETS),
        help="Target directories to prune (relative or absolute).",
    )
    parser.add_argument(
        "--keep-latest",
        type=int,
        default=0,
        help="Number of most-recent entries to preserve per directory.",
    )
    parser.add_argument(
        "--older-than",
        type=float,
        default=None,
        metavar="DAYS",
        help="Only delete entries older than this many days.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Perform deletions. Without this flag the script only prints actions.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print debugging details about candidate selection.",
    )
    return parser.parse_args(list(argv))


def is_protected(path: Path) -> bool:
    lower = str(path).lower()
    return any(substr in lower for substr in PROTECTED_SUBSTRINGS)


def collect_candidates(root: Path, verbose: bool = False) -> List[Candidate]:
    if not root.exists():
        if verbose:
            print(f"[skip] {root} (missing)")
        return []
    if not root.is_dir():
        if verbose:
            print(f"[skip] {root} (not a directory)")
        return []
    if is_protected(root):
        if verbose:
            print(f"[protect] {root} contains protected keyword; skipping")
        return []

    entries: List[Candidate] = []
    with os.scandir(root) as scan:
        for entry in scan:
            try:
                entry_path = Path(entry.path)
                if entry_path.name.startswith('.'):  # hidden entries
                    continue
                if entry.is_symlink():
                    continue
                if not entry.is_dir():
                    continue
                if is_protected(entry_path):
                    if verbose:
                        print(f"[protect] {entry_path} contains protected keyword; skipping")
                    continue
                stat_info = entry.stat(follow_symlinks=False)
            except Exception as exc:
                if verbose:
                    print(f"[warn] failed to stat {entry.path}: {exc}")
                continue
            entries.append(Candidate(entry_path, stat_info.st_mtime))
    return entries


def select_removals(candidates: List[Candidate], keep_latest: int, cutoff_ts: float | None) -> List[Candidate]:
    if not candidates:
        return []
    keep_n = max(keep_latest, 0)
    ordered = sorted(candidates, key=lambda c: c.mtime, reverse=True)
    removals: List[Candidate] = []
    for idx, candidate in enumerate(ordered):
        if idx < keep_n:
            continue
        if cutoff_ts is not None and candidate.mtime >= cutoff_ts:
            continue
        removals.append(candidate)
    return removals


def delete_path(path: Path, execute: bool) -> None:
    action = "Deleting" if execute else "Would delete"
    print(f"{action}: {path}")
    if not execute:
        return
    try:
        shutil.rmtree(path)
    except Exception as exc:
        print(f"[error] failed to delete {path}: {exc}")


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    cutoff_ts = None
    if args.older_than is not None:
        cutoff_ts = time.time() - float(args.older_than) * 86400.0

    for target in args.dirs:
        root = Path(target).expanduser().resolve()
        if args.verbose:
            print(f"[scan] {root}")
        candidates = collect_candidates(root, verbose=args.verbose)
        removals = select_removals(candidates, args.keep_latest, cutoff_ts)
        if args.verbose:
            kept = len(candidates) - len(removals)
            print(f"[kept] {kept} entries in {root}")
        for candidate in removals:
            delete_path(candidate.path, execute=args.execute)

    if not args.execute:
        print("(dry run: no deletions performed; pass --execute to apply)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
