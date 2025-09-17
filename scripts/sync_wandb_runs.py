#!/usr/bin/env python3
"""Utility that syncs all local W&B runs to the cloud."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("wandb"),
        help="Directory containing local W&B runs (default: ./wandb).",
    )
    parser.add_argument(
        "--wandb-cmd",
        default="wandb",
        help="Executable used to invoke the W&B CLI (default: wandb).",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Pass --clean to wandb sync to remove local files after upload.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the constructed command without executing it.",
    )
    args_list: List[str] | None = list(argv) if argv is not None else None
    return parser.parse_args(args=args_list)


def build_command(args: argparse.Namespace, root: Path) -> list[str]:
    """Construct the wandb sync command to execute."""
    command = [args.wandb_cmd, "sync", "--sync-all"]
    if args.clean:
        command.append("--clean")
    command.append(str(root))
    return command


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.expanduser().resolve()

    if not root.exists():
        print(f"[warn] W&B directory not found: {root}")
        return 0

    command = build_command(args, root)
    print("[info] Executing:", " ".join(command))
    if args.dry_run:
        return 0

    try:
        completed = subprocess.run(command, check=False)
    except FileNotFoundError as exc:  # wandb CLI missing
        print(f"[error] Failed to execute {command[0]!r}: {exc}")
        return 1

    if completed.returncode != 0:
        print(f"[error] wandb sync exited with code {completed.returncode}")
        return completed.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
