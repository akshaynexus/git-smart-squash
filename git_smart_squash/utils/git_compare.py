"""Git branch comparison helpers."""

import subprocess
from typing import Dict, List, Optional


def _ref_exists(ref: str) -> bool:
    result = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", ref], capture_output=True
    )
    return result.returncode == 0


def resolve_base_ref(base_branch: str) -> str:
    """Resolve a usable base ref with common fallbacks."""
    for cand in [
        base_branch,
        f"origin/{base_branch}",
        "master",
        "origin/master",
        "develop",
        "origin/develop",
    ]:
        if _ref_exists(cand):
            return cand

    first = subprocess.run(
        ["git", "rev-list", "--max-parents=0", "HEAD"], capture_output=True, text=True
    )
    first_commit = (
        first.stdout.strip().splitlines()[0] if first.stdout.strip() else None
    )
    return first_commit or "HEAD"


def get_branch_compare(
    base_branch: str, head_ref: str = "HEAD", max_commits: int = 20
) -> Dict[str, object]:
    """Return ahead/behind counts and recent commits for head vs base."""
    base_ref = resolve_base_ref(base_branch)

    counts = subprocess.run(
        ["git", "rev-list", "--left-right", "--count", f"{base_ref}...{head_ref}"],
        capture_output=True,
        text=True,
        check=True,
    )
    left, right = counts.stdout.strip().split()

    commits: List[Dict[str, str]] = []
    log = subprocess.run(
        [
            "git",
            "log",
            "--format=%H%x09%s",
            f"{base_ref}..{head_ref}",
            f"-n{max_commits}",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    for line in log.stdout.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t", 1)
        if len(parts) == 2:
            commits.append({"hash": parts[0], "subject": parts[1]})

    return {
        "base_ref": base_ref,
        "head_ref": head_ref,
        "ahead": int(right),
        "behind": int(left),
        "commits": commits,
        "commit_limit": max_commits,
    }
