"""Git commit history helpers."""

import subprocess
from typing import List, Dict

from .git_compare import resolve_base_ref


def get_commit_history(
    base_branch: str, head_ref: str = "HEAD"
) -> List[Dict[str, object]]:
    """Return commit history between base and head with subject/body/files."""
    base_ref = resolve_base_ref(base_branch)
    log = subprocess.run(
        ["git", "log", f"{base_ref}..{head_ref}", "--format=%H"],
        capture_output=True,
        text=True,
        check=True,
    )

    hashes = [line.strip() for line in log.stdout.splitlines() if line.strip()]
    commits: List[Dict[str, object]] = []

    for commit_hash in hashes:
        subject = subprocess.run(
            ["git", "log", "-1", "--format=%s", commit_hash],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        body = subprocess.run(
            ["git", "log", "-1", "--format=%b", commit_hash],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        files_output = subprocess.run(
            ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", commit_hash],
            capture_output=True,
            text=True,
            check=True,
        ).stdout

        files = [line.strip() for line in files_output.splitlines() if line.strip()]

        commits.append(
            {
                "hash": commit_hash,
                "subject": subject,
                "body": body,
                "files": files,
            }
        )

    return commits
