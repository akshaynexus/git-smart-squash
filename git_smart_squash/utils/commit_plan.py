"""Commit plan validation and normalization."""

from typing import Dict, List, Tuple

from ..diff_parser import Hunk


def normalize_commit_plan(
    commits: List[dict],
    hunks: List[Hunk],
    add_catch_all: bool = True,
) -> Tuple[List[dict], Dict[str, object]]:
    """Normalize commit plan by removing invalid/duplicate hunks and adding missing ones."""
    valid_ids = [h.id for h in hunks]
    valid_set = set(valid_ids)

    seen = set()
    invalid: List[str] = []
    duplicates: List[str] = []
    normalized: List[dict] = []

    for commit in commits:
        hunk_ids = commit.get("hunk_ids") or []
        unique: List[str] = []
        for hunk_id in hunk_ids:
            if hunk_id not in valid_set:
                invalid.append(hunk_id)
                continue
            if hunk_id in seen:
                duplicates.append(hunk_id)
                continue
            seen.add(hunk_id)
            unique.append(hunk_id)

        if unique:
            new_commit = dict(commit)
            new_commit["hunk_ids"] = unique
            normalized.append(new_commit)

    missing = [hunk_id for hunk_id in valid_ids if hunk_id not in seen]

    if add_catch_all and missing:
        normalized.append(
            {
                "message": "chore: remaining changes",
                "hunk_ids": missing,
                "rationale": "Catch-all for hunks not assigned by the model.",
            }
        )

    stats = {
        "total_hunks": len(valid_ids),
        "assigned_hunks": len(seen),
        "invalid_hunks": invalid,
        "duplicate_hunks": duplicates,
        "missing_hunks": missing,
    }

    return normalized, stats
