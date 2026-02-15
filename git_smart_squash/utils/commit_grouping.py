"""Commit-history grouping and hunk assignment prompts."""

from typing import Dict, List

from ..diff_parser import Hunk


COMMIT_GROUP_SCHEMA = {
    "type": "object",
    "properties": {
        "groups": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "commit_hashes": {"type": "array", "items": {"type": "string"}},
                    "suggested_message": {"type": "string"},
                },
                "required": [
                    "name",
                    "description",
                    "commit_hashes",
                    "suggested_message",
                ],
                "additionalProperties": False,
            },
        },
        "total_commits": {"type": "integer"},
    },
    "required": ["groups", "total_commits"],
    "additionalProperties": False,
}


HUNK_ASSIGNMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "assignments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "group_name": {"type": "string"},
                    "hunk_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["group_name", "hunk_ids"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["assignments"],
    "additionalProperties": False,
}


def build_commit_grouping_prompt(commit_history: List[Dict[str, object]]) -> str:
    commits_serialized = []
    for commit in commit_history:
        commits_serialized.append(
            {
                "hash": commit.get("hash"),
                "subject": commit.get("subject"),
                "body": commit.get("body"),
                "files": commit.get("files"),
                "shortstat": commit.get("shortstat"),
                "diffstat": commit.get("diffstat"),
            }
        )

    hashes = [str(c.get("hash")) for c in commit_history if c.get("hash")]

    return "\n".join(
        [
            "You are a Git commit analyzer.",
            f"Group {len(commit_history)} commits into logical feature/fix groups.",
            "",
            "RULES:",
            "- Every commit hash must appear in exactly one group",
            "- Use only the valid hashes listed below",
            "- Keep commit order within each group (oldest to newest)",
            "- Prefer feat/fix with clear scopes and PR-ready messages",
            "",
            "VALID COMMIT HASHES:",
            ", ".join(hashes),
            "",
            "COMMIT DATA (oldest to newest):",
            str(commits_serialized),
            "",
            "Output JSON only, matching this schema:",
            "{",
            '  "groups": [',
            "    {",
            '      "name": "Feature Name",',
            '      "description": "Brief description",',
            '      "commit_hashes": ["hash1", "hash2"],',
            '      "suggested_message": "feat(scope): summary"',
            "    }",
            "  ],",
            f'  "total_commits": {len(commit_history)}',
            "}",
        ]
    )


def build_hunk_assignment_prompt(
    groups: List[Dict[str, object]],
    hunks: List[Hunk],
) -> str:
    group_lines: List[str] = []
    for group in groups:
        group_lines.append(f"Group: {group.get('name')}")
        group_lines.append(f"Message: {group.get('suggested_message')}")
        group_lines.append(f"Description: {group.get('description')}")
        files = group.get("files")
        if isinstance(files, list) and files:
            group_lines.append(f"Files: {', '.join(files)}")
        subjects = group.get("subjects")
        if isinstance(subjects, list) and subjects:
            group_lines.append("Commit subjects:")
            for subject in subjects:
                group_lines.append(f"- {subject}")
        group_lines.append("")

    hunk_lines: List[str] = []
    for hunk in hunks:
        hunk_lines.extend(
            [
                f"Hunk ID: {hunk.id}",
                f"File: {hunk.file_path}",
                f"Change type: {hunk.change_type}",
                f"Depends on: {', '.join(sorted(hunk.dependencies)) if hunk.dependencies else 'None'}",
                "Context:",
                hunk.context
                if hunk.context
                else f"(Context unavailable for {hunk.file_path})",
                "Changes:",
                hunk.content,
                "---",
                "",
            ]
        )

    return "\n".join(
        [
            "Assign each hunk to exactly one of the groups below.",
            "Use only the provided hunk IDs and group names.",
            "Prefer grouping by feature/fix intent and module scope.",
            "",
            "GROUPS:",
            *group_lines,
            "HUNKS:",
            *hunk_lines,
            "",
            "Return JSON only:",
            "{",
            '  "assignments": [',
            "    {",
            '      "group_name": "Group Name",',
            '      "hunk_ids": ["file.ts:10-20", "file.ts:33-40"]',
            "    }",
            "  ]",
            "}",
        ]
    )
