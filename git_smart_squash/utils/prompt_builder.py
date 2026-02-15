"""Prompt building utilities for AI analysis."""

from typing import Any, Dict, List, Optional
from ..diff_parser import Hunk


def build_hunk_prompt(
    hunks: List[Hunk],
    custom_instructions: Optional[str] = None,
    compare_info: Optional[Dict[str, Any]] = None,
) -> str:
    parts = [
        "Analyze these code changes and organize them into logical commits for pull request review.",
        "",
        "Each change is represented as a 'hunk' with a unique ID. Group related hunks together",
        "based on functionality, not just file location. A single commit can contain hunks from",
        "multiple files if they implement the same feature or fix.",
        "",
    ]

    if custom_instructions:
        parts.extend(["CUSTOM INSTRUCTIONS FROM USER:", custom_instructions, ""])

    if compare_info:
        commit_list = compare_info.get("commits")
        if not isinstance(commit_list, list):
            commit_list = []
        parts.extend(
            [
                "BRANCH CONTEXT:",
                f"- Base ref: {compare_info.get('base_ref')}",
                f"- Head ref: {compare_info.get('head_ref')}",
                f"- Ahead/behind: {compare_info.get('ahead', 0)} ahead, {compare_info.get('behind', 0)} behind",
                "- Recent commits on head (most recent first):",
            ]
        )
        for commit in commit_list:
            parts.append(
                f"  * {commit.get('hash', '')[:7]} {commit.get('subject', '')}"
            )
        parts.append("")

    parts.extend(
        [
            "For each commit, provide:",
            "1. A properly formatted git commit message following these rules:",
            "   - First line: max 80 characters (type: brief description)",
            "   - If more detail needed: empty line, then body with lines max 80 chars",
            "   - Use conventional commit format: feat:, fix:, docs:, test:, refactor:, etc.",
            "2. The specific hunk IDs that should be included (not file paths!)",
            "3. A brief rationale for why these changes belong together",
            "",
            "REQUIREMENTS:",
            "- EVERY hunk must appear in EXACTLY one commit",
            "- Use ONLY the hunk IDs listed below; do not invent IDs",
            "- Avoid single-hunk commits unless the change is isolated or risky",
            "- Keep formatting-only changes separate unless they are required for the feature",
            "- Prefer grouping tests with the feature they validate, unless large",
            "",
            "Return your response in this exact structure:",
            "{",
            '  "commits": [',
            "    {",
            '      "message": "feat: add user authentication system\\n\\nImplemented JWT-based authentication with refresh tokens.\\nAdded user model with secure password hashing.",',
            '      "hunk_ids": ["auth.py:45-89", "models.py:23-45", "auth.py:120-145"],',
            '      "rationale": "Groups authentication functionality together"',
            "    }",
            "  ]",
            "}",
            "",
            "IMPORTANT:",
            "- Use hunk_ids (not files) and group by logical functionality",
            "- First line of commit message must be under 80 characters",
            "- Provide rationale for grouping",
            "- If any hunks feel ungroupable, include them in a final catch-all commit",
            "",
            "VALID HUNK IDS:",
            ", ".join([hunk.id for hunk in hunks]),
            "",
            "CODE CHANGES TO ANALYZE:",
        ]
    )

    for hunk in hunks:
        parts.extend(
            [
                f"Hunk ID: {hunk.id}",
                f"File: {hunk.file_path}",
                f"Context lines: {hunk.start_line}-{hunk.end_line}",
                "",
                "Context:",
                hunk.context
                if hunk.context
                else f"(Context unavailable for {hunk.file_path})",
                "",
                "Changes:",
                hunk.content,
                "",
                "---",
                "",
            ]
        )

    return "\n".join(parts)
