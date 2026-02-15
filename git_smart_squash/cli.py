"""Simplified command-line interface for Git Smart Squash."""

import argparse
import sys
import subprocess
import json
import os
from typing import List, Dict, Any, Optional, Set
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from .simple_config import ConfigManager
from .ai.providers.simple_unified import UnifiedAIProvider
from .diff_parser import parse_diff, Hunk
from .utils.git_diff import get_full_diff as util_get_full_diff
from .utils.git_compare import get_branch_compare
from .utils.git_history import get_commit_history
from .utils.commit_grouping import (
    build_commit_grouping_prompt,
    build_hunk_assignment_prompt,
    COMMIT_GROUP_SCHEMA,
    HUNK_ASSIGNMENT_SCHEMA,
)
from .hunk_applicator import apply_hunks_with_fallback, reset_staging_area
from .logger import get_logger, LogLevel
from .dependency_validator import DependencyValidator, ValidationResult
from .strategies.backup_manager import BackupManager
from .utils.commit_plan import normalize_commit_plan


class GitSmartSquashCLI:
    """Simplified CLI for git smart squash."""

    def __init__(self):
        self.console = Console()
        self.config_manager = ConfigManager()
        self.config = None
        self.logger = get_logger()
        self.logger.set_console(self.console)

    def main(self):
        """Main entry point for the CLI."""
        parser = self.create_parser()
        args = parser.parse_args()

        # Set debug logging if requested
        if args.debug:
            self.logger.set_level(LogLevel.DEBUG)
            self.logger.debug("Debug logging enabled")

        try:
            # Load configuration
            self.config = self.config_manager.load_config(args.config)

            # Override config with command line arguments
            if args.ai_provider:
                self.config.ai.provider = args.ai_provider
                # If provider is changed but no model specified, use provider default
                if not args.model:
                    self.config.ai.model = self.config_manager._get_default_model(
                        args.ai_provider
                    )
            if args.model:
                self.config.ai.model = args.model
            if args.reasoning is not None:
                self.config.ai.reasoning = args.reasoning or self.config.ai.reasoning

            # Gentle pre-check: OpenAI models must be GPT-5 family
            if self.config.ai.provider.lower() == "openai" and not str(
                self.config.ai.model
            ).startswith("gpt-5"):
                self.console.print(
                    "[yellow]OpenAI provider now uses GPT-5 models only.[/yellow]"
                )
                self.console.print(
                    "Use --model gpt-5 (or gpt-5-mini/gpt-5-nano), or choose another provider via --ai-provider."
                )
                sys.exit(1)
            if getattr(args, "max_predict_tokens", None) is not None:
                self.config.ai.max_predict_tokens = (
                    args.max_predict_tokens or self.config.ai.max_predict_tokens
                )

            # Use base branch from config if not provided via CLI
            if args.base is None:
                args.base = self.config.base

            # Run the simplified smart squash
            self.run_smart_squash(args)

        except Exception as e:
            self.console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)

    def create_parser(self) -> argparse.ArgumentParser:
        """Create the simplified argument parser."""
        parser = argparse.ArgumentParser(
            prog="git-smart-squash",
            description="AI-powered git commit reorganization for clean PR reviews",
        )

        parser.add_argument(
            "--base",
            default="main",
            help="Base branch to compare against (default: from config or main)",
        )

        parser.add_argument(
            "--ai-provider",
            choices=["openai", "openrouter", "anthropic", "local", "gemini"],
            help="AI provider to use",
        )

        parser.add_argument("--model", help="AI model to use")

        parser.add_argument("--config", help="Path to configuration file")

        parser.add_argument(
            "--auto-apply",
            action="store_true",
            help="Apply the commit plan immediately without confirmation",
        )

        parser.add_argument(
            "--instructions",
            "-i",
            type=str,
            help='Custom instructions for AI to follow when organizing commits (e.g., "Group by feature area", "Separate tests from implementation")',
        )

        parser.add_argument(
            "--no-attribution",
            action="store_true",
            help="Disable the attribution message in commit messages",
        )

        parser.add_argument(
            "--debug",
            action="store_true",
            help="Enable debug logging for detailed hunk application information",
        )

        parser.add_argument(
            "--reasoning",
            choices=["high", "medium", "low", "minimal"],
            default=None,
            help="Reasoning effort level for GPT-5 models (default: low)",
        )

        parser.add_argument(
            "--max-predict-tokens",
            type=int,
            default=None,
            help="Maximum tokens for completion/output (default: 200000)",
        )

        return parser

    def run_smart_squash(self, args):
        """Run the simplified smart squash operation."""
        try:
            # Ensure config is loaded
            if self.config is None:
                self.config = self.config_manager.load_config()

            # 1. Get the full diff between base branch and current branch
            full_diff = self.get_full_diff(args.base)
            if not full_diff:
                self.console.print("[yellow]No changes found to reorganize[/yellow]")
                return

            compare_info = None
            if self.config.ai.include_branch_context:
                try:
                    compare_info = get_branch_compare(
                        args.base, max_commits=self.config.ai.max_commit_context
                    )
                    self._display_branch_compare(compare_info)
                except Exception as e:
                    self.logger.debug(f"Branch comparison failed: {e}")

            commit_history = None
            if self.config.ai.include_commit_history:
                try:
                    commit_history = get_commit_history(args.base)
                    max_history = self.config.ai.max_commit_history
                    if isinstance(max_history, int) and max_history > 0:
                        commit_history = commit_history[-max_history:]
                    if commit_history:
                        commit_history = list(reversed(commit_history))
                    self._display_commit_history_summary(commit_history)
                except Exception as e:
                    self.logger.debug(f"Commit history fetch failed: {e}")

            # 1.5. Working directory pre-check: show guidance if dirty, but continue to analysis.
            # Final safety check happens again before any apply.
            status_info = self._check_working_directory_clean()
            if not status_info["is_clean"]:
                self._display_working_directory_help(status_info)

            # 2. Parse diff into individual hunks
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
            ) as progress:
                task = progress.add_task("Parsing changes into hunks...", total=None)
                hunks = parse_diff(
                    full_diff, context_lines=self.config.hunks.context_lines
                )

                if not hunks:
                    self.console.print("[yellow]No hunks found to reorganize[/yellow]")
                    return

                self.console.print(
                    f"[green]Found {len(hunks)} hunks to analyze[/green]"
                )

                # Check if we have too many hunks for the AI to process
                max_hunks = self.config.hunks.max_hunks_per_prompt
                if (
                    isinstance(max_hunks, int)
                    and max_hunks > 0
                    and len(hunks) > max_hunks
                ):
                    self.console.print(
                        f"[yellow]Warning: {len(hunks)} hunks found, limiting to {max_hunks} for AI analysis[/yellow]"
                    )
                    hunks = hunks[:max_hunks]

                # 3. Send hunks to AI for commit organization
                progress.update(task, description="Analyzing changes with AI...")
                # Use custom instructions from CLI args, or fall back to config
                custom_instructions = args.instructions or self.config.ai.instructions
                if self.config.ai.multi_stage_enabled and commit_history:
                    commit_plan = self._run_multi_stage_grouping(
                        hunks,
                        commit_history,
                        compare_info,
                    )
                else:
                    commit_plan = self.analyze_with_ai(
                        hunks,
                        full_diff,
                        custom_instructions,
                        compare_info,
                        commit_history,
                    )

            if not commit_plan:
                self.console.print("[red]Failed to generate commit plan[/red]")
                return

            commits_list = (
                commit_plan
                if isinstance(commit_plan, list)
                else commit_plan.get("commits", [])
            )
            normalized_commits, stats = normalize_commit_plan(
                commits_list, hunks, add_catch_all=False
            )

            if self._should_run_second_pass(stats):
                self.console.print(
                    "[yellow]Large unassigned set detected. Running second-pass grouping...[/yellow]"
                )
                missing_hunk_ids = stats.get("missing_hunks", [])
                missing_hunks = [h for h in hunks if h.id in set(missing_hunk_ids)]
                second_pass = self.analyze_with_ai(
                    missing_hunks,
                    full_diff,
                    self._build_second_pass_instructions(custom_instructions),
                    compare_info,
                    commit_history,
                )
                second_commits = (
                    second_pass
                    if isinstance(second_pass, list)
                    else (second_pass or {}).get("commits", [])
                )
                commits_list = normalized_commits + (second_commits or [])
            else:
                commits_list = normalized_commits

            normalized_commits, stats = normalize_commit_plan(commits_list, hunks)
            commit_plan = normalized_commits

            self._display_plan_quality(stats)

            # Validate the commit plan respects hunk dependencies
            validator = DependencyValidator()
            # Normalize to list for validation
            commits_list = (
                commit_plan
                if isinstance(commit_plan, list)
                else commit_plan.get("commits", [])
            )
            validation_result = validator.validate_commit_plan(commits_list, hunks)

            if not validation_result.is_valid:
                # Show dependency relationships to the user as informational
                self.console.print(
                    "\n[yellow]Dependency relationships detected between hunks (informational):[/yellow]"
                )
                for error in validation_result.errors:
                    self.console.print(f"  • {error}")
                self.console.print(
                    "[dim]Dependencies are informational; proceeding with the original plan.[/dim]"
                )
                # Also log at debug level
                self.logger.debug(
                    "Dependency relationships detected between hunks (informational):"
                )
                for error in validation_result.errors:
                    self.logger.debug(f"  • {error}")
                # Continue with the original plan - no need to reorganize

            # Log any warnings even if validation passed
            if validation_result.warnings:
                self.console.print("\n[yellow]Warnings:[/yellow]")
                for warning in validation_result.warnings:
                    self.console.print(f"  • {warning}")

            # 3. Display the plan
            self.display_commit_plan(commit_plan)

            # 5. Ask for confirmation (unless auto-applying)
            # Auto-apply if --auto-apply flag is provided or if config says to auto-apply
            auto_apply_from_config = getattr(self.config, "auto_apply", False)
            if args.auto_apply or auto_apply_from_config:
                if args.auto_apply:
                    self.console.print(
                        "\n[green]Applying commit plan (--auto-apply flag provided)[/green]"
                    )
                elif auto_apply_from_config:
                    self.console.print(
                        "\n[green]Auto-applying commit plan (configured in settings)[/green]"
                    )
                # Final check right before apply
                self.console.print(
                    "[dim]Final working directory check before applying changes...[/dim]"
                )
                final_status_info = self._check_working_directory_clean()
                if not final_status_info["is_clean"]:
                    self.console.print(
                        "[red]❌ Working directory changed during operation![/red]"
                    )
                    self._display_working_directory_help(final_status_info)
                    return
                self.apply_commit_plan(
                    commit_plan, hunks, full_diff, args.base, args.no_attribution
                )
            elif self.get_user_confirmation():
                # Final check after user confirms
                self.console.print(
                    "[dim]Final working directory check before applying changes...[/dim]"
                )
                final_status_info = self._check_working_directory_clean()
                if not final_status_info["is_clean"]:
                    self.console.print(
                        "[red]❌ Working directory changed during operation![/red]"
                    )
                    self._display_working_directory_help(final_status_info)
                    return
                self.apply_commit_plan(
                    commit_plan, hunks, full_diff, args.base, args.no_attribution
                )
            else:
                self.console.print("Operation cancelled.")

        except Exception as e:
            self.console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)

    def get_full_diff(self, base_branch: str) -> Optional[str]:
        """Get the full diff between base branch and current branch."""
        return util_get_full_diff(base_branch, console=self.console)

    def analyze_with_ai(
        self,
        hunks: List[Hunk],
        full_diff: str,
        custom_instructions: Optional[str] = None,
        compare_info: Optional[Dict[str, Any]] = None,
        commit_history: Optional[List[Dict[str, Any]]] = None,
    ):
        """Send hunks to AI and get back commit organization plan (as a list)."""
        try:
            # Ensure config is loaded
            if self.config is None:
                self.config = self.config_manager.load_config()

            ai_provider = UnifiedAIProvider(self.config)

            # Build hunk-based prompt
            prompt = self._build_hunk_prompt(
                hunks, custom_instructions, compare_info, commit_history
            )

            response = ai_provider.generate(prompt)

            # With structured output, response should be valid JSON
            result = json.loads(response)

            self.logger.debug(f"AI response type: {type(result).__name__}")
            self.logger.debug(
                f"AI response: {json.dumps(result, indent=2) if isinstance(result, (dict, list)) else str(result)}"
            )

            # Normalize to list of commits
            if (
                isinstance(result, dict)
                and "commits" in result
                and isinstance(result["commits"], list)
            ):
                return result["commits"]
            if isinstance(result, list):
                return result
            self.console.print(
                f"[red]AI returned invalid response format: expected list or object with 'commits'[/red]"
            )
            return None

        except json.JSONDecodeError as e:
            self.console.print(f"[red]AI returned invalid JSON: {e}[/red]")
            return None
        except Exception as e:
            self.console.print(f"[red]AI analysis failed: {e}[/red]")
            return None

    def _build_hunk_prompt(
        self,
        hunks: List[Hunk],
        custom_instructions: Optional[str] = None,
        compare_info: Optional[Dict[str, Any]] = None,
        commit_history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        from .utils.prompt_builder import build_hunk_prompt

        return build_hunk_prompt(
            hunks, custom_instructions, compare_info, commit_history
        )

    def display_commit_plan(self, commit_plan):
        from .utils.display import display_commit_plan as _display

        _display(self.console, commit_plan)

    def get_user_confirmation(self) -> bool:
        """Get user confirmation to proceed."""
        self.console.print("\n[bold]Apply this commit structure?[/bold]")
        try:
            response = input("Continue? (y/N): ")
            self.logger.debug(f"User input received: '{response}'")
            result = response.lower().strip() == "y"
            self.logger.debug(f"Confirmation result: {result}")
            return result
        except (EOFError, KeyboardInterrupt):
            self.logger.debug("Input interrupted or EOF received")
            return False

    def apply_commit_plan(
        self,
        commit_plan,
        hunks: List[Hunk],
        full_diff: str,
        base_branch: str,
        no_attribution: bool = False,
    ):
        from .strategies.commit_applier import apply_commit_plan as _apply

        _apply(self, commit_plan, hunks, full_diff, base_branch, no_attribution)

    def _check_working_directory_clean(self) -> Dict[str, Any]:
        from .utils.working_dir import check_clean

        return check_clean()

    def _display_working_directory_help(self, status_info: Dict[str, Any]):
        from .utils.working_dir import display_help

        display_help(self.console, status_info)

    def _display_branch_compare(self, compare_info: Dict[str, Any]) -> None:
        base_ref = compare_info.get("base_ref")
        head_ref = compare_info.get("head_ref")
        ahead = int(compare_info.get("ahead", 0))
        behind = int(compare_info.get("behind", 0))
        commit_limit = int(compare_info.get("commit_limit", 0))
        self.console.print(
            f"[dim]Comparing {head_ref} to {base_ref}: {ahead} ahead, {behind} behind[/dim]"
        )
        commits = compare_info.get("commits")
        if not isinstance(commits, list):
            commits = []
        if commits:
            self.console.print(f"[dim]Recent commits (max {commit_limit}):[/dim]")
            for commit in commits:
                subject = commit.get("subject", "")
                self.console.print(
                    f"[dim]  • {commit.get('hash', '')[:7]} {subject}[/dim]"
                )

    def _display_plan_quality(self, stats: Dict[str, Any]) -> None:
        total_hunks = int(stats.get("total_hunks", 0))
        assigned = int(stats.get("assigned_hunks", 0))
        invalid = stats.get("invalid_hunks")
        duplicates = stats.get("duplicate_hunks")
        missing = stats.get("missing_hunks")

        if not isinstance(invalid, list):
            invalid = []
        if not isinstance(duplicates, list):
            duplicates = []
        if not isinstance(missing, list):
            missing = []

        if total_hunks:
            pct = (assigned / total_hunks) * 100
            self.console.print(
                f"[dim]AI assigned {assigned}/{total_hunks} hunks ({pct:.1f}%) before normalization[/dim]"
            )

        if invalid:
            preview = ", ".join(invalid[:10])
            more = "..." if len(invalid) > 10 else ""
            self.console.print(
                f"[yellow]Removed invalid hunk IDs from plan: {preview}{more}[/yellow]"
            )

        if duplicates:
            preview = ", ".join(duplicates[:10])
            more = "..." if len(duplicates) > 10 else ""
            self.console.print(
                f"[yellow]Removed duplicate hunk IDs from plan: {preview}{more}[/yellow]"
            )

        if missing:
            self.console.print(
                f"[yellow]Added {len(missing)} unassigned hunks to a final catch-all commit[/yellow]"
            )

    def _should_run_second_pass(self, stats: Dict[str, Any]) -> bool:
        if not self.config.ai.second_pass_enabled:
            return False
        total = int(stats.get("total_hunks", 0))
        missing = stats.get("missing_hunks")
        if not isinstance(missing, list) or not missing:
            return False
        ratio = (len(missing) / total) if total else 0
        return (
            len(missing) >= self.config.ai.second_pass_min_hunks
            or ratio >= self.config.ai.second_pass_min_ratio
        )

    def _build_second_pass_instructions(self, base_instructions: Optional[str]) -> str:
        extra = (
            "Second pass: Only categorize the unassigned hunks below. "
            "Do NOT reuse hunk IDs already assigned. "
            "Prefer feat/fix with clear scope and PR-ready wording."
        )
        if base_instructions:
            return f"{base_instructions}\n\n{extra}"
        return extra

    def _run_multi_stage_grouping(
        self,
        hunks: List[Hunk],
        commit_history: List[Dict[str, Any]],
        compare_info: Optional[Dict[str, Any]],
    ):
        groups = self._group_commits_with_ai(commit_history)
        if not groups:
            return None

        validated_groups = self._validate_commit_groups(groups, commit_history)
        if not validated_groups:
            return None

        group_infos = self._build_group_infos(validated_groups, commit_history)
        assignments = self._assign_hunks_to_groups(hunks, group_infos, compare_info)

        commit_plan: List[Dict[str, Any]] = []
        for group in group_infos:
            hunk_ids = assignments.get(group["name"], [])
            if not hunk_ids:
                continue
            commit_plan.append(
                {
                    "message": group["suggested_message"],
                    "hunk_ids": hunk_ids,
                    "rationale": group["description"],
                }
            )

        return commit_plan

    def _group_commits_with_ai(self, commit_history: List[Dict[str, Any]]):
        prompt = build_commit_grouping_prompt(commit_history)
        ai_provider = UnifiedAIProvider(self.config)
        response = ai_provider.generate_with_schema(prompt, COMMIT_GROUP_SCHEMA)
        try:
            result = json.loads(response)
        except json.JSONDecodeError as e:
            self.console.print(
                f"[red]AI returned invalid JSON for commit grouping: {e}[/red]"
            )
            return None
        groups = result.get("groups") if isinstance(result, dict) else None
        if not isinstance(groups, list):
            self.console.print("[red]Commit grouping response missing groups[/red]")
            return None
        return groups

    def _validate_commit_groups(
        self, groups: List[Dict[str, Any]], commit_history: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        valid_hashes = [c.get("hash") for c in commit_history if c.get("hash")]
        valid_set = set(valid_hashes)
        seen = set()
        normalized: List[Dict[str, Any]] = []

        for group in groups:
            hashes = group.get("commit_hashes") or []
            unique: List[str] = []
            for commit_hash in hashes:
                if commit_hash not in valid_set:
                    continue
                if commit_hash in seen:
                    continue
                seen.add(commit_hash)
                unique.append(commit_hash)
            if unique:
                new_group = dict(group)
                new_group["commit_hashes"] = unique
                normalized.append(new_group)

        missing = [h for h in valid_hashes if h not in seen]
        if missing:
            normalized.append(
                {
                    "name": "Miscellaneous Changes",
                    "description": "Commits not grouped by AI",
                    "commit_hashes": missing,
                    "suggested_message": "chore: miscellaneous updates",
                }
            )

        return normalized

    def _build_group_infos(
        self, groups: List[Dict[str, Any]], commit_history: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        commit_map = {c.get("hash"): c for c in commit_history}
        enriched: List[Dict[str, Any]] = []
        for group in groups:
            files = set()
            subjects: List[str] = []
            for commit_hash in group.get("commit_hashes", []):
                commit = commit_map.get(commit_hash)
                if not commit:
                    continue
                subjects.append(str(commit.get("subject", "")))
                for file_path in commit.get("files", []) or []:
                    files.add(file_path)
            enriched.append(
                {
                    "name": group.get("name"),
                    "description": group.get("description"),
                    "suggested_message": group.get("suggested_message"),
                    "commit_hashes": group.get("commit_hashes"),
                    "files": sorted(files),
                    "subjects": subjects,
                }
            )
        return enriched

    def _assign_hunks_to_groups(
        self,
        hunks: List[Hunk],
        groups: List[Dict[str, Any]],
        compare_info: Optional[Dict[str, Any]],
    ) -> Dict[str, List[str]]:
        assignments: Dict[str, List[str]] = {g["name"]: [] for g in groups}
        group_files = {g["name"]: set(g.get("files", [])) for g in groups}

        unassigned: List[Hunk] = []
        for hunk in hunks:
            candidates = [
                name for name, files in group_files.items() if hunk.file_path in files
            ]
            if len(candidates) == 1:
                assignments[candidates[0]].append(hunk.id)
            else:
                unassigned.append(hunk)

        if unassigned and self.config.ai.assignment_ai_enabled:
            prompt = build_hunk_assignment_prompt(groups, unassigned)
            ai_provider = UnifiedAIProvider(self.config)
            response = ai_provider.generate_with_schema(prompt, HUNK_ASSIGNMENT_SCHEMA)
            try:
                result = json.loads(response)
            except json.JSONDecodeError as e:
                self.console.print(
                    f"[red]AI returned invalid JSON for hunk assignment: {e}[/red]"
                )
                result = None
            if isinstance(result, dict):
                for assignment in result.get("assignments", []):
                    group_name = assignment.get("group_name")
                    if group_name not in assignments:
                        continue
                    for hunk_id in assignment.get("hunk_ids", []) or []:
                        assignments[group_name].append(hunk_id)

        return assignments

    def _display_commit_history_summary(
        self, commit_history: Optional[List[Dict[str, Any]]]
    ) -> None:
        if not commit_history:
            return
        self.console.print(
            f"[dim]Included {len(commit_history)} commits of history in the AI prompt[/dim]"
        )

        # (removed duplicate implementation; commit application lives in strategies.commit_applier)


def main():
    """Entry point for the git-smart-squash command."""
    cli = GitSmartSquashCLI()
    cli.main()


if __name__ == "__main__":
    main()
