"""Simplified configuration for Git Smart Squash."""

import os
import yaml
from typing import Optional
from dataclasses import dataclass


@dataclass
class AIConfig:
    """AI provider configuration."""

    provider: str = "local"
    model: str = "devstral"
    api_key_env: Optional[str] = None
    instructions: Optional[str] = None
    reasoning: str = "low"  # Reasoning effort level for GPT-5 models (default: low)
    max_predict_tokens: int = (
        64000  # Maximum tokens for completion/output (default: 64k)
    )
    include_branch_context: bool = True
    max_commit_context: int = 20
    include_commit_history: bool = True
    max_commit_history: Optional[int] = None
    second_pass_enabled: bool = True
    second_pass_min_hunks: int = 20
    second_pass_min_ratio: float = 0.15
    multi_stage_enabled: bool = True
    assignment_ai_enabled: bool = True


@dataclass
class HunkConfig:
    """Hunk-based grouping configuration."""

    show_hunk_context: bool = True
    context_lines: int = 8
    max_hunks_per_prompt: Optional[int] = None


@dataclass
class AttributionConfig:
    """Attribution message configuration."""

    enabled: bool = False


@dataclass
class Config:
    """Simplified configuration."""

    ai: AIConfig
    hunks: HunkConfig
    attribution: AttributionConfig
    auto_apply: bool = False
    base: str = "main"


class ConfigManager:
    """Simplified configuration manager."""

    def __init__(self):
        self.default_config_path = os.path.expanduser("~/.git-smart-squash.yml")

    def _get_default_model(self, provider: str) -> str:
        """Get the default model for a given provider."""
        defaults = {
            "local": "devstral",
            "openai": "gpt-5",  # GPT-5 with high reasoning capability
            "openrouter": "openai/gpt-4o-mini",
            "anthropic": "claude-sonnet-4-20250514",  # Claude Sonnet 4 model
            "gemini": "gemini-2.5-pro",  # Gemini 2.5 Pro model
        }
        return defaults.get(provider, "devstral")

    def load_config(self, config_path: Optional[str] = None) -> Config:
        """Load configuration from file or create default."""

        # Try to load from specified path, then default path
        paths_to_try = []
        if config_path:
            paths_to_try.append(config_path)

        # Try local project config
        if os.path.exists(".git-smart-squash.yml"):
            paths_to_try.append(".git-smart-squash.yml")

        # Try global config
        paths_to_try.append(self.default_config_path)

        config_data = {}
        for path in paths_to_try:
            if os.path.exists(path):
                try:
                    with open(path, "r") as f:
                        config_data = yaml.safe_load(f) or {}
                    break
                except Exception:
                    continue

        # Create config with provider-aware defaults
        provider = config_data.get("ai", {}).get("provider", "local")
        model = config_data.get("ai", {}).get("model")

        # If no model specified, use provider-specific default
        if not model:
            model = self._get_default_model(provider)

        ai_config = AIConfig(
            provider=provider,
            model=model,
            api_key_env=config_data.get("ai", {}).get("api_key_env"),
            instructions=config_data.get("ai", {}).get("instructions"),
            reasoning=config_data.get("ai", {}).get("reasoning", "low"),
            max_predict_tokens=config_data.get("ai", {}).get(
                "max_predict_tokens", 64000
            ),
            include_branch_context=config_data.get("ai", {}).get(
                "include_branch_context", True
            ),
            max_commit_context=config_data.get("ai", {}).get("max_commit_context", 20),
            include_commit_history=config_data.get("ai", {}).get(
                "include_commit_history", True
            ),
            max_commit_history=config_data.get("ai", {}).get("max_commit_history"),
            second_pass_enabled=config_data.get("ai", {}).get(
                "second_pass_enabled", True
            ),
            second_pass_min_hunks=config_data.get("ai", {}).get(
                "second_pass_min_hunks", 20
            ),
            second_pass_min_ratio=config_data.get("ai", {}).get(
                "second_pass_min_ratio", 0.15
            ),
            multi_stage_enabled=config_data.get("ai", {}).get(
                "multi_stage_enabled", True
            ),
            assignment_ai_enabled=config_data.get("ai", {}).get(
                "assignment_ai_enabled", True
            ),
        )

        # Load hunk configuration
        hunk_config_data = config_data.get("hunks", {})
        hunk_config = HunkConfig(
            show_hunk_context=hunk_config_data.get("show_hunk_context", True),
            context_lines=hunk_config_data.get("context_lines", 8),
            max_hunks_per_prompt=hunk_config_data.get("max_hunks_per_prompt"),
        )

        # Load attribution configuration
        attribution_config_data = config_data.get("attribution", {})
        attribution_config = AttributionConfig(
            enabled=attribution_config_data.get("enabled", False)
        )

        # Load auto-apply setting
        auto_apply = config_data.get("auto_apply", False)

        # Load base branch setting
        base = config_data.get("base", "main")

        return Config(
            ai=ai_config,
            hunks=hunk_config,
            attribution=attribution_config,
            auto_apply=auto_apply,
            base=base,
        )

    def create_default_config(self, global_config: bool = False) -> str:
        """Create a default config file."""
        config = {
            "ai": {
                "provider": "local",
                "model": "devstral",
                "api_key_env": None,
                "include_branch_context": True,
                "max_commit_context": 20,
                "include_commit_history": True,
                "max_commit_history": None,
                "second_pass_enabled": True,
                "second_pass_min_hunks": 20,
                "second_pass_min_ratio": 0.15,
                "multi_stage_enabled": True,
                "assignment_ai_enabled": True,
            },
            "hunks": {
                "show_hunk_context": True,
                "context_lines": 8,
                "max_hunks_per_prompt": None,
            },
            "attribution": {"enabled": False},
            "auto_apply": False,
            "base": "main",
        }

        if global_config:
            path = self.default_config_path
        else:
            path = ".git-smart-squash.yml"

        with open(path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        return path
