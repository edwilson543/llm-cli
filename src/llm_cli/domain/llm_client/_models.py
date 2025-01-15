from __future__ import annotations

import enum


class Model(enum.Enum):
    # Anthropic.
    CLAUDE_SONNET = "claude-3-5-sonnet-latest"
    CLAUDE_HAIKU = "claude-3-5-haiku-latest"
    CLAUDE_OPUS = "claude-3-opus-latest"

    # xAI.
    GROK_2 = "grok-2-latest"

    # Local implementations.
    ECHO = "echo"
    BROKEN = "broken"

    @classmethod
    def available_models(cls) -> list[Model]:
        return [model for model in cls if model not in cls.unavailable_models()]

    @classmethod
    def unavailable_models(cls) -> list[Model]:
        return [cls.BROKEN]
