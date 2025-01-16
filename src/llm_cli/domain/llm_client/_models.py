from __future__ import annotations

import dataclasses
import enum


class Vendor(enum.Enum):
    ANTHROPIC = "ANTHROPIC"
    XAI = "XAI"
    FAKE_AI = "FAKE_AI"


@dataclasses.dataclass(frozen=True)
class Model:
    vendor: Vendor
    friendly_name: str
    official_name: str


# Anthropic.
CLAUDE_HAIKU = Model(
    vendor=Vendor.ANTHROPIC,
    friendly_name="claude-haiku",
    official_name="claude-3-5-haiku-latest",
)
CLAUDE_SONNET = Model(
    vendor=Vendor.ANTHROPIC,
    friendly_name="claude-sonnet",
    official_name="claude-3-5-sonnet-latest",
)
CLAUDE_OPUS = Model(
    vendor=Vendor.ANTHROPIC,
    friendly_name="claude-opus",
    official_name="claude-3-opus-latest",
)

# xAI.
GROK_2 = Model(vendor=Vendor.XAI, friendly_name="grok-2", official_name="grok-2-latest")

# Fakes.
ECHO = Model(vendor=Vendor.FAKE_AI, friendly_name="echo", official_name="echo")
BROKEN = Model(vendor=Vendor.FAKE_AI, friendly_name="broken", official_name="broken")
