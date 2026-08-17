"""Moderator agent: enforces turn limits and topic relevance."""

import logging
from dataclasses import dataclass
from typing import List, Optional

from .llm_client import LLMClient


@dataclass
class ModerationDecision:
    """Outcome of a moderation check."""

    allowed: bool
    reason: str = ""
    action: str = "continue"  # continue | warn | end

    def to_text(self) -> str:
        return f"[Moderator] {self.reason}" if self.reason else "[Moderator] Allowed."


@dataclass
class DebateConfig:
    """Configuration controlling debate flow."""

    topic: str = ""
    max_rounds: int = 3
    max_arguments: int = 6
    min_arguments: int = 2
    topic_keywords: Optional[List[str]] = None
    relevance_threshold: float = 0.25


class Moderator:
    """Enforces turn limits and topic relevance during the debate."""

    def __init__(self, llm: Optional[LLMClient] = None, config: Optional[DebateConfig] = None):
        self.llm = llm or LLMClient()
        self.config = config or DebateConfig()
        self.logger = logging.getLogger(__name__)
        self._turn_count = 0

    def check(self, argument_text: str, argument_count: int) -> ModerationDecision:
        """Check whether an argument may proceed."""
        self._turn_count += 1

        if argument_count > self.config.max_arguments:
            return ModerationDecision(
                allowed=False,
                reason=f"Turn limit reached ({self.config.max_arguments} arguments). Debate concludes.",
                action="end",
            )

        relevance = self._relevance_score(argument_text)
        if relevance < self.config.relevance_threshold:
            return ModerationDecision(
                allowed=False,
                reason=f"Argument is off-topic (relevance {relevance:.2f}).",
                action="warn",
            )

        if self._turn_count > self.config.max_rounds * 2:
            return ModerationDecision(
                allowed=True,
                reason="Final rounds approaching; keep arguments concise.",
                action="warn",
            )

        return ModerationDecision(allowed=True, action="continue")

    @property
    def hard_limit_reached(self) -> bool:
        """True when the debate has exceeded every safety bound."""
        return self._turn_count > max(self.config.max_rounds * 4, self.config.max_arguments + 2)

    def _relevance_score(self, text: str) -> float:
        """Token-overlap relevance of an argument against the topic."""
        if not self.config.topic_keywords:
            return 1.0
        tokens = set(text.lower().split())
        hits = sum(1 for kw in self.config.topic_keywords if kw.lower() in tokens)
        return hits / max(len(self.config.topic_keywords), 1)

    def topic_introduction(self) -> str:
        """Return the moderator's framing of the topic."""
        return (
            f"[Moderator] Topic: {self.config.topic}. "
            f"{self.config.max_rounds} rounds, {self.config.max_arguments} arguments total. "
            "Stay on topic and address the opposition directly."
        )