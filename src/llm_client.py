"""LLM abstraction with deterministic fallback for the debate system."""

import logging
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


@dataclass
class LLMConfig:
    """Configuration for the LLM client."""

    provider: str = "fallback"
    model: str = "local-template"
    temperature: float = 0.7
    max_tokens: int = 600
    api_key: Optional[str] = None
    base_url: Optional[str] = None


class LLMClient:
    """Thin LLM wrapper.

    Uses a provided ``callable`` when given (any OpenAI-compatible client),
    otherwise falls back to a deterministic template-based generator so the
    debate system runs fully offline.
    """

    def __init__(self, config: Optional[LLMConfig] = None, callable_fn: Optional[Callable[[str], str]] = None):
        self.config = config or LLMConfig()
        self.callable_fn = callable_fn
        self.logger = logging.getLogger(__name__)

    def generate(self, prompt: str) -> str:
        """Generate text for a prompt."""
        if self.callable_fn is not None:
            return self.callable_fn(prompt)
        return self._fallback_generate(prompt)

    def _fallback_generate(self, prompt: str) -> str:
        """Deterministic template-based generator (offline mode).

        Parses structured hints embedded in the prompt (POSITION, PERSONA,
        STANCE, PREVIOUS) and produces a claim->evidence->rebuttal response.
        """
        position = self._extract(prompt, "POSITION") or "the motion"
        stance = self._extract(prompt, "STANCE") or "neutral"
        persona = self._extract(prompt, "PERSONA") or "Analyst"
        previous = self._extract(prompt, "PREVIOUS") or ""

        seed = random.Random(hash(prompt) % 10000)
        templates = self._templates(stance)

        claims = [
            f"{templates['claim']} {position}",
            f"{templates['claim2']} {position}",
        ]
        evidence = f"Evidence supports this: relevant statistics and documented case studies indicate the consequences are significant and measurable."
        rebuttal = ""
        if previous:
            rebuttal = f"Rebuttal to prior argument: the previous claim overstates its evidence and ignores countervailing factors; {position} requires weighing trade-offs carefully."

        return (
            f"[Persona: {persona}]\n"
            f"Claim: {claims[0]}\n"
            f"Evidence: {evidence}\n"
            f"Rebuttal: {rebuttal or 'No prior argument to rebut.'}"
        )

    def _templates(self, stance: str) -> Dict[str, str]:
        if stance.lower() == "affirmative":
            return {
                "claim": "The motion deserves support because",
                "claim2": "Strong historical precedent demonstrates that",
            }
        if stance.lower() == "negative":
            return {
                "claim": "The motion is unsound because",
                "claim2": "Critical flaws emerge when",
            }
        return {
            "claim": "A careful analysis of",
            "claim2": "The most defensible position on",
        }

    @staticmethod
    def _extract(prompt: str, key: str) -> str:
        """Extract a `KEY: value` line from the prompt."""
        for line in prompt.splitlines():
            line = line.strip()
            if line.startswith(f"{key}:"):
                return line[len(key) + 1:].strip()
        return ""


def openai_compatible(callable_fn):
    """Factory: wrap an OpenAI-compatible callable in an LLMClient."""
    return LLMClient(callable_fn=callable_fn)