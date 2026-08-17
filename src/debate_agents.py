"""Debater agents: two LLM agents arguing opposite positions with personas."""

import logging
from dataclasses import dataclass
from typing import List, Optional

from .personas import Persona
from .llm_client import LLMClient


@dataclass
class Argument:
    """A single structured argument from a debater."""

    speaker: str
    stance: str
    claim: str = ""
    evidence: str = ""
    rebuttal: str = ""
    raw: str = ""
    turn: int = 0

    def to_text(self) -> str:
        """Render the argument in structured claim -> evidence -> rebuttal format."""
        lines = [f"**{self.speaker}** ({self.stance}):"]
        if self.claim:
            lines.append(f"- Claim: {self.claim}")
        if self.evidence:
            lines.append(f"- Evidence: {self.evidence}")
        if self.rebuttal:
            lines.append(f"- Rebuttal: {self.rebuttal}")
        return "\n".join(lines)


class DebatePromptBuilder:
    """Builds prompts for debater agents."""

    @staticmethod
    def build(
        topic: str,
        stance: str,
        persona: Persona,
        previous_arguments: List[Argument],
        max_tokens_hint: int = 300,
    ) -> str:
        """Construct a structured prompt for one debater turn."""
        prev_text = ""
        if previous_arguments:
            prev_text = "\n".join(
                f"[{a.speaker}] {a.claim} {a.evidence}" for a in previous_arguments[-3:]
            )
        return (
            f"POSITION: {topic}\n"
            f"STANCE: {stance}\n"
            f"PERSONA: {persona.describe()}\n"
            f"TASK: Argue the {stance} side of the motion in structured format.\n"
            f"FORMAT: Claim -> Evidence -> Rebuttal (rebut the opponent's latest point).\n"
            f"PREVIOUS: {prev_text or 'None (opening statement)'}\n"
            f"Respond with Claim:, Evidence:, Rebuttal: sections."
        )


class DebateAgent:
    """An LLM agent that argues one side of the debate."""

    def __init__(self, persona: Persona, stance: str, llm: LLMClient):
        self.persona = persona
        self.stance = stance
        self.llm = llm
        self.logger = logging.getLogger(__name__)

    def argue(self, topic: str, previous: List[Argument], turn: int) -> Argument:
        """Produce one structured argument turn."""
        prompt = DebatePromptBuilder.build(topic, self.stance, self.persona, previous)
        raw = self.llm.generate(prompt)
        return self._parse(raw, turn)

    def _parse(self, raw: str, turn: int) -> Argument:
        """Parse Claim/Evidence/Rebuttal sections from raw LLM output."""
        claim = self._section(raw, "Claim")
        evidence = self._section(raw, "Evidence")
        rebuttal = self._section(raw, "Rebuttal")
        return Argument(
            speaker=self.persona.name,
            stance=self.stance,
            claim=claim,
            evidence=evidence,
            rebuttal=rebuttal,
            raw=raw,
            turn=turn,
        )

    @staticmethod
    def _section(text: str, key: str) -> str:
        """Extract a labeled section from LLM output."""
        lines = text.splitlines()
        in_section = False
        parts = []
        for line in lines:
            stripped = line.strip()
            if stripped.lower().startswith(key.lower() + ":"):
                in_section = True
                parts.append(stripped[len(key) + 1:].strip())
            elif in_section:
                if ":" in stripped[:12] and not stripped[0].isspace():
                    break
                parts.append(stripped)
        return " ".join(p for p in parts if p).strip()