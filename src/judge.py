"""Judge agent: scores arguments on logic / evidence / persuasion with a rubric."""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .debate_agents import Argument
from .llm_client import LLMClient


@dataclass
class RubricCriterion:
    """One scoring criterion in the judge's rubric."""

    key: str
    label: str
    weight: float
    description: str


@dataclass
class ScoreBreakdown:
    """Per-criterion scores for a single argument."""

    argument_index: int
    speaker: str
    logic: float = 0.0
    evidence: float = 0.0
    persuasion: float = 0.0

    def total(self) -> float:
        """Weighted total score."""
        return self.logic + self.evidence + self.persuasion


@dataclass
class JudgeVerdict:
    """Final verdict from the judge."""

    winner: str
    score_affirmative: float
    score_negative: float
    breakdown: List[ScoreBreakdown]
    rationale: str = ""
    reasons: List[str] = field(default_factory=list)


DEFAULT_RUBRIC: Dict[str, RubricCriterion] = {
    "logic": RubricCriterion("logic", "Logic", 0.4,
                             "Validity of reasoning, internal consistency, absence of fallacies"),
    "evidence": RubricCriterion("evidence", "Evidence", 0.35,
                                "Quality and relevance of evidence, citations, data support"),
    "persuasion": RubricCriterion("persuasion", "Persuasion", 0.25,
                                  "Rhetorical force, clarity, impact on the audience"),
}


class JudgeAgent:
    """Scores all arguments on a rubric and issues a verdict."""

    def __init__(self, llm: Optional[LLMClient] = None,
                 rubric: Optional[Dict[str, RubricCriterion]] = None):
        self.llm = llm or LLMClient()
        self.rubric = rubric or DEFAULT_RUBRIC
        self.logger = logging.getLogger(__name__)

    def judge(self, topic: str, arguments: List[Argument]) -> JudgeVerdict:
        """Score every argument and produce a final verdict."""
        breakdown = [self._score_argument(i, arg) for i, arg in enumerate(arguments)]

        aff_total = sum(b.total() for b in breakdown if b.speaker == arguments[0].speaker)
        neg_total = sum(b.total() for b in breakdown if b.speaker != arguments[0].speaker)

        # Normalize by number of arguments per side
        aff_count = max(sum(1 for b in breakdown if b.speaker == arguments[0].speaker), 1)
        neg_count = max(sum(1 for b in breakdown if b.speaker != arguments[0].speaker), 1)
        aff_score = aff_total / aff_count
        neg_score = neg_total / neg_count

        winner = arguments[0].speaker if aff_score >= neg_score else arguments[1].speaker

        reasons = self._rationale(topic, arguments, breakdown, winner)
        return JudgeVerdict(
            winner=winner,
            score_affirmative=aff_score,
            score_negative=neg_score,
            breakdown=breakdown,
            rationale=" ".join(reasons),
            reasons=reasons,
        )

    def _score_argument(self, index: int, arg: Argument) -> ScoreBreakdown:
        """Score one argument heuristically (offline); LLM path overridable."""
        text = f"{arg.claim} {arg.evidence} {arg.rebuttal}"
        tokens = text.split()

        logic = self._score_logic(text)
        evidence = self._score_evidence(text)
        persuasion = self._score_persuasion(text)

        return ScoreBreakdown(
            argument_index=index,
            speaker=arg.speaker,
            logic=logic,
            evidence=evidence,
            persuasion=persuasion,
        )

    def _score_logic(self, text: str) -> float:
        """Heuristic logic score: presence of causal connectors and structure."""
        connectors = ["because", "therefore", "thus", "since", "hence", "consequently",
                      "if", "then", "implies", "demonstrates"]
        count = sum(1 for c in connectors if c in text.lower())
        return min(count / 3.0, 1.0)

    def _score_evidence(self, text: str) -> float:
        """Heuristic evidence score: mentions of data/studies/stats."""
        markers = ["evidence", "study", "data", "statistics", "research", "percent",
                   "%", "report", "survey", "documented", "case"]
        count = sum(1 for m in markers if m in text.lower())
        return min(count / 3.0, 1.0)

    def _score_persuasion(self, text: str) -> float:
        """Heuristic persuasion score: framing strength, rebuttal presence."""
        score = 0.2
        if len(text) > 150:
            score += 0.2
        if text.lower().count("rebuttal") > 0 or "rebut" in text.lower():
            score += 0.3
        if any(w in text.lower() for w in ["must", "should", "clearly", "undeniably", "critical"]):
            score += 0.2
        return min(score, 1.0)

    def _rationale(self, topic, arguments, breakdown, winner) -> List[str]:
        """Generate a short rationale for the verdict."""
        reasons = []
        strongest = max(breakdown, key=lambda b: b.total())
        reasons.append(
            f"Strongest single argument by {strongest.speaker} "
            f"(logic {strongest.logic:.2f}, evidence {strongest.evidence:.2f}, "
            f"persuasion {strongest.persuasion:.2f})."
        )
        if strongest.speaker != winner:
            reasons.append("Despite a strong individual turn, aggregate side performance favored the winner.")
        return reasons