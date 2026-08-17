"""Verdict + transcript report rendering for the debate system."""

from dataclasses import dataclass, field
from typing import List, Optional

from .debate_agents import Argument
from .graph import DebateState
from .judge import JudgeVerdict
from .moderator import DebateConfig, ModerationDecision


@dataclass
class TranscriptReport:
    """A fully rendered debate report."""

    topic: str
    config: DebateConfig
    arguments: List[Argument]
    decisions: List[ModerationDecision]
    verdict: Optional[JudgeVerdict]
    history: List[str]

    def render(self) -> str:
        """Render the complete report as text."""
        lines = [
            "=" * 70,
            f"DEBATE TRANSCRIPT — {self.topic}",
            "=" * 70,
            "",
        ]

        # Configuration
        lines.append(f"Rounds: {self.config.max_rounds} | "
                     f"Max arguments: {self.config.max_arguments}")
        lines.append("")

        # Arguments
        lines.append("--- Arguments ---")
        for i, arg in enumerate(self.arguments, start=1):
            lines.append(f"[{i}] {arg.to_text()}")
            lines.append("")

        # Verdict
        lines.append("--- Verdict ---")
        if self.verdict is None:
            lines.append("No verdict (debate ended before judging).")
        else:
            v = self.verdict
            lines.append(f"Winner: {v.winner}")
            lines.append(f"Affirmative score: {v.score_affirmative:.2f}")
            lines.append(f"Negative score: {v.score_negative:.2f}")
            lines.append("")
            lines.append("Score breakdown:")
            for b in v.breakdown:
                lines.append(
                    f"  {b.speaker} (#{b.argument_index + 1}): "
                    f"logic {b.logic:.2f} | evidence {b.evidence:.2f} | "
                    f"persuasion {b.persuasion:.2f} | total {b.total():.2f}"
                )
            lines.append("")
            lines.append(f"Rationale: {v.rationale}")

        lines.append("")
        lines.append("--- Moderation Log ---")
        for d in self.decisions:
            lines.append(f"  {d.to_text()}")

        lines.append("")
        lines.append("--- Full History ---")
        for h in self.history:
            lines.append(f"  {h}")

        return "\n".join(lines)

    def to_markdown(self) -> str:
        """Render the report as markdown."""
        lines = [f"# Debate Transcript: {self.topic}", ""]
        if self.verdict:
            lines.append(f"## Verdict: **{self.verdict.winner}**")
            lines.append(f"- Affirmative: {self.verdict.score_affirmative:.2f}")
            lines.append(f"- Negative: {self.verdict.score_negative:.2f}")
            lines.append("")
            lines.append(f"Rationale: {self.verdict.rationale}")
            lines.append("")
            lines.append("### Score Breakdown")
            lines.append("| # | Speaker | Logic | Evidence | Persuasion | Total |")
            lines.append("|---|---------|-------|----------|------------|-------|")
            for b in self.verdict.breakdown:
                lines.append(
                    f"| {b.argument_index + 1} | {b.speaker} | {b.logic:.2f} | "
                    f"{b.evidence:.2f} | {b.persuasion:.2f} | {b.total():.2f} |"
                )
            lines.append("")
        lines.append("## Arguments")
        for i, arg in enumerate(self.arguments, start=1):
            lines.append(f"### [{i}] {arg.speaker} ({arg.stance})")
            if arg.claim:
                lines.append(f"**Claim:** {arg.claim}")
            if arg.evidence:
                lines.append(f"**Evidence:** {arg.evidence}")
            if arg.rebuttal:
                lines.append(f"**Rebuttal:** {arg.rebuttal}")
            lines.append("")
        lines.append("## Moderation Log")
        for d in self.decisions:
            lines.append(f"- {d.to_text()}")
        return "\n".join(lines)


def build_report(state: DebateState) -> TranscriptReport:
    """Build a TranscriptReport from a finished DebateState."""
    return TranscriptReport(
        topic=state.topic,
        config=state.config,
        arguments=state.arguments,
        decisions=state.decisions,
        verdict=state.verdict,
        history=state.history,
    )