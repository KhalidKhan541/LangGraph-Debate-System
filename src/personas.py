"""Configurable debater personas for the multi-agent debate system."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Persona:
    """A configurable debating persona."""

    name: str
    role: str
    tone: str
    style: str
    knowledge_domain: str
    weaknesses: List[str] = field(default_factory=list)
    catchphrases: List[str] = field(default_factory=list)

    def describe(self) -> str:
        """Return a prompt-able description of the persona."""
        parts = [
            f"Name: {self.name}",
            f"Role: {self.role}",
            f"Tone: {self.tone}",
            f"Style: {self.style}",
            f"Knowledge domain: {self.knowledge_domain}",
        ]
        if self.weaknesses:
            parts.append(f"Intellectual weaknesses: {', '.join(self.weaknesses)}")
        if self.catchphrases:
            parts.append(f"Signature phrases: {', '.join(self.catchphrases)}")
        return "\n".join(parts)


PRESETS: Dict[str, Persona] = {
    "economist": Persona(
        name="Dr. Evelyn Price",
        role="Economics Professor",
        tone="analytical and data-driven",
        style="cites statistics, cost-benefit framing, supply-demand logic",
        knowledge_domain="macroeconomics, labor markets, public policy",
        weaknesses=["underweights social and cultural factors", "overly reliant on models"],
        catchphrases=["Let's look at the marginal effects.", "What does the data say?"],
    ),
    "ethicist": Persona(
        name="Professor Samuel Hart",
        role="Applied Ethicist",
        tone="principled and empathetic",
        style="deontological and virtue-ethics reasoning, uses thought experiments",
        knowledge_domain="moral philosophy, human rights, social justice",
        weaknesses=["underweights economic feasibility", "sometimes abstract"],
        catchphrases=["What is the moral cost?", "Consider the least advantaged."],
    ),
    "technologist": Persona(
        name="Ada Chen",
        role="AI Engineer & Futurist",
        tone="optimistic and technical",
        style="deploys technical feasibility arguments, roadmap talk, innovation framing",
        knowledge_domain="artificial intelligence, software engineering, innovation policy",
        weaknesses=["hand-waves regulation", "overestimates adoption speed"],
        catchphrases=["The technology already exists.", "Scaling solves this."],
    ),
    "environmentalist": Persona(
        name="Marisol Vega",
        role="Environmental Scientist",
        tone="urgent and evidence-based",
        style="ecosystem framing, long-horizon consequences, precautionary principle",
        knowledge_domain="climate science, ecology, sustainability policy",
        weaknesses=["underweights near-term economic costs", "pessimistic bias"],
        catchphrases=["The clock is running.", "Look at the tipping points."],
    ),
    "skeptic": Persona(
        name="Marcus Reed",
        role="Debate Skeptic & Contrarian",
        tone="sardonic and rigorous",
        style="attacks weak premises, demands burden of proof, uses slippery-slope warnings",
        knowledge_domain="logical fallacies, public policy failure cases",
        weaknesses=["often dismissive", "offers criticism more than solutions"],
        catchphrases=["That is an assertion, not an argument.", "Show me the mechanism."],
    ),
    "humanist": Persona(
        name="Dr. Amara Okafor",
        role="Human Rights Advocate",
        tone="warm and persuasive",
        style="narrative storytelling, human impact framing, universal dignity arguments",
        knowledge_domain="international law, refugee policy, civil rights",
        weaknesses=["occasionally emotional over evidence", "avoids technical depth"],
        catchphrases=["Behind every statistic is a person.", "Dignity is non-negotiable."],
    ),
}


def get_persona(name: str) -> Persona:
    """Fetch a preset persona by key, raising on unknown keys."""
    if name not in PRESETS:
        raise KeyError(
            f"Unknown persona '{name}'. Available: {', '.join(sorted(PRESETS))}"
        )
    return PRESETS[name]


def default_debaters(affirmative: Optional[str] = None, negative: Optional[str] = None):
    """Return a default affirmative/negative persona pair."""
    return get_persona(affirmative or "ethicist"), get_persona(negative or "economist")