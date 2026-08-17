# Multi-Agent Debate + Judge System

Two LLM agents argue opposite positions on any topic with **configurable personas**, structured **claim → evidence → rebuttal** format, a **moderator agent** enforcing turn limits and topic relevance, and a **judge agent** scoring each argument on **logic / evidence / persuasion** with a rubric — producing a final **verdict + transcript report**. Orchestrated as a **LangGraph state machine**.

```
                 ┌────────────────────────────────────────────────┐
                 │            LANGGRAPH STATE MACHINE             │
                 └────────────────────────────────────────────────┘
     START ──► ┌──────────────────┐
               │  MODERATOR_INTRO │  frames topic, rules, rounds
               └────────┬─────────┘
                        ▼
               ┌──────────────────┐        ┌──────────────────┐
               │    DEBATER_A     │───────►│    DEBATER_B     │
               │ affirmative      │◄───────│ negative         │
               │ claim→evidence   │        │ claim→evidence   │
               │ →rebuttal        │        │ →rebuttal        │
               └────────┬─────────┘        └────────┬─────────┘
                        │  conditional edge          │
                        ▼  (turn limit? off-topic?)  ▼
               ┌──────────────────┐
               │      JUDGE       │  rubric: logic 40% · evidence 35%
               │  verdict+report  │  · persuasion 25%
               └────────┬─────────┘
                        ▼
                     END ──► TranscriptReport (txt + markdown)
```

## Features

| # | Feature | Description | Where |
|---|---------|-------------|-------|
| 1 | **LangGraph state machine** | `StateGraph` with conditional edges; bundled fallback when langgraph isn't installed | `src/graph.py` |
| 2 | **Two debating agents** | Opposite stances, structured `Argument` (claim/evidence/rebuttal) parsing | `src/debate_agents.py` |
| 3 | **Configurable personas** | 6 preset personas + prompt-able persona descriptions | `src/personas.py` |
| 4 | **Moderator agent** | Turn limits, max arguments, topic-relevance scoring, warnings | `src/moderator.py` |
| 5 | **Judge agent** | Rubric scoring: logic (0.4) · evidence (0.35) · persuasion (0.25) | `src/judge.py` |
| 6 | **Verdict + transcript** | Text and Markdown reports with score breakdown, moderation log, full history | `src/transcript.py` |
| 7 | **Offline fallback** | Deterministic template LLM so the system runs with zero dependencies | `src/llm_client.py` |

## Quick Start

```bash
pip install -r requirements.txt

# Run with defaults (offline fallback LLM)
python main.py

# Custom debate
python main.py --topic "Should universal basic income be implemented?" \
               --rounds 4 --persona-a humanist --persona-b economist
```

Output: full transcript printed to console + `debate_transcript.md` written to disk.

## Architecture

```
LangGraph-Debate-System/
├── src/
│   ├── graph.py          # LangGraph StateGraph (moderator → debaters ⇄ → judge)
│   ├── personas.py       # Persona dataclass + 6 presets (economist, ethicist, ...)
│   ├── llm_client.py     # LLM wrapper w/ deterministic offline fallback
│   ├── debate_agents.py  # DebateAgent, Argument, prompt builder
│   ├── moderator.py      # Moderator, DebateConfig, ModerationDecision
│   ├── judge.py          # JudgeAgent, rubric criteria, ScoreBreakdown, Verdict
│   └── transcript.py     # TranscriptReport (render + to_markdown)
├── main.py               # CLI demo
├── requirements.txt
└── README.md
```

## The State Machine (`src/graph.py`)

The pipeline is a `StateGraph` over a shared `DebateState`:

| Node | Role |
|------|------|
| `moderator_intro` | Moderator frames the topic, rounds, and argument cap |
| `debater_a` | Affirmative persona argues (claim → evidence → rebuttal) |
| `debater_b` | Negative persona argues; rebuts A |
| `judge` | Rubric scores, winner, rationale |

**Conditional edges** route between `debater_a` ⇄ `debater_b` until the
moderator signals `end` (turn limit) or the argument cap is reached — then the
graph routes to `judge` and terminates.

> `LANGGRAPH_AVAILABLE` auto-detects `langgraph`. If missing, a bundled
> `StateGraph` fallback with the same API runs the exact same topology.

## Personas (`src/personas.py`)

Each `Persona` carries a name, role, tone, style, knowledge domain, weaknesses,
and catchphrases — rendered into the prompt. Presets:

| Key | Persona | Domain |
|-----|---------|--------|
| `economist` | Dr. Evelyn Price | economics, cost-benefit |
| `ethicist` | Prof. Samuel Hart | moral philosophy |
| `technologist` | Ada Chen | AI, innovation policy |
| `environmentalist` | Marisol Vega | climate science |
| `skeptic` | Marcus Reed | logic, policy failures |
| `humanist` | Dr. Amara Okafor | human rights |

```python
from src.personas import Persona
custom = Persona(
    name="Dr. Critique", role="Legal Scholar", tone="formal and adversarial",
    style="precedent-based reasoning", knowledge_domain="constitutional law",
    weaknesses=["ignores empirical data"], catchphrases=["Precedent is clear."],
)
```

## Moderator (`src/moderator.py`)

`Moderator.check(argument, count)` returns a `ModerationDecision`:

- `allowed=False, action="end"` — argument cap reached → route to judge
- `allowed=False, action="warn"` — relevance score below threshold (token-overlap vs topic keywords) → argument rejected
- `allowed=True, action="continue"` — proceed to next debater

## Judge & Rubric (`src/judge.py`)

Every argument gets a `ScoreBreakdown`:

| Criterion | Weight | Heuristic signal |
|-----------|--------|------------------|
| **Logic** | 0.40 | causal connectors (because, therefore, thus…) |
| **Evidence** | 0.35 | data markers (study, statistics, %, research…) |
| **Persuasion** | 0.25 | rebuttal presence, framing strength, length |

Side scores are averaged per side; the winner is the side with the higher
score. The rationale highlights the strongest individual argument.

## Reports (`src/transcript.py`)

`TranscriptReport.render()` prints: topic, config, numbered structured
arguments, verdict + score breakdown + rationale, moderation log, and the
full node history. `to_markdown()` produces a publishable `.md`.

## Wiring in a Real LLM

Any callable works — swap the fallback for a real model:

```python
from src.llm_client import LLMClient
import openai

client = openai.OpenAI()

def call(prompt: str) -> str:
    return client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    ).choices[0].message.content

llm = LLMClient(callable_fn=call)   # keep the same API
```

## Example Output

```
DEBATE TRANSCRIPT — Should artificial intelligence be heavily regulated?
========================================================================
Rounds: 3 | Max arguments: 6

--- Arguments ---
[1] **Prof. Samuel Hart** (affirmative):
- Claim: The motion deserves support because ...
- Evidence: Evidence supports this: relevant statistics ...
- Rebuttal: No prior argument to rebut.

--- Verdict ---
Winner: Prof. Samuel Hart
Affirmative score: 0.78
Negative score: 0.62

Score breakdown:
  Prof. Samuel Hart (#1): logic 0.67 | evidence 1.00 | persuasion 0.70 | total 2.37
  ...
Rationale: Strongest single argument by ...

--- Moderation Log ---
  [Moderator] Allowed.
  ...
```

## Dependencies

```
langgraph>=0.2.0      # state machine (optional — bundled fallback)
langchain-core>=0.2.0
openai>=1.30.0        # only needed when wiring a real LLM
pydantic>=2.0.0
```

The system runs **out of the box with no dependencies installed** via the
deterministic offline LLM fallback.