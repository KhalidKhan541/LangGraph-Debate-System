"""LangGraph state machine for the debate pipeline.

Uses langgraph when installed; otherwise falls back to a bundled
mini state-machine with the same StateGraph API so the system runs anywhere.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .debate_agents import Argument, DebateAgent
from .judge import JudgeAgent, JudgeVerdict
from .moderator import DebateConfig, ModerationDecision, Moderator

try:
    from langgraph.graph import END, StateGraph
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    END = "END"

    class StateGraph:
        """Minimal StateGraph fallback mirroring langgraph's API."""

        def __init__(self, state_schema):
            self.state_schema = state_schema
            self.nodes: Dict[str, Callable] = {}
            self.edges: Dict[str, Dict[str, Any]] = {}
            self._entry = None
            self._finish = None

        def add_node(self, name, fn):
            self.nodes[name] = fn
            return self

        def add_edge(self, start, end):
            self.edges.setdefault(start, {}).setdefault("edges", []).append(end)
            return self

        def add_conditional_edges(self, start, router, mapping=None):
            self.edges.setdefault(start, {})["conditional"] = (router, mapping)
            return self

        def set_entry_point(self, name):
            self._entry = name
            return self

        def set_finish_point(self, name):
            self._finish = name
            return self

        def compile(self):
            return self

        def invoke(self, initial_state: Dict) -> Dict:
            """Run the graph, threading a schema instance between nodes."""
            state = self.state_schema(**dict(initial_state))
            current = self._entry
            visited = set()
            while current and current != self._finish and current != END:
                if current in visited:
                    raise RuntimeError(f"Cycle detected at node '{current}'")
                visited.add(current)
                update = self.nodes[current](state)
                if isinstance(update, dict):
                    for key, value in update.items():
                        setattr(state, key, value)
                edge = self.edges.get(current, {})
                if "conditional" in edge:
                    router, mapping = edge["conditional"]
                    nxt = router(state)
                    current = mapping.get(nxt, nxt) if mapping else nxt
                else:
                    nxt = (edge.get("edges") or [self._finish])[0]
                    current = nxt if nxt else self._finish
            return {f: getattr(state, f) for f in state.__dataclass_fields__}


@dataclass
class DebateState:
    """Shared state threaded through the LangGraph pipeline."""

    topic: str
    config: DebateConfig
    arguments: List[Argument] = field(default_factory=list)
    decisions: List[ModerationDecision] = field(default_factory=list)
    verdict: Optional[JudgeVerdict] = None
    transcript: str = ""
    history: List[str] = field(default_factory=list)


class DebateGraph:
    """Builds and runs the LangGraph debate state machine.

    Nodes are closures over the agents, moderator and judge so no
    non-schema attributes ever live on the state object (langgraph
    reconstructs state from its channel schema between nodes). Nodes
    receive the schema instance and return partial dict updates.
    """

    def __init__(self, agent_a: DebateAgent, agent_b: DebateAgent,
                 moderator: Moderator, judge: JudgeAgent,
                 use_langgraph: Optional[bool] = None):
        self.agent_a = agent_a
        self.agent_b = agent_b
        self.moderator = moderator
        self.judge = judge
        self.use_langgraph = LANGGRAPH_AVAILABLE if use_langgraph is None else use_langgraph
        self.logger = logging.getLogger(__name__)

    # -- node factories -----------------------------------------------------

    def _make_intro_node(self):
        moderator = self.moderator

        def _node_moderator_intro(state: DebateState) -> Dict:
            history = list(state.history)
            history.append(moderator.topic_introduction())
            history.append("[Moderator] Debate begins.")
            return {"history": history}

        return _node_moderator_intro

    def _make_debater_node(self, agent: DebateAgent):
        moderator = self.moderator

        def _debater_turn(state: DebateState) -> Dict:
            arguments = list(state.arguments)
            decisions = list(state.decisions)
            history = list(state.history)

            arg = agent.argue(state.topic, arguments, turn=len(arguments))
            decision = moderator.check(arg.to_text(), len(arguments) + 1)
            decisions.append(decision)
            if decision.allowed:
                arguments.append(arg)
                history.append(arg.to_text())
            else:
                history.append(decision.to_text())
            return {"arguments": arguments, "decisions": decisions, "history": history}

        return _debater_turn

    def _make_judge_node(self):
        judge = self.judge

        def _node_judge(state: DebateState) -> Dict:
            history = list(state.history)
            if len(state.arguments) < 2:
                history.append("[Moderator] Insufficient arguments for a verdict.")
                verdict = JudgeVerdict(
                    winner="No contest",
                    score_affirmative=0.0,
                    score_negative=0.0,
                    breakdown=[],
                    rationale="Debate ended early.",
                )
            else:
                verdict = judge.judge(state.topic, state.arguments)
                history.append(
                    f"[Judge] Winner: {verdict.winner} | "
                    f"Affirmative {verdict.score_affirmative:.2f} vs "
                    f"Negative {verdict.score_negative:.2f}"
                )
                history.append(f"[Judge] Rationale: {verdict.rationale}")
            return {"verdict": verdict, "history": history}

        return _node_judge

    def _make_router(self):
        moderator = self.moderator

        def _should_continue(state: DebateState) -> str:
            if len(state.arguments) >= state.config.max_arguments:
                return "judge"
            if state.decisions and state.decisions[-1].action == "end":
                return "judge"
            if moderator.hard_limit_reached:
                return "judge"
            return "opponent"

        return _should_continue

    # -- graph construction -------------------------------------------------

    def _build(self):
        builder = StateGraph(DebateState)
        builder.add_node("moderator_intro", self._make_intro_node())
        builder.add_node("debater_a", self._make_debater_node(self.agent_a))
        builder.add_node("debater_b", self._make_debater_node(self.agent_b))
        builder.add_node("judge", self._make_judge_node())

        builder.set_entry_point("moderator_intro")
        builder.add_edge("moderator_intro", "debater_a")
        builder.add_edge("judge", END)

        router = self._make_router()
        builder.add_conditional_edges("debater_a", router, {"judge": "judge", "opponent": "debater_b"})
        builder.add_conditional_edges("debater_b", router, {"judge": "judge", "opponent": "debater_a"})
        return builder.compile()

    # -- execution ----------------------------------------------------------

    def run(self, topic: str, config: DebateConfig) -> DebateState:
        """Execute the debate and return the final state."""
        self.moderator.config = config
        initial = {
            "topic": topic,
            "config": config,
            "arguments": [],
            "decisions": [],
            "verdict": None,
            "transcript": "",
            "history": [],
        }
        graph = self._build()
        final = graph.invoke(initial)
        return DebateState(
            topic=final["topic"],
            config=final["config"],
            arguments=final.get("arguments") or [],
            decisions=final.get("decisions") or [],
            verdict=final.get("verdict"),
            transcript="\n\n".join(final.get("history") or []),
            history=final.get("history") or [],
        )