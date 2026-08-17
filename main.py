"""Demo: run a multi-agent debate with moderator and judge."""

import argparse

from src.llm_client import LLMClient
from src.personas import get_persona
from src.debate_agents import DebateAgent
from src.moderator import Moderator, DebateConfig
from src.judge import JudgeAgent
from src.graph import DebateGraph, LANGGRAPH_AVAILABLE
from src.transcript import build_report


def main():
    parser = argparse.ArgumentParser(description="Multi-agent debate + judge system")
    parser.add_argument("--topic", default="Should artificial intelligence be heavily regulated?",
                        help="Debate motion")
    parser.add_argument("--rounds", type=int, default=3, help="Number of debate rounds")
    parser.add_argument("--persona-a", default="ethicist", help="Affirmative persona")
    parser.add_argument("--persona-b", default="economist", help="Negative persona")
    args = parser.parse_args()

    print("=" * 70)
    print("MULTI-AGENT DEBATE + JUDGE SYSTEM (LangGraph)")
    print(f"LangGraph: {'installed' if LANGGRAPH_AVAILABLE else 'fallback state machine'}")
    print("=" * 70)

    llm = LLMClient()

    agent_a = DebateAgent(get_persona(args.persona_a), "affirmative", llm)
    agent_b = DebateAgent(get_persona(args.persona_b), "negative", llm)
    moderator = Moderator(llm)
    judge = JudgeAgent(llm)

    config = DebateConfig(
        topic=args.topic,
        max_rounds=args.rounds,
        max_arguments=args.rounds * 2,
        topic_keywords=[w for w in args.topic.lower().split() if len(w) > 3],
    )

    graph = DebateGraph(agent_a, agent_b, moderator, judge)
    state = graph.run(args.topic, config)

    report = build_report(state)
    print(report.render())

    # Save markdown transcript
    out_path = "debate_transcript.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report.to_markdown())
    print(f"\n[Saved] Transcript written to {out_path}")


if __name__ == "__main__":
    main()