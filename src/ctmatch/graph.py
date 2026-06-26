"""Milestone 3/6 (graph): wire the nodes into the self-correcting state machine.

    analyze -> retrieve -> grade -(relevant)-> generate -> verify -> END
                             ^
                             |-(not relevant, retries<MAX)- broaden -|
                             (not relevant, retries>=MAX) -> generate

The conditional edge after `grade` is what makes this an agent rather than a
pipeline. `rank_matches` is the aggregation/ranking stage: most actionable first.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from ctmatch.nodes import MAX_RETRIES, analyze, broaden, generate, grade, retrieve, verify
from ctmatch.schema import AgentState
from ctmatch.tracing import observe

# Lower sorts first: actionable on top, dead ends at the bottom. Stable sort keeps
# retrieval (relevance) order within a tier.
_RANK = {"likely_eligible": 0, "needs_review": 1, "likely_ineligible": 2}


def rank_matches(matches: list[dict]) -> list[dict]:
    """Order trials by eligibility tier, keeping relevance order within each tier."""
    return sorted(matches, key=lambda m: _RANK.get(m.get("overall"), 1))


def _after_grade(state: AgentState) -> str:
    if state.get("grade_ok"):
        return "generate"
    if state.get("retries", 0) < MAX_RETRIES:
        return "broaden"
    return "generate"


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("analyze", analyze)
    g.add_node("retrieve", retrieve)
    g.add_node("grade", grade)
    g.add_node("broaden", broaden)
    g.add_node("generate", generate)
    g.add_node("verify", verify)

    g.set_entry_point("analyze")
    g.add_edge("analyze", "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", _after_grade, {"generate": "generate", "broaden": "broaden"})
    g.add_edge("broaden", "retrieve")
    g.add_edge("generate", "verify")
    g.add_edge("verify", END)
    return g.compile()


@observe(name="match")
def run(patient: str) -> list[dict]:
    """Run the full agent on a patient summary; returns ranked per-trial matches."""
    final = build_graph().invoke({"patient": patient, "retries": 0})
    return rank_matches(final.get("matches", []))
