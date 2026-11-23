"""
graph.py
Cyclic LangGraph StateGraph Workflow Orchestrator and Markdown Academic Brief Generator.

Workflow Architecture:
[User Query]
      │
      ▼
┌──────────────┐
│Research Agent│ ──(Conversational Greeting)──┐
└──────────────┘                              │
      │ (Academic Research)                   │
      ▼                                       │
┌──────────────┐                              │
│Analysis Agent│                              │
└──────────────┘                              │
      │                                       │
      ▼                                       │
┌──────────────┐ <─────────────────────────────┘
│ Writing Agent│
└──────────────┘
      │
      ▼
┌─────────────────┐
│Proofreader Agent│
└─────────────────┘
      │
      ▼
    [END]

Course: Natural Language Processing (NLP)
Team: Sameer Sirsath (PRN: 202401110023) & Rameshwar Sanap (PRN: 202401110021)
"""

import os
import datetime
from typing import Dict, Any, Optional

from langgraph.graph import StateGraph, END
from src.agents import (
    AgentState,
    research_agent,
    analysis_agent,
    writing_agent,
    proofreader_agent
)


def route_after_research(state: AgentState) -> str:
    """
    Conditional routing edge from Research Agent:
    - If intent is conversational greeting -> skip to Writing Agent
    - If intent is academic research -> proceed to Analysis Agent
    """
    intent = state.get("intent", "academic_research")
    if intent == "conversational":
        return "writing"
    return "analysis"


def build_research_graph() -> Any:
    """
    Constructs and compiles the 4-agent LangGraph StateGraph workflow.
    """
    workflow = StateGraph(AgentState)

    # 1. Add Agent Nodes
    workflow.add_node("research", research_agent)
    workflow.add_node("analysis", analysis_agent)
    workflow.add_node("writing", writing_agent)
    workflow.add_node("proofreader", proofreader_agent)

    # 2. Set Entry Point
    workflow.set_entry_point("research")

    # 3. Add Conditional Routing
    workflow.add_conditional_edges(
        "research",
        route_after_research,
        {
            "analysis": "analysis",
            "writing": "writing"
        }
    )

    # 4. Add Standard Directed Edges
    workflow.add_edge("analysis", "writing")
    workflow.add_edge("writing", "proofreader")
    workflow.add_edge("proofreader", END)

    # 5. Compile Graph
    return workflow.compile()


# Singleton compiled graph
_COMPILED_GRAPH = None


def get_compiled_graph():
    """Returns singleton compiled LangGraph instance."""
    global _COMPILED_GRAPH
    if _COMPILED_GRAPH is None:
        _COMPILED_GRAPH = build_research_graph()
    return _COMPILED_GRAPH


def run_research_workflow(query: str,
                          force_arxiv: bool = False,
                          model_backend: Optional[str] = None) -> AgentState:
    """
    Executes the end-to-end multi-agent research workflow on a user query.
    Returns the complete populated AgentState.
    """
    initial_state: AgentState = {
        "query": query,
        "english_query": "",
        "detected_language": "en",
        "is_translated": False,
        "intent": "academic_research",
        "force_arxiv": force_arxiv,
        "arxiv_searched": False,
        "retrieved_chunks": [],
        "entities": [],
        "chunk_sentiments": [],
        "draft_answer": "",
        "final_answer": "",
        "provenance": {},
        "execution_trace": []
    }

    graph = get_compiled_graph()
    final_state = graph.invoke(initial_state)
    return final_state


def generate_markdown_report(state: AgentState) -> str:
    """
    Generates a structured, publication-grade academic Markdown research brief
    summarizing query intent, agent execution trace, retrieved evidence, and final answer.
    """
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    query = state.get("query", "N/A")
    english_query = state.get("english_query", query)
    detected_lang = state.get("detected_language", "en")
    chunks = state.get("retrieved_chunks", [])
    entities = state.get("entities", [])
    sentiments = state.get("chunk_sentiments", [])
    final_answer = state.get("final_answer", "")
    trace = state.get("execution_trace", [])

    report = (
        "# 🎓 Academic Research Synthesis Brief\n\n"
        f"**Course:** Natural Language Processing (NLP)  \n"
        f"**Project Title:** Agentic RAG-Based AI Research Assistant  \n"
        f"**Authors:** Sameer Sirsath (PRN: 202401110023) & Rameshwar Sanap (PRN: 202401110021)  \n"
        f"**Generated At:** {now}  \n\n"
        "---\n\n"
        "## 1. Inquiry Specification\n"
        f"- **Original Query:** `{query}`\n"
        f"- **Normalized English Query:** `{english_query}`\n"
        f"- **Detected Language:** `{detected_lang}`\n\n"
        "## 2. Synthesized Scholarly Answer\n\n"
        f"{final_answer}\n\n"
        "---\n\n"
        "## 3. Multi-Agent Execution Trace\n"
        "| Time | Agent | Action Description |\n"
        "| :--- | :--- | :--- |\n"
    )

    for item in trace:
        report += f"| {item.get('timestamp', '')} | {item.get('agent', '')} | {item.get('action', '')} |\n"

    report += "\n## 4. Empirical Evidence Passages\n"
    for idx, c in enumerate(chunks):
        report += (
            f"### Passage #{idx+1} [Source: {c.get('source')} | Page: {c.get('page')}]\n"
            f"- **Hybrid Relevance Score:** {c.get('score', 0.0):.4f}\n"
            f"- **Excerpt:**\n> {c.get('content')}\n\n"
        )

    if entities:
        report += "## 5. Extracted Named Entities (spaCy)\n"
        report += "| Entity Text | Entity Type |\n| :--- | :--- |\n"
        for ent in entities[:15]:
            report += f"| {ent.get('text')} | {ent.get('label')} |\n"
        report += "\n"

    if sentiments:
        report += "## 6. Rhetorical Stance Analysis (DistilBERT)\n"
        report += "| Passage # | Source | Classification | Confidence | Stance Interpretation |\n| :--- | :--- | :--- | :--- | :--- |\n"
        for s in sentiments:
            report += f"| #{s.get('chunk_idx')} | {s.get('source')} (P.{s.get('page')}) | {s.get('label')} | {s.get('score'):.4f} | {s.get('tone')} |\n"
        report += "\n"

    report += "---\n*Synthesized by Agentic RAG-Based AI Research Assistant.*"
    return report


if __name__ == "__main__":
    print("Testing src/graph.py with multi-agent workflow...")
    # Test conversational query
    res_greeting = run_research_workflow("Hello, introduce yourself")
    print("\n--- Greeting Test Trace ---")
    for t in res_greeting["execution_trace"]:
        print(f"[{t['timestamp']}] {t['agent']}: {t['action']}")
    print("\nFinal Greeting Output:\n", res_greeting["final_answer"][:200])

    # Test technical question
    res_tech = run_research_workflow("What is the Self-Attention mechanism in Transformers?")
    print("\n--- Technical Test Trace ---")
    for t in res_tech["execution_trace"]:
        print(f"[{t['timestamp']}] {t['agent']}: {t['action']}")
    print("\nFinal Answer Sample:\n", res_tech["final_answer"][:300])

    print("\nsrc/graph.py executed successfully!")
