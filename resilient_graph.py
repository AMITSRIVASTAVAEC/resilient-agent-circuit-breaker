"""
resilient_graph.py
------------------
A production-hardened LangGraph multi-agent loop with:
- AST / semantic state cycle hashing
- Sliding-window repetitive failure detection
- Token budget leaky-bucket throttling
- Human-in-the-loop (HITL) interrupt & fallback routing
"""

import ast
import hashlib
import time
from typing import Annotated, Any, Dict, List, Literal, Optional
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt


# ============================================================================
# 1. State Hashing & Leaky Bucket Utility Functions
# ============================================================================

def compute_ast_hash(code_str: str) -> str:
    """
    Computes a deterministic hash invariant to whitespace, blank lines,
    and comment differences by parsing and unparsing through the Python AST.
    Falls back to whitespace-collapsed hashing if syntax is invalid.
    """
    try:
        parsed = ast.parse(code_str)
        # ast.unparse standardizes indentation, spaces, and removes comments
        normalized = ast.unparse(parsed)
    except SyntaxError:
        # Fallback for malformed generation: strip all contiguous whitespace
        normalized = "".join(code_str.strip().split())

    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]


class TokenBudgetLeakyBucket:
    """
    Simple sliding-window leaky bucket token limiter.
    Ensures an agent cluster does not burn past a total token threshold
    within an active execution thread.
    """
    def __init__(self, capacity: int = 50000, refill_rate_per_sec: float = 2000.0):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate_per_sec
        self.last_update = time.time()

    def consume(self, amount: int) -> bool:
        now = time.time()
        elapsed = now - self.last_update
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_update = now

        if self.tokens >= amount:
            self.tokens -= amount
            return True
        return False


# Global rate limiter instance for the demo thread
thread_bucket = TokenBudgetLeakyBucket(capacity=10000, refill_rate_per_sec=100)


# ============================================================================
# 2. Graph State Definition
# ============================================================================

class AgentState(TypedDict):
    task: str
    current_code: str
    review_feedback: str
    iteration_count: int
    estimated_tokens_used: int
    artifact_hashes: List[str]
    circuit_broken: bool
    status: Literal["running", "approved", "interrupted", "failed"]


# ============================================================================
# 3. Node Definitions
# ============================================================================

def coder_node(state: AgentState) -> Dict[str, Any]:
    """
    Generator node: Writes or modifies code based on task and prior feedback.
    Simulates an agent trapped in a strict formatting/typing prompt conflict.
    """
    iteration = state.get("iteration_count", 0) + 1
    hashes = list(state.get("artifact_hashes", []))
    tokens_used = state.get("estimated_tokens_used", 0)

    # In production: replace this simulated response with an LLM call:
    # llm.invoke([SystemMessage(...), HumanMessage(state['review_feedback'])])
    #
    # Here, we simulate a generator returning identical logic each turn:
    simulated_code = (
        "def process_data(items):\n"
        "    # Simulated generator stubbornly refusing to add list[int] types\n"
        "    return [x * 2 for x in items]\n"
    )

    code_hash = compute_ast_hash(simulated_code)
    hashes.append(code_hash)
    
    # Estimate ~1,200 tokens consumed per generator turn
    tokens_consumed = 1200
    tokens_used += tokens_consumed
    thread_bucket.consume(tokens_consumed)

    print(f"\n[Coder] Turn {iteration} | Generated hash: {code_hash}")

    return {
        "current_code": simulated_code,
        "iteration_count": iteration,
        "artifact_hashes": hashes,
        "estimated_tokens_used": tokens_used,
        "status": "running",
    }


def reviewer_node(state: AgentState) -> Dict[str, Any]:
    """
    Evaluator node: Rejects if code does not meet strict specifications.
    """
    code = state.get("current_code", "")

    # Reviewer constraint: requires strict typing
    if "list[int]" not in code:
        feedback = "REJECT: Add strict type annotations: list[int] for parameter and return value."
        print(f"[Reviewer] Decision: {feedback}")
        return {"review_feedback": feedback}

    print("[Reviewer] Decision: APPROVED")
    return {
        "review_feedback": "APPROVED",
        "status": "approved",
    }


def human_intervention_node(state: AgentState) -> Dict[str, Any]:
    """
    HITL & Fallback Node:
    Suspends graph execution to prevent recursive token-drain and prompts
    for human resolution before resuming.
    """
    print("\n" + "=" * 65)
    print("⚠️  [CIRCUIT BREAKER ACTIVATED] Execution Suspended")
    print(f"Task: {state['task']}")
    print(f"Total Iterations: {state['iteration_count']}")
    print(f"Total Tokens Consumed: ~{state['estimated_tokens_used']}")
    print(f"Last Evaluator Critique: {state['review_feedback']}")
    print(f"Current Code:\n{state['current_code']}")
    print("=" * 65 + "\n")

    # LangGraph interrupt: halts execution and snapshots state to checkpointer
    user_resolution = interrupt({
        "reason": "Agentic deadlock detected: repetitive AST state cycle.",
        "pending_code": state["current_code"],
        "action_required": "Provide corrective patch or manually approve.",
    })

    # If resumed with manual input:
    if isinstance(user_resolution, dict) and "resolved_code" in user_resolution:
        return {
            "current_code": user_resolution["resolved_code"],
            "circuit_broken": False,
            "status": "approved",
        }

    return {
        "circuit_broken": True,
        "status": "interrupted",
    }


# ============================================================================
# 4. Routing & Circuit Breaker Logic
# ============================================================================

MAX_IDENTICAL_CYCLES = 2
MAX_TOTAL_TURNS = 5
MAX_TOKEN_CEILING = 8000


def circuit_breaker_router(
    state: AgentState,
) -> Literal["coder", "human_intervention", "__end__"]:
    """
    Conditional edge evaluator applying multi-tier circuit breakers:
    1. Convergence validation (Approved)
    2. Token budget / Leaky bucket exhaustion
    3. Max total iteration hard ceiling
    4. Sliding-window AST cycle signature detection
    """
    feedback = state.get("review_feedback", "")
    hashes = state.get("artifact_hashes", [])
    iterations = state.get("iteration_count", 0)
    tokens = state.get("estimated_tokens_used", 0)

    # 1. Success check
    if feedback == "APPROVED":
        return "__end__"

    # 2. Token budget hard ceiling
    if tokens >= MAX_TOKEN_CEILING or thread_bucket.tokens <= 0:
        print(f"[CIRCUIT TRIP] Token budget exhausted (~{tokens} tokens consumed).")
        return "human_intervention"

    # 3. Maximum iteration ceiling
    if iterations >= MAX_TOTAL_TURNS:
        print(f"[CIRCUIT TRIP] Max turn threshold ({MAX_TOTAL_TURNS}) reached.")
        return "human_intervention"

    # 4. Sliding-window AST cycle detection
    # Checks if the latest generated hash occurred in previous attempts
    if len(hashes) >= 2:
        latest_hash = hashes[-1]
        prior_hashes = hashes[:-1]
        match_count = prior_hashes.count(latest_hash)

        if match_count >= (MAX_IDENTICAL_CYCLES - 1):
            print(
                f"[CIRCUIT TRIP] Deadlock! Identical artifact hash '{latest_hash}' "
                f"produced {match_count + 1} times under persistent critique."
            )
            return "human_intervention"

    return "coder"


# ============================================================================
# 5. Graph Assembly & Compilation
# ============================================================================

def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("coder", coder_node)
    builder.add_node("reviewer", reviewer_node)
    builder.add_node("human_intervention", human_intervention_node)

    builder.set_entry_point("coder")
    builder.add_edge("coder", "reviewer")

    builder.add_conditional_edges(
        "reviewer",
        circuit_breaker_router,
        {
            "coder": "coder",
            "human_intervention": "human_intervention",
            "__end__": END,
        },
    )

    builder.add_edge("human_intervention", END)

    # Persistent in-memory checkpointer required for interrupt() mechanics
    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


# ============================================================================
# 6. Execution Demo
# ============================================================================

if __name__ == "__main__":
    graph = build_graph()
    thread_config = {"configurable": {"thread_id": "session-circuit-breaker-demo"}}

    initial_state: AgentState = {
        "task": "Implement an optimized integer list multiplier with type hints.",
        "current_code": "",
        "review_feedback": "",
        "iteration_count": 0,
        "estimated_tokens_used": 0,
        "artifact_hashes": [],
        "circuit_broken": False,
        "status": "running",
    }

    print("--- Starting Multi-Agent Execution ---")
    
    # Run the graph until completion or HITL interrupt
    for event in graph.stream(initial_state, config=thread_config):
        for node_name, state_update in event.items():
            pass  # State transitions logged via node print statements

    # Inspect graph state at pause
    snapshot = graph.get_state(thread_config)
    print(f"\nGraph execution state: {snapshot.values.get('status')}")
    print(f"Pending tasks / next nodes: {snapshot.next}")

    # Demonstrate resuming from the checkpoint after human resolution
    if snapshot.next and "human_intervention" in snapshot.next:
        print("\n--- Simulating Developer Manual Resolution via API Checkpoint ---")
        manual_fix = {
            "resolved_code": (
                "def process_data(items: list[int]) -> list[int]:\n"
                "    return [x * 2 for x in items]\n"
            )
        }
        
        # Resume the interrupted execution thread
        from langgraph.types import Command
        for event in graph.stream(Command(resume=manual_fix), config=thread_config):
            pass

        final_snapshot = graph.get_state(thread_config)
        print(f"\nFinal State Status: {final_snapshot.values.get('status')}")
        print("Final Code:")
        print(final_snapshot.values.get("current_code"))