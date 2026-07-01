import logging
import os
import uuid
import warnings
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.graph import START

from dotenv import load_dotenv

from app.agent import workflow
from app.cli import registry
from app.db import get_checkpointer
from app.utils.state import RagState

# Suppress progress bars and verbose warnings
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = "1"
warnings.filterwarnings("ignore")
logging.getLogger("pydantic").setLevel(logging.ERROR)
logging.getLogger("langchain").setLevel(logging.ERROR)


# =====================================================================
# RAG EXECUTION RUNNER
# =====================================================================


def run_agentic_rag(
    content: str, 
    graph: Any,
    thread_id: str,
    checkpoint_id: str | None = None
) -> str:
    """Executes the agent graph for a query, supporting time travel checkpoints."""
    config = {"configurable": {"thread_id": thread_id}}

    if checkpoint_id:
        # Time travel: fetch the state config from history matching checkpoint_id
        history = list(graph.get_state_history(config))
        target_state = next(
            (state for state in history if state.config["configurable"]["checkpoint_id"] == checkpoint_id),
            None
        )

        if not target_state:
            raise ValueError(f"Checkpoint ID {checkpoint_id} not found in history.")

        # Fork state history with the target config and run from START
        fork_config = graph.update_state(
            target_state.config,
            {"messages": [HumanMessage(content=content)]},
            as_node=START
        )
        result = graph.invoke(None, config=fork_config)
    else:
        # Normal query execution: appends messages to latest checkpoint
        result = graph.invoke(
            {"messages": [HumanMessage(content=content)]},
            config=config
        )

    parsed_result = RagState(**result)
    final_message = parsed_result.messages[-1]
    
    print("\n[Agent]:")
    print(final_message.content)
    print(f"--- (Rewrite count: {parsed_result.rewrite_count}) ---\n")

    # Fetch the newest checkpoint ID that was just created
    history = list(graph.get_state_history(config))
    return history[0].config["configurable"]["checkpoint_id"]


# =====================================================================
# MAIN ENTRYPOINT
# =====================================================================

def main():
    # Load environment variables
    load_dotenv()

    print("=" * 60)
    print("Welcome to Open Politics Agentic RAG CLI")
    print("=" * 60)

    # Get local SQLite checkpointer (db/checkpoints.db)
    checkpointer = get_checkpointer()
    
    # Compile graph specifically with the local SQLite checkpointer
    graph = workflow.compile(checkpointer=checkpointer)

    # Let user select session
    session_id = input("Enter a session ID (or press Enter to create a new session): ").strip()
    if not session_id:
        session_id = f"session_{str(uuid.uuid7())}"
        print(f"Created new session with ID: {session_id}")

    context = {
        "graph": graph,
        "session_id": session_id,
        "active_checkpoint": None,
        "should_exit": False
    }

    print("-" * 60)
    print("Type your question to query the agent.")
    print("Type /help to see command options, or /exit to exit.")
    print("-" * 60)

    while not context["should_exit"]:
        try:
            user_input = input(">> ").strip()
            if not user_input:
                continue

            # Check if this is a command and handle it
            if user_input.startswith("/"):
                if registry.handle(user_input, context):
                    continue
                else:
                    print("❌ Unknown command. Type /help to see available commands.")
                    continue

            # Otherwise, perform standard agent execution
            context["active_checkpoint"] = run_agentic_rag(
                content=user_input,
                graph=graph,
                thread_id=context["session_id"],
                checkpoint_id=context["active_checkpoint"]
            )
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()