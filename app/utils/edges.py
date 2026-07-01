from langgraph.graph import END
from app.utils.state import RagState

def retrieve_if_tool_call_or_end(state: RagState):
    last_msg = state.messages[-1]
    return "retrieve" if getattr(last_msg, "tool_calls", None) else END
