from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from app.config import GraphConfiguration
from app.utils.edges import retrieve_if_tool_call_or_end
from app.utils.nodes import (
    generate_answer,
    generate_query_or_respond,
    grade_documents,
    rewrite_question,
)
from app.utils.state import RagState
from app.utils.tools import retrieve_blog_posts

# 1. Define the Graph Workflow and bind its Configuration Schema
workflow = (
    StateGraph(RagState, config_schema=GraphConfiguration)
    .add_node("generate_query_or_respond", generate_query_or_respond)
    .add_node("retrieve", ToolNode([retrieve_blog_posts]))
    .add_node("rewrite_question", rewrite_question)
    .add_node("generate_answer", generate_answer)
    .add_edge(START, "generate_query_or_respond")
    .add_conditional_edges("generate_query_or_respond", retrieve_if_tool_call_or_end)
    .add_conditional_edges("retrieve", grade_documents)
    .add_edge("generate_answer", END)
    .add_edge("rewrite_question", "generate_query_or_respond")
)

# 2. Compile a default graph (without checkpointer, to be overridden by local runners or cloud platforms)
graph = workflow.compile()