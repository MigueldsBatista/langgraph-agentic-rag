from dataclasses import dataclass
from typing import Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field

@dataclass
class RagState:
    messages: Annotated[list[AnyMessage], add_messages]
    rewrite_count: int = 0

class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""
    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )
