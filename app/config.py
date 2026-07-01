import os
from dataclasses import dataclass
from typing import Any, Optional

@dataclass(frozen=True)
class GraphConfiguration:
    """Configuration class for the RAG agent graph."""

    # LLM Settings
    model_name: str = "unsloth/gemma-4-E4B-it-GGUF"
    model_base_url: str = "http://localhost:8080"
    model_api_key: str = "i_dont_know"

    # Embedding and Document Settings
    embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    doc_splits_path: str = "models/doc_splits.pkl"

    @classmethod
    def from_runnable_config(cls, config: Optional[dict[str, Any]] = None) -> "GraphConfiguration":
        """Extract configurations from a LangGraph RunnableConfig."""
        configurable = (config or {}).get("configurable", {})
        
        return cls(
            model_name=configurable.get(
                "model_name", 
                os.getenv("MODEL_NAME", cls.model_name)
            ),
            model_base_url=configurable.get(
                "model_base_url", 
                os.getenv("MODEL_BASE_URL", cls.model_base_url)
            ),
            model_api_key=configurable.get(
                "model_api_key", 
                os.getenv("MODEL_API_KEY", cls.model_api_key)
            ),
            embedding_model_name=configurable.get(
                "embedding_model_name", 
                os.getenv("EMBEDDING_MODEL_NAME", cls.embedding_model_name)
            ),
            doc_splits_path=configurable.get(
                "doc_splits_path", 
                os.getenv("DOC_SPLITS_PATH", cls.doc_splits_path)
            ),
        )
