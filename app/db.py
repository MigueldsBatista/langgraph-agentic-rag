import os
import pickle
import sqlite3
from functools import lru_cache

from langchain.chat_models import BaseChatModel
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import SecretStr

from app.config import GraphConfiguration


@lru_cache(maxsize=1)
def get_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    """Load HuggingFace Embeddings, caching the model in memory."""
    return HuggingFaceEmbeddings(
        model_name=model_name, 
        cache_folder="./models",
    )

@lru_cache(maxsize=1)
def get_vector_store(doc_splits_path: str, embedding_model_name: str) -> InMemoryVectorStore:
    """Load doc_splits from pickle file and compile the InMemoryVectorStore."""
    if not os.path.exists(doc_splits_path):
        raise FileNotFoundError(
            f"❌ Vector store file not found at: '{os.path.abspath(doc_splits_path)}'. "
            f"Please run 'uv run ingest' to ingest and parse documents first."
        )
    
    embeddings = get_embeddings(embedding_model_name)
    
    with open(doc_splits_path, "rb") as f:
        doc_splits = pickle.load(f)
        
    if not doc_splits:
        raise ValueError(f"❌ Loaded file '{doc_splits_path}' is empty. No documents were loaded.")
        
    print(f"✅ Success: Loaded {len(doc_splits)} document splits from '{doc_splits_path}' into InMemoryVectorStore.")
    
    return InMemoryVectorStore.from_documents(
        documents=doc_splits,
        embedding=embeddings,
    )

def get_model(config: GraphConfiguration) -> BaseChatModel:
    """Instantiate the chat model dynamically based on GraphConfiguration."""
    return ChatOpenAI(
        api_key=SecretStr(config.model_api_key),
        base_url=config.model_base_url,
        model=config.model_name,
    )

@lru_cache(maxsize=1)
def get_checkpointer() -> BaseCheckpointSaver:
    """Initialize and return a SQLite checkpointer stored inside a dedicated db/ directory."""
    db_dir = "db"
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, "checkpoints.db")
    
    # Establish connection with SQLite
    conn = sqlite3.connect(db_path, check_same_thread=False)
    return SqliteSaver(conn)
