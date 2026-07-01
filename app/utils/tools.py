from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from app.config import GraphConfiguration
from app.db import get_vector_store

@tool
def retrieve_blog_posts(query: str, config: RunnableConfig) -> str:
    """Search and return information about Lilian Weng blog posts."""
    # Build GraphConfiguration from the runnable config
    graph_config = GraphConfiguration.from_runnable_config(config)
    
    # Retrieve the cached vector store instance
    vectorstore = get_vector_store(
        doc_splits_path=graph_config.doc_splits_path,
        embedding_model_name=graph_config.embedding_model_name
    )
    print("TOOL CALL DETECTED - Retrieved vector store instance:", vectorstore)
    # Query vector store
    retriever = vectorstore.as_retriever()
    retrieved_docs = retriever.invoke(query)
    
    return "\n\n".join([doc.page_content for doc in retrieved_docs])
