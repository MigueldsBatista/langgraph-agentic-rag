import os
import pickle
from dotenv import load_dotenv
from app.config import GraphConfiguration
from app.db import get_vector_store

def main():
    print("Running vector store verification test...")
    # Load configuration
    load_dotenv()
    config = GraphConfiguration.from_runnable_config()
    
    print(f"Loading vector store from: {config.doc_splits_path}")
    print(f"Using embedding model: {config.embedding_model_name}")
    
    try:
        # Fetch the vector store
        vectorstore = get_vector_store(
            doc_splits_path=config.doc_splits_path,
            embedding_model_name=config.embedding_model_name
        )
        
        # Run a test query
        query = "reward hacking"
        print(f"Executing test query: '{query}'...")
        docs = vectorstore.similarity_search(query, k=2)
        
        print("\n--- Query Results ---")
        for i, doc in enumerate(docs):
            print(f"[{i+1}] Source: {doc.metadata.get('source')}")
            print(f"    Snippet: {doc.page_content[:150].strip()}...\n")
            
        print("✅ Verification passed: Vector store loaded and returned similarity search results.")
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")

if __name__ == "__main__":
    main()
