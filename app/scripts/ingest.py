import os
import pickle
import bs4
import requests
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

URLS = [
    "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
    "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/"
]

def load_web_page(url: str, bs_kwargs: dict | None = None) -> list[Document]:
    print(f"Fetching: {url}")
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    soup = bs4.BeautifulSoup(response.text, "html.parser", **(bs_kwargs or {}))
    return [Document(page_content=soup.get_text(), metadata={"source": url})]

def main():
    print("Starting ingestion...")
    docs_list = []
    for url in URLS:
        try:
            docs = load_web_page(url)
            docs_list.extend(docs)
        except Exception as e:
            print(f"Error loading {url}: {e}")
            
    if not docs_list:
        print("No documents loaded. Ingestion failed.")
        return

    print(f"Loaded {len(docs_list)} documents. Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=100,
        chunk_overlap=50,
    )
    doc_splits = text_splitter.split_documents(docs_list)
    print(f"Created {len(doc_splits)} document splits.")

    os.makedirs("models", exist_ok=True)
    output_path = os.path.join("models", "doc_splits.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(doc_splits, f)
        
    print(f"Saved chunked documents to {output_path}")

if __name__ == "__main__":
    main()
