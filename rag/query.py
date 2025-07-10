import chromadb
import requests

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="research_chunks")

def get_local_embeddings(text):
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={
            "model": "nomic-embed-text:latest",
            "prompt": text
        }
    )
    return response.json()["embedding"]

def get_all_sources():
    """Return a list of unique source names (filenames) from metadata."""
    results = collection.get(include=["metadatas"])
    metadatas = results.get("metadatas", [])
    sources = [meta.get("source") for meta in metadatas if isinstance(meta, dict) and meta.get("source")]
    return sorted(set(sources))

def query_documents(user_query, sources=None, top_k=5):
    """Query the collection for similar documents based on the query text."""
    query_embedding = get_local_embeddings(user_query)

    query_params = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
        "include": ["documents", "distances", "metadatas"]
    }

    if sources:
        query_params["where"] = {"source": {"$in": sources}}

    results = collection.query(**query_params)
    return results
