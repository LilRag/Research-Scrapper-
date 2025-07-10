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

def query_documents(user_query , top_k = 5):
    """Query the collection for similar documents based on the query text."""
    query = get_local_embeddings(user_query)
    results = collection.query(
        query_embeddings=[query],
        n_results = top_k,
        include=["documents", "distances"])
    return results 
