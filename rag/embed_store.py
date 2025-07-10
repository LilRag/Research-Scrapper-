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

def embed_and_store(chunks):
    for i, chunk in enumerate(chunks):
        text = chunk.page_content
        emb = get_local_embeddings(text)
        collection.add(
            documents=[text],
            embeddings=[emb],
            ids=[f"chunk_{i}"]
        )