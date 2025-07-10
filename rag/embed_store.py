import chromadb
import requests

client = chromadb.PersistentClient(path="./chroma_db")



def get_local_embeddings(text):
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={
            "model": "nomic-embed-text:latest",
            "prompt": text
        }
    )
    return response.json()["embedding"]

def embed_and_store(chunks, source_name):
    collection = client.get_or_create_collection(name="research_chunks")
    for i, chunk in enumerate(chunks):
        text = chunk.page_content
        emb = get_local_embeddings(text)
        collection.add(
            documents=[text],
            embeddings=[emb],
            ids=[f"{source_name}_chunk_{i}"],
            metadatas=[{"source":source_name}]
        )