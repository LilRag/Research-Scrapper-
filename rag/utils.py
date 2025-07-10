import chromadb
def clear_database():
    client = chromadb.PersistentClient(path="./chroma_db")
    try:
        client.delete_collection("research_chunks")
    except:
        pass

    # Recreate empty collection
    client.get_or_create_collection(name="research_chunks")