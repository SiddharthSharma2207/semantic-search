import chromadb
import json
from config import Settings

settings=Settings()

def main():
    # 1. Initialize Chroma Client to save to disk
    persist_dir = settings.chroma_db_dir
    client = chromadb.PersistentClient(path=persist_dir)
    
    # 2. Create or get collection
    collection_name = settings.collection_name
    # Delete if exists to start fresh for this build script
    try:
        client.delete_collection(collection_name)
        print(f"Deleted existing collection '{collection_name}'")
    except ValueError:
        pass
        
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"} # Use cosine similarity
    )
    
    # 3. Load generated data
    with open("data/generated_faq_llm.json", "r") as f:
        data = json.load(f)
        
    print(f"Loaded {len(data)} documents to index.")
    
    # 4. Prepare data for insertion
    documents = []
    metadatas = []
    ids = []
    
    for i, item in enumerate(data):
        documents.append(item["question"])
        metadatas.append({
            "query_id": item["query_id"],
            "topic": item["topic"],
        })
        ids.append(f"faq_{i}")
        
    # 5. Add to collection
    # Upsert is safer, though we just deleted the collection
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    
    print(f"Successfully indexed {len(documents)} documents into '{persist_dir}'.")
    print(f"Collection count: {collection.count()}")

if __name__ == "__main__":
    main()
