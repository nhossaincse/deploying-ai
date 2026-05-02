"""Script to create ChromaDB embeddings for music reviews.

Run this script once to initialize the vector database:
    python -m assignment_chat.create_embeddings

The embeddings are stored in ./chroma_data/ and persisted to disk.
"""

import os
import json
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv(".env")
load_dotenv(".secrets")

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DOCUMENTS_PATH = os.path.join(SCRIPT_DIR, "..", "documents", "pitchfork_content.jsonl")
CHROMA_PATH = os.path.join(SCRIPT_DIR, "chroma_data")


def load_music_reviews():
    """Load music reviews from the JSONL file."""
    reviews = []
    with open(DOCUMENTS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                reviews.append(json.loads(line))
    return reviews


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Get embeddings using OpenAI API via course gateway."""
    client = OpenAI(
        base_url="https://k7uffyg03f.execute-api.us-east-1.amazonaws.com/prod/openai/v1",
        api_key="any",
        default_headers={"x-api-key": os.getenv("API_GATEWAY_KEY")}
    )
    
    response = client.embeddings.create(
        input=texts,
        model="text-embedding-3-small"
    )
    
    return [item.embedding for item in response.data]


def create_embeddings():
    """Create ChromaDB collection with music review embeddings."""
    print(f"Loading reviews from: {DOCUMENTS_PATH}")
    reviews = load_music_reviews()
    print(f"Loaded {len(reviews)} reviews")
    
    # Create persistent ChromaDB client
    print(f"Creating ChromaDB at: {CHROMA_PATH}")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    
    # Delete existing collection if it exists
    try:
        client.delete_collection(name="music_reviews")
        print("Deleted existing collection")
    except Exception:
        pass
    
    # Create new collection (without embedding function - we'll add embeddings manually)
    collection = client.create_collection(
        name="music_reviews",
        metadata={"description": "Pitchfork music album reviews"}
    )
    
    # Prepare data for insertion
    documents = []
    metadatas = []
    ids = []
    
    for review in reviews:
        documents.append(review["content"])
        metadatas.append({
            "title": review["title"],
            "artist": review["artist"],
            "score": review["score"],
            "reviewid": review["reviewid"]
        })
        ids.append(str(review["reviewid"]))
    
    # Generate embeddings using OpenAI API
    print("Generating embeddings...")
    embeddings = get_embeddings(documents)
    print(f"Generated {len(embeddings)} embeddings")
    
    # Add to collection with pre-computed embeddings
    print("Adding to ChromaDB collection...")
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
        embeddings=embeddings
    )
    
    print(f"Successfully added {len(documents)} reviews to ChromaDB")
    print(f"Collection count: {collection.count()}")
    print("Done!")


if __name__ == "__main__":
    create_embeddings()
