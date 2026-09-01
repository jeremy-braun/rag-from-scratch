from sentence_transformers import SentenceTransformer
import chromadb
import os
import sys

sys.path.append(os.path.dirname(__file__))
from embedder import get_embedding_model, get_chroma_collection

def retrieve(query: str, n_results: int = 3, distance_threshold: float = 0.5) -> list[dict]:
    model = get_embedding_model()
    collection = get_chroma_collection()

    query_vector = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results,
        include=["documents", "distances"]
    )


    chunks = []
    documents = results["documents"][0]
    distances = results["distances"][0]

    for doc, dist in zip(documents, distances):
        if dist <= distance_threshold:
            chunks.append({
                "text": doc,
                "distance": round(dist, 4)
            })

    return chunks


if __name__ == "__main__":
    query = "Welche Tiere eignen sich fuer eine Wohnung?"
    print(f"Query: {query}\n")

    results = retrieve(query, n_results=3, distance_threshold=0.5)

    if not results:
        print("Keine relevanten Chunks gefunden.")
    else:
        for i, r in enumerate(results):
            print(f"Chunk {i+1} (Distanz: {r['distance']}):")
            print(f"  {r['text']}\n")