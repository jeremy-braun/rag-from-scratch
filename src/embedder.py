from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import os

# Pfad wo ChromaDB die Daten persistent speichert
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_db")

def get_embedding_model():
    """
    Laedt das Embedding-Modell.
    Beim ersten Aufruf wird all-MiniLM-L6-v2 heruntergeladen (~90 MB).
    Danach wird es aus dem Cache geladen.
    """
    return SentenceTransformer("all-MiniLM-L6-v2")

def get_chroma_collection(collection_name: str = "rag_documents"):
    """
    Erstellt oder oeffnet eine ChromaDB Collection.
    PersistentClient speichert die Daten auf Disk — bleiben also
    zwischen Sessions erhalten.
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}  # Cosine Similarity als Distanzmetrik
    )
    return collection

def embed_and_store(chunks: list[str], collection_name: str = "rag_documents"):
    """
    Nimmt eine Liste von Text-Chunks, erstellt Embeddings
    und speichert alles in ChromaDB.
    
    chunks: Output von chunking.py
    """
    model = get_embedding_model()
    collection = get_chroma_collection(collection_name)

    # Batch-Processing: alle Chunks auf einmal durch das Modell
    # statt einer Schleife mit einzelnen Aufrufen — deutlich schneller
    print(f"Erstelle Embeddings fuer {len(chunks)} Chunks...")
    embeddings = model.encode(chunks, show_progress_bar=True)

    # IDs generieren — ChromaDB braucht eindeutige IDs pro Eintrag
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    # Alles in ChromaDB speichern
    collection.add(
        documents=chunks,           # Originaltext
        embeddings=embeddings.tolist(),  # Vektoren als Python-Liste
        ids=ids
    )

    print(f"{len(chunks)} Chunks gespeichert in Collection '{collection_name}'.")
    return collection

if __name__ == "__main__":
    # Test: chunking.py importieren und direkt durchlaufen lassen
    import sys
    sys.path.append(os.path.dirname(__file__))
    from chunking import chunk_text_fixed

    sample_text = """
    Katzen sind haeusliche Tiere die gut in Wohnungen leben koennen.
    Sie schlafen durchschnittlich 16 Stunden am Tag.
    Hunde benoetigen mehr Auslauf und eignen sich fuer Haeuser mit Garten.
    Kaninchen sind ruhige Tiere und brauchen wenig Platz.
    Voegel wie Wellensittiche sind pflegeleicht und sprechen manchmal nach.
    Hamster sind nachtaktiv und benoetigen ein grosses Gehege.
    """

    chunks = chunk_text_fixed(sample_text, chunk_size=20, overlap=4)
    print(f"Chunks erstellt: {len(chunks)}")
    for i, c in enumerate(chunks):
        print(f"  Chunk {i+1}: {c}")

    collection = embed_and_store(chunks)
    print(f"\nCollection enthaelt jetzt {collection.count()} Eintraege.")