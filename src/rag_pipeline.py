import os
import sys
sys.path.append(os.path.dirname(__file__))

from pypdf import PdfReader
import ollama

from chunking import chunk_text_fixed
from embedder import embed_and_store, get_chroma_collection
from retriever import retrieve


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Liest ein PDF und gibt den gesamten Text als String zurück.
    Bereinigt Zeilenumbrüche die mitten im Satz auftreten.
    """
    reader = PdfReader(pdf_path)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            text = text.replace("-\n", "").replace("\n", " ")
            pages.append(text)
    return " ".join(pages)


def index_pdfs(pdf_folder: str, chunk_size: int = 256, overlap: int = 32):
    """
    Liest alle PDFs in einem Ordner, chunked und embedded sie.
    Nur einmal ausführen — danach sind die Daten in ChromaDB.
    """
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

    if not pdf_files:
        print("Keine PDFs gefunden.")
        return

    all_chunks = []
    for pdf_file in pdf_files:
        path = os.path.join(pdf_folder, pdf_file)
        print(f"Lese: {pdf_file}")
        text = extract_text_from_pdf(path)
        chunks = chunk_text_fixed(text, chunk_size=chunk_size, overlap=overlap)
        print(f"  → {len(chunks)} Chunks")
        all_chunks.extend(chunks)

    print(f"\nGesamt: {len(all_chunks)} Chunks. Starte Indexierung...")
    embed_and_store(all_chunks)
    print("Indexierung abgeschlossen.")


def query_pipeline(query: str, n_results: int = 5, distance_threshold: float = 0.8) -> str:
    """
    Nimmt eine Query, retrievet relevante Chunks und lässt
    llama3.1:8b eine Antwort generieren.
    """
    chunks = retrieve(query, n_results=n_results, distance_threshold=distance_threshold)

    if not chunks:
        return "Keine relevanten Informationen in der Wissensbasis gefunden."

    context = "\n\n".join([f"Quelle {i+1}:\n{c['text']}" for i, c in enumerate(chunks)])

    prompt = f"""Du bist ein Assistent der ausschließlich auf Basis des gegebenen Kontexts antwortet.
Wenn die Antwort nicht im Kontext steht, sagst du explizit: "Diese Information ist nicht in der Wissensbasis vorhanden."
Füge kein Wissen aus deinem Training hinzu.

Kontext:
{context}

Frage: {query}

Antwort:"""

    response = ollama.chat(
        model="llama3.1:8b",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]


if __name__ == "__main__":
    PDF_FOLDER = os.path.join(os.path.dirname(__file__), "..", "examples", "papers")

    modus = "query"  # "index" beim ersten Mal, danach "query"

    if modus == "index":
        index_pdfs(PDF_FOLDER)

    elif modus == "query":
        frage = "What design principles should AI systems follow when co-creating stories with children?"
        print(f"Frage: {frage}\n")
        antwort = query_pipeline(frage)
        print(f"Antwort:\n{antwort}")
