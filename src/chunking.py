def chunk_text_fixed(text: str, chunk_size: int = 256, overlap: int = 32) -> list[str]:
    """
    Teilt Text in Chunks fixer Größe auf (gemessen in Wörtern).
    
    chunk_size: Anzahl Wörter pro Chunk
    overlap:    Anzahl Wörter die zwischen zwei Chunks überlappen
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def chunk_text_sentences(text: str, sentences_per_chunk: int = 5, overlap: int = 1) -> list[str]:
    """s
    Teilt Text in Chunks auf Basis von Sätzen auf.
    
    sentences_per_chunk: Anzahl Sätze pro Chunk
    overlap:             Anzahl Sätze die zwischen zwei Chunks überlappen
    """
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s for s in sentences if s]
    chunks = []
    start = 0

    while start < len(sentences):
        end = start + sentences_per_chunk
        chunk = " ".join(sentences[start:end])
        chunks.append(chunk)
        start += sentences_per_chunk - overlap

    return chunks


if __name__ == "__main__":
    sample_text = """
    Retrieval-Augmented Generation ist ein Ansatz der Large Language Models mit externem Wissen verbindet.
    Das Modell ruft relevante Dokumente ab bevor es eine Antwort generiert.
    Dadurch wird Halluzination reduziert und die Antwortqualität bei domänenspezifischen Fragen verbessert.
    Der Retrieval-Schritt basiert auf Vektorähnlichkeit zwischen Query und gespeicherten Dokumenten.
    Jedes Dokument wird als Vektor im hochdimensionalen Raum repräsentiert.
    Ähnliche Bedeutungen erzeugen ähnliche Vektoren die nah beieinander liegen.
    """

    print("=== Fixed-Size Chunking ===")
    fixed_chunks = chunk_text_fixed(sample_text, chunk_size=20, overlap=5)
    for i, chunk in enumerate(fixed_chunks):
        print(f"Chunk {i+1}: {chunk}\n")

    print("=== Sentence-Based Chunking ===")
    sentence_chunks = chunk_text_sentences(sample_text, sentences_per_chunk=2, overlap=1)
    for i, chunk in enumerate(sentence_chunks):
        print(f"Chunk {i+1}: {chunk}\n")