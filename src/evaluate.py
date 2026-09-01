import ollama
import os
import sys
sys.path.append(os.path.dirname(__file__))

from rag_pipeline import query_pipeline
from retriever import retrieve


def evaluate_groundedness(query: str, n_results: int = 5, distance_threshold: float = 0.8) -> dict:
    """
    Führt eine RAG-Query aus und lässt llama3.1:8b als Judge
    den Groundedness-Score der Antwort bewerten.
    
    Gibt dict zurück mit:
    - query
    - answer
    - chunks
    - groundedness_score (0.0 - 1.0)
    - reasoning
    """
    # RAG-Antwort generieren
    answer = query_pipeline(query, n_results=n_results, distance_threshold=distance_threshold)
    
    # Chunks die übergeben wurden nochmal holen für den Judge
    chunks = retrieve(query, n_results=n_results, distance_threshold=distance_threshold)
    context = "\n\n".join([f"Chunk {i+1}:\n{c['text']}" for i, c in enumerate(chunks)])

    # Judge-Prompt
    judge_prompt = f"""Du bist ein Evaluierungssystem für RAG-Pipelines. Deine Aufgabe ist es zu prüfen ob eine gegebene Antwort ausschließlich auf dem gegebenen Kontext basiert.

Kontext:
{context}

Generierte Antwort:
{answer}

Bewerte die Antwort nach folgendem Kriterium:
- Score 1.0: Jede Aussage in der Antwort ist direkt durch den Kontext belegbar.
- Score 0.5: Einige Aussagen sind belegbar, andere gehen über den Kontext hinaus.
- Score 0.0: Die Antwort enthält überwiegend Informationen die nicht im Kontext stehen.

Antworte ausschließlich in folgendem Format:
SCORE: [Zahl zwischen 0.0 und 1.0]
BEGRUENDUNG: [Eine kurze Erklärung welche Aussagen belegbar sind und welche nicht]"""

    # Judge-Call
    response = ollama.chat(
        model="llama3.1:8b",
        messages=[{"role": "user", "content": judge_prompt}]
    )

    raw = response["message"]["content"]

    #Score aus Antwort extrahieren
    import re
    
    
    score = None
    reasoning = ""
    for line in raw.split("\n"):
        if "score" in line.lower():
            match = re.search(r"(\d+[.,]\d*)", line)
            if match:
                score = float(match.group(1).replace(",", "."))
                if score > 1.0:
                    score = score / 10
        if "begr" in line.lower() or "reason" in line.lower():
            reasoning = line.split(":", 1)[-1].strip()



    return {

        "query": query,
        "answer": answer,
        "chunks_used": len(chunks),
        "groundedness_score": score,
        "reasoning": reasoning
    }


if __name__ == "__main__":
    testfragen = [
        "What design principles should AI systems follow when co-creating stories with children?",
        "How do children interact differently with AI compared to human partners in storytelling?",
        "What are the limitations of current child-AI co-creative systems?"
    ]

    for frage in testfragen:
        print(f"\nFrage: {frage}")
        result = evaluate_groundedness(frage)
        print(f"Antwort: {result['answer']}")
        print(f"Chunks verwendet: {result['chunks_used']}")
        print(f"Groundedness Score: {result['groundedness_score']}")
        print(f"Begründung: {result['reasoning']}")
        print("-" * 60)