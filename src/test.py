

import sys; sys.path.append('src')
from embedder import get_embedding_model, get_chroma_collection
model = get_embedding_model()
collection = get_chroma_collection()
results = collection.query(
    query_embeddings=[model.encode('3.42 creative blocks children').tolist()],
    n_results=5,
    include=['documents', 'distances']
)
for doc, dist in zip(results['documents'][0], results['distances'][0]):
    print(f'{dist:.4f}: {doc[:150]}')
