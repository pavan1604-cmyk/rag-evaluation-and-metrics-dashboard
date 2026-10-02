import faiss
import numpy as np


class FAISSRetriever:
    def __init__(self, embeddings, texts):
        """
        Create a FAISS retriever.

        Args:
            embeddings: Numpy array containing document embeddings.
            texts: Original text chunks.
        """
        self.texts = texts

        embeddings = np.asarray(embeddings).astype("float32")

        self.index = faiss.IndexFlatIP(embeddings.shape[1])

        faiss.normalize_L2(embeddings)

        self.index.add(embeddings)

    def search(self, query_embedding, top_k=5):
        """
        Search for the most similar text chunks.

        Args:
            query_embedding: Embedding vector for the query.
            top_k: Number of results to return.

        Returns:
            list[dict]: Retrieved chunks with similarity scores.
        """
        query_embedding = np.asarray(query_embedding).astype("float32")

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, top_k)

        results = []

        for score, index in zip(scores[0], indices[0]):
            if index == -1:
                continue

            results.append({
                "text": self.texts[index],
                "score": float(score),
                "rank": len(results) + 1
            })

        return results