from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    def __init__(self):
        """
        Load the cross-encoder reranking model.
        """
        self.model = CrossEncoder(MODEL_NAME)

    def rerank(self, query, documents, top_k=5):
        """
        Rerank retrieved documents using a cross-encoder.

        Args:
            query (str): User query.
            documents (list[dict]): Retrieved documents.
            top_k (int): Number of final documents.

        Returns:
            list[dict]: Reranked documents.
        """

        if not documents:
            return []

        pairs = [
            [query, document["text"]]
            for document in documents
        ]

        scores = self.model.predict(pairs)

        reranked = []

        for document, score in zip(documents, scores):
            result = document.copy()
            result["rerank_score"] = float(score)
            reranked.append(result)

        reranked.sort(
            key=lambda item: item["rerank_score"],
            reverse=True
        )

        for rank, result in enumerate(reranked[:top_k], start=1):
            result["rerank_rank"] = rank

        return reranked[:top_k]