from src.document_loader import load_pdf
from src.chunker import chunk_text
from src.embeddings import load_embedding_model, create_embeddings
from src.retriever import FAISSRetriever
from src.reranker import CrossEncoderReranker
from src.generator import RAGGenerator


class RAGPipeline:
    def __init__(
        self,
        pdf_path,
        chunk_size=800,
        overlap=150,
        candidate_k=8,
        top_k=5
    ):
        """
        Initialize the complete RAG pipeline.
        """

        self.pdf_path = pdf_path
        self.candidate_k = candidate_k
        self.top_k = top_k

        print("Loading document...")

        text = load_pdf(pdf_path)

        print("Chunking document...")

        self.chunks = chunk_text(
            text,
            chunk_size=chunk_size,
            overlap=overlap
        )

        print(f"Total chunks: {len(self.chunks)}")

        print("Loading embedding model...")

        self.embedding_model = load_embedding_model()

        print("Creating document embeddings...")

        embeddings = create_embeddings(
            self.embedding_model,
            self.chunks
        )

        print("Building FAISS index...")

        self.retriever = FAISSRetriever(
            embeddings,
            self.chunks
        )

        print("Loading reranker...")

        self.reranker = CrossEncoderReranker()

        print("Loading generator...")

        self.generator = RAGGenerator()

        print("RAG pipeline ready.")

    def query(self, question):
        """
        Run a question through the complete RAG pipeline.

        Returns:
            dict containing retrieval results,
            reranked contexts, and generated answer.
        """

        print("\nRunning query...")

        query_embedding = self.embedding_model.encode(
            [question],
            convert_to_numpy=True
        )

        dense_results = self.retriever.search(
            query_embedding,
            top_k=self.candidate_k
        )

        reranked_results = self.reranker.rerank(
            question,
            dense_results,
            top_k=self.top_k
        )

        contexts = [
            result["text"]
            for result in reranked_results
        ]

        answer = self.generator.generate(
            question,
            contexts
        )

        return {
            "question": question,
            "dense_results": dense_results,
            "reranked_results": reranked_results,
            "contexts": contexts,
            "answer": answer
        }