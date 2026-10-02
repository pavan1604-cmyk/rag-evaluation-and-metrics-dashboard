from sentence_transformers import SentenceTransformer
from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.run_config import RunConfig


class SentenceTransformerRagasEmbeddings(BaseRagasEmbeddings):
    """
    RAGAS embedding adapter using sentence-transformers.
    """

    def __init__(
        self,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    ):
        super().__init__()

        self.model = SentenceTransformer(model_name)
        self.set_run_config(RunConfig())

    async def aembed_query(self, text):
        return self.model.encode(
            text,
            convert_to_numpy=True
        ).tolist()

    async def aembed_documents(self, texts):
        return self.model.encode(
            texts,
            convert_to_numpy=True
        ).tolist()

    def embed_query(self, text):
        return self.model.encode(
            text,
            convert_to_numpy=True
        ).tolist()

    def embed_documents(self, texts):
        return self.model.encode(
            texts,
            convert_to_numpy=True
        ).tolist()
