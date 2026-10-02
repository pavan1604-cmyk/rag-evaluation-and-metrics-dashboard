from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_embedding_model():
    """
    Load the sentence-transformer embedding model.
    """
    return SentenceTransformer(MODEL_NAME)


def create_embeddings(model, texts):
    """
    Convert text chunks into embedding vectors.
    """
    return model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True
    )
