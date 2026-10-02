import os

from groq import Groq
from dotenv import load_dotenv


load_dotenv()


class RAGGenerator:
    def __init__(self):
        """
        Initialize the Groq client.
        """
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not set in the .env file."
            )

        self.client = Groq(api_key=api_key)

        self.model = "openai/gpt-oss-120b"

    def generate(self, query, contexts):
        """
        Generate an answer using the retrieved contexts.

        Args:
            query (str): User question.
            contexts (list[str]): Retrieved context passages.

        Returns:
            str: Generated answer.
        """

        context_text = "\n\n".join(contexts)

        prompt = f"""
Answer the user's question using only the provided context.

If the answer cannot be found in the context, say:
"I don't have enough information in the provided context."

Context:
{context_text}

Question:
{query}

Answer:
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        return response.choices[0].message.content