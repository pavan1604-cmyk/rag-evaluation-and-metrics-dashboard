import os
import time
import threading

import litellm
from dotenv import load_dotenv
from langchain_core.outputs import Generation, LLMResult
from ragas.llms.base import BaseRagasLLM
from ragas.run_config import RunConfig


load_dotenv()


class GroqRagasLLM(BaseRagasLLM):
    """
    RAGAS LLM adapter using LiteLLM with Groq.

    Groq currently limits the number of generated
    choices to n=1 and the available model has a
    token-per-minute limit.

    This adapter therefore:
    - always sends n=1
    - serializes API requests
    - waits between requests to reduce rate-limit errors
    """

    def __init__(
        self,
        model="groq/openai/gpt-oss-120b",
        request_interval=25
    ):
        super().__init__(
            run_config=RunConfig(
                max_workers=1
            )
        )

        self.model = model
        self.request_interval = request_interval

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not set in the .env file."
            )

        self.api_key = api_key

        self._last_request_time = 0.0
        self._request_lock = threading.Lock()

    def _wait_for_rate_limit(self):
        """
        Ensure enough time has passed since the
        previous Groq request.
        """

        with self._request_lock:

            current_time = time.monotonic()

            elapsed = (
                current_time -
                self._last_request_time
            )

            remaining = (
                self.request_interval -
                elapsed
            )

            if remaining > 0:
                print(
                    f"Waiting {remaining:.1f}s "
                    "before next Groq request..."
                )

                time.sleep(remaining)

            self._last_request_time = time.monotonic()

    def generate_text(
        self,
        prompt,
        n=1,
        temperature=0.01,
        stop=None,
        callbacks=None,
    ):
        """
        Generate one structured response using Groq.

        Groq requires n=1, so RAGAS requests for
        multiple generations are handled as a
        single-generation request.
        """

        self._wait_for_rate_limit()

        response = litellm.completion(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt.to_string(),
                }
            ],
            response_format={
                "type": "json_object"
            },
            temperature=temperature,
            n=1,
            stop=stop,
            api_key=self.api_key,
        )

        generations = [
            Generation(
                text=choice.message.content
            )
            for choice in response.choices
        ]

        return LLMResult(
            generations=[generations]
        )

    async def agenerate_text(
        self,
        prompt,
        n=1,
        temperature=0.01,
        stop=None,
        callbacks=None,
    ):
        """
        Async-compatible RAGAS generation method.

        Groq requests are intentionally serialized
        through generate_text().
        """

        return self.generate_text(
            prompt=prompt,
            n=1,
            temperature=temperature,
            stop=stop,
            callbacks=callbacks,
        )

    def is_finished(self, response):
        """
        Check whether the response contains
        generated text.
        """

        return bool(
            response.generations
            and response.generations[0]
            and response.generations[0][0].text
        )