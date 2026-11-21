from time import perf_counter

from openai import APITimeoutError, OpenAI

from voting_nli.config import Settings


class Model:
    def __init__(self):
        self.settings = Settings()
        self.client = OpenAI(
            api_key=self.settings.api_key, base_url=self.settings.base_url,
            timeout=180.0, max_retries=0,
        )

    def get_response(self, message: str) -> tuple[str | None, dict]:
        started = perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=self.settings.model,
                messages=[
                    {"role": "user", "content": message},
                ],
            )
        except APITimeoutError as error:
            raise TimeoutError(
                "Model API timed out (timeout: 180 seconds); request was not retried."
            ) from error
        elapsed_ms = (perf_counter() - started) * 1000
        usage = response.usage
        metrics = {
            "input_tokens": usage.prompt_tokens if usage is not None else None,
            "output_tokens": usage.completion_tokens if usage is not None else None,
            "inference_time_ms": round(elapsed_ms, 3),
        }
        content = response.choices[0].message.content
        # Apertus can include reasoning before its final-answer marker.
        if content and "<|inner_suffix|>" in content:
            content = content.rsplit("<|inner_suffix|>", 1)[1].strip()
        return content, metrics

