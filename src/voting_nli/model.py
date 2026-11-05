from time import perf_counter

from openai import OpenAI

from voting_nli.config import settings


class Model:
    def __init__(self):
        self.llm = OpenAI(api_key=settings.api_key, base_url=settings.base_url)

    def get_response(self, message) -> tuple[str | None, dict]:
        started = perf_counter()
        response = self.llm.chat.completions.create(
            model="swiss-ai/Apertus-v1.5-70B-thinking",
            messages=[
                {"role": "user", "content": message},
            ],
        )
        elapsed_ms = (perf_counter() - started) * 1000
        usage = response.usage
        metrics = {
            "input_tokens": usage.prompt_tokens if usage is not None else None,
            "output_tokens": usage.completion_tokens if usage is not None else None,
            "inference_time_ms": round(elapsed_ms, 3),
        }
        return response.choices[0].message.content, metrics

