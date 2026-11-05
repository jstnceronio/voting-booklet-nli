from openai import OpenAI

from voting_nli.config import settings


class Model:
    def __init__(self):
        self.llm = OpenAI(api_key=settings.api_key, base_url=settings.base_url)

    def get_response(self, message) -> str | None:
        response = self.llm.chat.completions.create(
            model="swiss-ai/Apertus-v1.5-70B-thinking",
            messages=[
                {"role": "user", "content": message},
            ],
        )
        return response.choices[0].message.content

