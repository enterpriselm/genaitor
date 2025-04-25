import openai
from ..base import LLMProvider, LLMConfig
from typing import Optional, List, Iterator

class OpenAIConfig(LLMConfig):
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4",
        organization: Optional[str] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.api_key = api_key
        self.model = model
        self.organization = organization

class OpenAIProvider(LLMProvider):
    def __init__(self, config: OpenAIConfig):
        super().__init__(config)
        openai.api_key = config.api_key
        if config.organization:
            openai.organization = config.organization

    def generate(self, prompt: str, **kwargs) -> str:
        try:
            response = openai.ChatCompletion.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get('temperature', self.config.temperature),
                max_tokens=kwargs.get('max_tokens', self.config.max_tokens)
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"OpenAI generation failed: {str(e)}")

    def stream(self, prompt: str, **kwargs) -> Iterator[str]:
        try:
            response = openai.ChatCompletion.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get('temperature', self.config.temperature),
                max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
                stream=True
            )
            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            raise Exception(f"OpenAI streaming failed: {str(e)}") 