import anthropic
from ..base import LLMProvider, LLMConfig
from typing import Optional, Iterator

class ClaudeConfig(LLMConfig):
    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-opus-20240229",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.api_key = api_key
        self.model = model

class ClaudeProvider(LLMProvider):
    def __init__(self, config: ClaudeConfig):
        super().__init__(config)
        self.client = anthropic.Anthropic(api_key=config.api_key)

    def generate(self, prompt: str, **kwargs) -> str:
        try:
            message = self.client.messages.create(
                model=self.config.model,
                max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
                temperature=kwargs.get('temperature', self.config.temperature),
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return message.content[0].text
        except Exception as e:
            raise Exception(f"Claude generation failed: {str(e)}") 