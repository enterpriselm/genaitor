import requests
from ..base import LLMProvider, LLMConfig
from typing import Optional, Dict, Any, Iterator

class DeepSeekConfig(LLMConfig):
    def __init__(
        self,
        api_key: str,
        base_url: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 30,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.api_key = api_key
        self.base_url = base_url
        self.headers = headers or {}
        self.timeout = timeout

class DeepSeekProvider(LLMProvider):
    def __init__(self, config: DeepSeekConfig):
        super().__init__(config)
        self.session = requests.Session()
        self.session.headers.update(config.headers)
        self.session.headers["Authorization"] = f"Bearer {config.api_key}"

    def generate(self, prompt: str, **kwargs) -> str:
        try:
            response = self.session.post(
                self.config.base_url,
                json={
                    "prompt": prompt,
                    "temperature": kwargs.get('temperature', self.config.temperature),
                    "max_tokens": kwargs.get('max_tokens', self.config.max_tokens),
                    **kwargs
                },
                timeout=self.config.timeout
            )
            response.raise_for_status()
            return response.json()["text"]
        except Exception as e:
            raise Exception(f"DeepSeek generation failed: {str(e)}") 