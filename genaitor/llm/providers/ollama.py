import requests
from ..base import LLMProvider, LLMConfig
from typing import Optional, Dict, Any

class OllamaConfig(LLMConfig):
    def __init__(
        self,
        base_url: str = "http://localhost:11434",  # Default Ollama server URL
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 30,
        model: str = "llama3.2",  # Default model
        **kwargs
    ):
        super().__init__(**kwargs)
        self.base_url = base_url
        self.headers = headers or {}
        self.timeout = timeout
        self.model = model

class OllamaProvider(LLMProvider):
    def __init__(self, config: OllamaConfig):
        self.config = config

    def generate(self, prompt: str) -> str:
        """Generate text from the Ollama model based on the provided prompt."""
        try:
            response = requests.post(
                f"{self.config.base_url}/generate",
                json={"model": self.config.model, "prompt": prompt},
                headers=self.config.headers,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            return response.json().get("text", "")
        except Exception as e:
            raise Exception(f"Ollama generation failed: {str(e)}")

    def chat(self, messages: list) -> str:
        """Handles chat interactions with the Ollama model."""
        try:
            response = requests.post(
                f"{self.config.base_url}/chat",
                json={"model": self.config.model, "messages": messages},
                headers=self.config.headers,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            return response.json().get("message", {}).get("content", "")
        except Exception as e:
            raise Exception(f"Ollama chat interaction failed: {str(e)}") 