import os
from genaitor.llm.providers.gemini import GeminiProvider, GeminiConfig
from genaitor.llm.base import LLMProvider

def create_gemini_provider(api_keys: list[str], temperature: float = 0.7, verbose: bool = False, max_tokens: int = 5000) -> GeminiProvider:
    gemini_config = GeminiConfig(
        api_keys=api_keys,
        temperature=temperature,
        verbose=verbose,
        max_tokens=max_tokens
    )
    return GeminiProvider(gemini_config)

def create_gemini_provider_long_context(api_keys: list[str], temperature: float = 0.1, verbose: bool = False, max_tokens: int = 15000) -> GeminiProvider:
    gemini_config = GeminiConfig(
        api_keys=api_keys,
        temperature=temperature,
        verbose=verbose,
        max_tokens=max_tokens
    )
    return GeminiProvider(gemini_config)
