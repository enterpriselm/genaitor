from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Iterator
from dataclasses import dataclass

@dataclass
class LLMConfig:
    """Basic Settings for LLMs"""
    temperature: float = 0.1
    max_tokens: int = 1000
    verbose: bool = False
    max_retries: int = 3

class LLMProvider(ABC):
    def __init__(self, config: LLMConfig):
        self.config = config

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text based on prompt"""
        pass

    @abstractmethod
    def stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """Text stream based on prompt"""
        pass 
