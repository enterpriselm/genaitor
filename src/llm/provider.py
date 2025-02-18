import os
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
import google.generativeai as genai
from absl import logging as absl_logging
import time
from tenacity import retry, stop_after_attempt, wait_exponential
from functools import lru_cache
from .key_manager import APIKeyManager

# Suprimir avisos do absl
absl_logging.set_verbosity(absl_logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        pass

class GeminiProvider(LLMProvider):
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_keys: Optional[List[str]] = None,
        model: str = "gemini-1.5-pro",
        temperature: float = 0.1,
        max_tokens: int = 1000,
        verbose: bool = False,
        max_retries: int = 3
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verbose = verbose
        self.max_retries = max_retries
        
        # Configurar gerenciador de chaves
        self.key_manager = APIKeyManager(api_keys)
        if api_key:
            self.key_manager.add_key(api_key)
        
        # Configurar logging
        if not self.verbose:
            logging.getLogger('tensorflow').setLevel(logging.ERROR)
            logging.getLogger('absl').setLevel(logging.ERROR)
        
        try:
            # Tenta obter uma chave válida
            self.current_key = self.key_manager.get_next_key()
            if not self.current_key:
                raise ValueError("No API key provided")
                
            genai.configure(api_key=self.current_key)
            self._model = genai.GenerativeModel(model)
        except Exception as e:
            raise Exception(f"Failed to initialize Gemini model: {str(e)}")

    def _try_with_new_key(self):
        """Tenta configurar uma nova chave API"""
        try:
            self.current_key = self.key_manager.get_next_key()
            genai.configure(api_key=self.current_key)
            self._model = genai.GenerativeModel(self.model)
            if self.verbose:
                print(f"Switched to new API key")
        except Exception as e:
            raise Exception(f"Failed to switch API key: {str(e)}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def generate(self, prompt: str, **kwargs) -> str:
        try:
            generation_config = genai.GenerationConfig(
                max_output_tokens=kwargs.get('max_tokens', self.max_tokens),
                temperature=kwargs.get('temperature', self.temperature),
            )
            
            if self.verbose:
                print(f"Attempting to generate content...")
            
            response = self._model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            if self.verbose:
                print(f"Generation successful!")
            
            return response.text
            
        except Exception as e:
            if "429" in str(e):
                if self.verbose:
                    print(f"Rate limit hit, switching API key...")
                self.key_manager.mark_key_failed(self.current_key)
                self._try_with_new_key()
                raise  # Permite que o retry tente novamente com a nova chave
            raise Exception(f"Failed to generate content: {str(e)}")

    @lru_cache(maxsize=100)
    def _cached_generate(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """Versão cacheada da geração de conteúdo"""
        return self.generate(prompt, temperature=temperature, max_tokens=max_tokens)

    def generate(self, prompt: str, **kwargs) -> str:
        # Use cache apenas se temperatura e max_tokens forem os padrões
        if (kwargs.get('temperature', self.temperature) == self.temperature and 
            kwargs.get('max_tokens', self.max_tokens) == self.max_tokens):
            return self._cached_generate(
                prompt, 
                self.temperature, 
                self.max_tokens
            )
        return self._generate(prompt, **kwargs)

    def __del__(self):
        """Cleanup when the provider is destroyed"""
        try:
            # Limpar recursos do gRPC
            import grpc
            grpc.shutdown_all_channels()
        except:
            pass 