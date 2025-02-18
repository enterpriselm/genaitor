import random
import time
from typing import List, Optional, Dict

class APIKeyManager:
    def __init__(self, api_keys: Optional[List[str]] = None):
        """
        Inicializa o gerenciador de chaves API.
        
        Args:
            api_keys: Lista opcional de chaves API. Se não fornecida, 
                     uma chave deverá ser fornecida diretamente ao provider.
        """
        self._api_keys = api_keys or []
        self._current_index = 0
        self._failed_keys: Dict[str, float] = {}  # key -> timestamp
        self._failure_timeout = 60  # segundos para resetar uma chave

    def add_key(self, key: str) -> None:
        """Adiciona uma nova chave API ao gerenciador"""
        if key not in self._api_keys:
            self._api_keys.append(key)

    def add_keys(self, keys: List[str]) -> None:
        """Adiciona múltiplas chaves API ao gerenciador"""
        for key in keys:
            self.add_key(key)

    def get_next_key(self) -> Optional[str]:
        """Retorna a próxima chave API disponível ou None se não houver chaves"""
        if not self._api_keys:
            return None
            
        self._reset_timed_out_keys()
        
        attempts = 0
        while attempts < len(self._api_keys):
            key = self._api_keys[self._current_index]
            self._current_index = (self._current_index + 1) % len(self._api_keys)
            
            if key not in self._failed_keys:
                return key
                
            attempts += 1
            
        return None

    def mark_key_failed(self, key: str) -> None:
        """Marca uma chave como falha com timestamp"""
        if key in self._api_keys:
            self._failed_keys[key] = time.time()

    def reset_failed_keys(self) -> None:
        """Reseta as chaves falhas"""
        self._failed_keys.clear()

    def _reset_timed_out_keys(self) -> None:
        """Reseta chaves que falharam há mais tempo que o timeout"""
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self._failed_keys.items()
            if current_time - timestamp > self._failure_timeout
        ]
        for key in expired_keys:
            del self._failed_keys[key]

    def get_random_key(self) -> Optional[str]:
        """Retorna uma chave aleatória que não falhou ou None se não houver chaves disponíveis"""
        available_keys = [k for k in self._api_keys if k not in self._failed_keys]
        return random.choice(available_keys) if available_keys else None 