import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm import GeminiProvider, GeminiConfig
from dotenv import load_dotenv
load_dotenv('.env')

api_keys = [os.getenv('API_KEY')]

def gemini_provider(api_keys=api_keys, temperature=0.7, verbose=False, max_tokens=5000):
    gemini_config = GeminiConfig(
        api_keys=api_keys,
        temperature=temperature,
        verbose=verbose,
        max_tokens=max_tokens
    )

    return GeminiProvider(gemini_config)
