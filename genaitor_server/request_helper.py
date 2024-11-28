import os
import logging
from cachetools import TTLCache
import requests
from genaitor.config import config

# Configure logging
logging.basicConfig(level=logging.ERROR)

# Cache setup
cache = TTLCache(maxsize=100, ttl=300)

def sanitize_input(input_str):
    if not isinstance(input_str) or len(input_str) > 10000:
        raise ValueError("Invalid input")
    return input_str.strip()

def get_llama_answers(user_query, system_message, max_tokens=150, temperature=0.8):
    payload = {
        "model": "LLaMA_CPP",
        "messages": [
            {"role": "system", "content": sanitize_input(system_message)},
            {"role": "user", "content": sanitize_input(user_query)}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    try:
        response = requests.post(config.LLAMA_API_URL, headers=config.HEADERS, json=payload, timeout=90)
        response.raise_for_status()
        return {"content": response.json()['choices'][0]['message']['content']}
    except requests.exceptions.RequestException as e:
        logging.error("Unexpected response structure", exc_info=True)
        return {"error": "Failed to connect to AI service", "status_code": 500}
    except (KeyError, IndexError, TypeError) as e:
        return {"error": "Unexpected response structure from AI service", "status_code": 500}
    
def make_llama_request(user_query, system_message, max_tokens=2000, max_iterations=1, temperature=0.8):
    cache_key = f"{user_query}:{system_message}:{max_tokens}:{temperature}"
    if cache_key in cache:
        return cache[cache_key]
    
    full_response = ""
    current_query = user_query
    iteration = 0

    try:
        while iteration < max_iterations:
            iteration += 1
            response_chunk = get_llama_answers(current_query, system_message, max_tokens=max_tokens, temperature=temperature)['content']
            full_response += response_chunk.strip() + " "

            if response_chunk.strip().endswith((".", "!", "?", "”", "\"")):
                break            

            current_query = "Continue: " + response_chunk.strip()

        result = {"content": full_response.strip().replace(system_message, '')}
        cache[cache_key] = result
        return result
    except Exception as e:
        logging.error("Unexpected error", exc_info=True)
        return {"error": "An unexpected error occured", "status_code": 500}
    