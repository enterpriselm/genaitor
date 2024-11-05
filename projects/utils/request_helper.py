import requests
from config import config

def make_llama_request(user_query, system_message):
    payload = {
        "model": "LLaMA_CPP",
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_query}
        ]
    }
    try:
        response = requests.post(config.LLAMA_API_URL, headers=config.HEADERS, json=payload)
        response.raise_for_status()
        return {"content": response.json()['choices'][0]['message']['content']}
    except requests.exceptions.RequestException as e:
        return {"error": "Failed to connect to AI service", "status_code": 500}
    except (KeyError, IndexError, TypeError) as e:
        return {"error": "Unexpected response structure from AI service", "status_code": 500}
