import requests
from genaitor.config import config

def get_llama_answers(user_query, system_message, max_tokens=150, temperature=0.8):
    payload = {
        "model": "LLaMA_CPP",
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_query}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    try:
        response = requests.post(config.LLAMA_API_URL, headers=config.HEADERS, json=payload)
        response.raise_for_status()
        return {"content": response.json()['choices'][0]['message']['content']}
    except requests.exceptions.RequestException as e:
        return {"error": "Failed to connect to AI service", "status_code": 500}
    except (KeyError, IndexError, TypeError) as e:
        return {"error": "Unexpected response structure from AI service", "status_code": 500}
    
def make_llama_request(user_query, system_message, max_tokens=50, max_iterations=20, temperature=0.8):
    full_response = ""
    current_query = user_query
    iteration = 0

    try:
        while iteration < max_iterations:
            iteration += 1
            # Get the response from LLaMA
            response_chunk = get_llama_answers(current_query, system_message, max_tokens=max_tokens, temperature=temperature)['content']

            # Append the chunk to the full response
            full_response += response_chunk.strip() + " "

            # Check if the response seems complete (e.g., ends with punctuation)
            if response_chunk.strip().endswith((".", "!", "?", "”", "\"")):
                break  # Stop if it looks complete

            # Update the current query to ask for continuation
            current_query = "Continue: " + response_chunk.strip()

        return {"content": full_response.strip().replace(system_message, '')}
    except requests.exceptions.RequestException as e:
        return {"error": "Failed to connect to AI service", "status_code": 500}
    except (KeyError, IndexError, TypeError) as e:
        return {"error": "Unexpected response structure from AI service", "status_code": 500}
    