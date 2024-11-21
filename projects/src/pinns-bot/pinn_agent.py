import requests

LLAMA_API_URL = 'http://localhost:8080/v1/chat/completions'
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": "Bearer no-key"
}

payload = {
    "model": "LLaMA_CPP",
    "messages": [
        {
            "role": "system",
            "content": """From now on, you are an AI agent specialized in Physics Informed Neural Networks. I will present a physical problem, and you will show me the PyTorch modeling to solve it."""
        },
        {
            "role": "user",
            "content": user_request
        }
    ]

}

response = requests.post(LLAMA_API_URL, headers=HEADERS, json=payload)

ai_response = response.json()['choices'][0]['message']['content']