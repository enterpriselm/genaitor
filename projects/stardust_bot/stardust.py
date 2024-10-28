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
            "content": """You are a helpful assistant that helps authistic people to study and learn new things. Given the following hyperfocus, and subject for studying, help the student:
            Give the student a guideline of topics to study, techniques to help, references, ambient preparation, exercises, everything that make this authistic person improve their knowledge."""
        },
        {
            "role": "user",
            "content": """hyperfocus = math puzzles
subject = "calculus"
"""
        }
    ]

}

response = requests.post(LLAMA_API_URL, headers=HEADERS, json=payload)

ai_response = response.json()['choices'][0]['message']['content']