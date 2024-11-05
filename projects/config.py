import os

class Config:
    LLAMA_API_URL = os.getenv("LLAMA_API_URL", "http://localhost:8080/v1/chat/completions")
    HEADERS = {
        "Content-Type": "application/json",
        "Authorization": os.getenv("AUTH_TOKEN", "Bearer no-key")
    }

config = Config()
