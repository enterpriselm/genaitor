from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    LLAMA_API_URL = os.getenv("LLAMA_API_URL")
    HEADERS = {
        "Content-Type": "application/json",
        "Authorization": os.getenv("AUTH_TOKEN")
    }

config = Config()
