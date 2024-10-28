import PyPDF2
import json
import requests
import os

LLAMA_API_URL = 'http://localhost:8080/v1/chat/completions'
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": "Bearer no-key"
}


def extract_text(filename):
    reader = PyPDF2.PdfReader(filename)
    text = ''
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        text += page.extract_text()

    abstract = text.lower().partition('abstract')[2].partition('introduction')[0]
    metadata = reader.metadata
    return text, metadata, abstract

filenames = ['src/'+x for x in os.listdir('src') if x.endswith('pdf')]
abstract_data = []
i=0
for filename in filenames:
    abstract_data.append(f"Paper {i+1}: {extract_text(filename)[2]}")

SYSTEM_MESSAGE = """You are an AI agent specialized in recommendation system.
The user needs to find the best paper for solving problems with wave solutions.
You have the following papers and their abstracts:
{abstracts}
"""

payload = {
    "model": "LLaMA_CPP",
    "messages": [
        {
            "role": "system",
            "content": SYSTEM_MESSAGE.format(abstracts = ';'.join(abstract_data))
        },
        {
            "role": "user",
            "content": "Based on this, which Paper would you recommend to the user?"
        }
    ]
}

response = requests.post(LLAMA_API_URL, headers=HEADERS, json=payload)

ai_response = response.json()['choices'][0]['message']['content']