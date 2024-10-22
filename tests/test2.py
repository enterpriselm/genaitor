api_key = 'fc288116-2bd1-442d-abe2-a8cc6e5e0111'
payload = {
  "user_query": "I need an AI agent for summarize youtube videos."
}

url = 'http://localhost:5000/generate-agent'

import requests
print(requests.post(url, headers={"X-API-Key": api_key}, json=payload))