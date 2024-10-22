url = 'http://localhost:5001/generate-api-key'

import requests

print(requests.post(url, json={
  "username": "yan.barros"
}
).content)