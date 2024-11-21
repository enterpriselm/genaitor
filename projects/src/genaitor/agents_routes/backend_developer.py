from flask import Flask, request, jsonify
import sqlite3
import requests

app = Flask(__name__)

LLAMA_API_URL = 'http://localhost:8080/v1/chat/completions'
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": "Bearer no-key"
}

SYSTEM_MESSAGE = "You are a highly skilled backend developer proficient in Python. You have experience creating Flask APIs and can easily follow project guidelines. Please create a Flask API for the task provided.'"

@app.route('/generate-agent', methods=['POST'])
def generate_agent():
    data = request.json
    user_query = data.get('user_query')
    
    if not user_query:
        return jsonify({"error": "User query is required"}), 400

    payload = {
        "model": "LLaMA_CPP",
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_MESSAGE
            },
            {
                "role": "user",
                "content": user_query
            }
        ]
    }

    response = requests.post(LLAMA_API_URL, headers=HEADERS, json=payload)

    if response.status_code == 200:
        ai_response = response.json()['choices'][0]['message']['content']
        return jsonify({"ai_agent_prompt": ai_response})
    else:
        return jsonify({"error": "Failed to generate AI agent"}), response.status_code

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
