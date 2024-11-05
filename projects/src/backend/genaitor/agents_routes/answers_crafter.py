from flask import Flask, request, jsonify
import sqlite3
import requests

app = Flask(__name__)

LLAMA_API_URL = 'http://localhost:8080/v1/chat/completions'
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": "Bearer no-key"
}

SYSTEM_MESSAGE = "I need an AI Agent that does this: Based on an input to a llm and the output generated, you will validate if the output is correct or not. If is not correct, you will generate a new output'"

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
