import os
import json

flask_code_string = """from flask import Flask, request, jsonify
import sqlite3
import requests

app = Flask(__name__)

LLAMA_API_URL = 'http://localhost:8080/v1/chat/completions'
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": "Bearer no-key"
}

SYSTEM_MESSAGE = "PROMPT"

def get_api_keys_from_db():
    conn = sqlite3.connect('genaitor.db')
    cursor = conn.cursor()
    cursor.execute("SELECT api_key FROM api_keys")
    api_keys = [row[0] for row in cursor.fetchall()]
    conn.close()
    return api_keys

@app.before_request
def require_api_key():
    api_key = request.headers.get('X-API-Key')
    if not api_key:
        return jsonify({"error": "API Key is required"}), 401

    api_keys = get_api_keys_from_db()
    if api_key not in api_keys:
        return jsonify({"error": "Unauthorized - Invalid API Key"}), 401

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
"""

agents_prompts = ['agents_prompts/'+x for x in os.listdir('agents_prompts')]
for agent in agents_prompts:
    with open(agent) as f:
        agent_data = json.load(f)
    prompt = agent_data['agent_prompt'].split("""content""")[1][2:].partition('}]}')[0].partition(' ')[2][1:]
    path = agent.replace('.json','.py').replace('agents_prompts/','agents_routes/')
    with open(path, 'w') as f:
        f.write(flask_code_string.replace('PROMPT',prompt))