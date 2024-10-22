from flask import Flask, request, jsonify, stream_with_context, Response
import sqlite3
import requests
from langchain_community.document_loaders import YoutubeLoader
import json

app = Flask(__name__)

LLAMA_API_URL = 'http://localhost:8080/v1/chat/completions'
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": "Bearer no-key"
}

prompt_template = """
You are a helpful assistant that explains YT videos. Given the following video transcript:
{video_transcript}
and the following history of chat:
{history}
Help the user with the following request:
"""

def get_api_keys_from_db():
    conn = sqlite3.connect('genaitor.db')
    cursor = conn.cursor()
    cursor.execute("SELECT api_key FROM api_keys")
    api_keys = [row[0] for row in cursor.fetchall()]
    conn.close()
    return api_keys

def get_payload(youtube_url, user_query):
    loader = YoutubeLoader.from_youtube_url(
    youtube_url, add_video_info=False)

    docs = loader.load()
    transcript = docs[0].page_content
    history = str(json.dumps(load_history()))
    return {
        "model": "LLaMA_CPP",
        "messages": [
            {
                "role": "system",
                "content": prompt_template.format(video_transcript=transcript, history=history)
            },
            {
                "role": "user",
                "content": user_query
            }
        ],
        "stream": True 
    }

def load_history():
    with open('history/chat_history.json', 'r') as f:
        history = json.loads(f.read())
    return history

def save_history(user_query, ai_response):
    history = load_history()
    history.append({"user_query":user_query, "ai_response":ai_response})
    
    if len(history) > 5:
        history = history[1:]
    
    with open('history/chat_history.json', 'w') as f:
        f.write(json.dumps(history))
     
@app.before_request
def require_api_key():
    api_key = request.headers.get('X-API-Key')
    if not api_key:
        return jsonify({"error": "API Key is required"}), 401

    api_keys = get_api_keys_from_db()
    if api_key not in api_keys:
        return jsonify({"error": "Unauthorized - Invalid API Key"}), 401


@app.route('/youtube', methods=['POST'])
def generate_agent():
    data = request.json
    youtube_url = data.get('youtube_url')
    user_query = data.get('user_query')
    
    if not user_query:
        return jsonify({"error": "User query is required"}), 400
    if not youtube_url:
        return jsonify({"error": "Youtube url is required"}), 400
    
    payload = get_payload(youtube_url, user_query)
    
    accumulated_response = []

    def generate():
        nonlocal accumulated_response
        with requests.post(LLAMA_API_URL, headers=HEADERS, json=payload, stream=True) as r:
            if r.status_code != 200:
                yield f"Error: {r.status_code}\n"
                return
            for line in r.iter_lines():
                if line:
                    response_data = json.loads(line.decode('utf-8'))
                    if 'choices' in response_data:
                        ai_response = response_data['choices'][0]['delta']['content']
                        accumulated_response.append(ai_response)  # Acumular resposta
                        yield ai_response  # Enviar parte da resposta para o cliente

        final_response = ''.join(accumulated_response)
        save_history(user_query, final_response)

    return Response(stream_with_context(generate()), content_type='text/plain')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)