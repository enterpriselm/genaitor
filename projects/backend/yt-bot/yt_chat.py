from flask import Flask, request, jsonify
import sqlite3
import requests
from langchain_community.document_loaders import YoutubeLoader
import json
import speech_recognition as sr
from pydub import AudioSegment
from moviepy.editor import VideoFileClip
import os

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
    conn = sqlite3.connect('youtube_chat.db')
    cursor = conn.cursor()
    cursor.execute("SELECT api_key FROM api_keys")
    api_keys = [row[0] for row in cursor.fetchall()]
    conn.close()
    return api_keys

def transcribe_audio(file_path):
    recognizer = sr.Recognizer()

    if file_path.endswith('.mp4'):
        video = VideoFileClip(file_path)
        audio_path = "temp_audio.wav"
        video.audio.write_audiofile(audio_path)
    elif file_path.endswith('.mp3'):
        audio = AudioSegment.from_mp3(file_path)
        audio.export("temp_audio.wav", format="wav")
        audio_path = "temp_audio.wav"
    else:
        raise ValueError("Unsupported file format")

    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        transcript = recognizer.recognize_google(audio_data, language="en-US")

    os.remove(audio_path)
    return transcript

def get_payload(video_transcript, user_query):
    history = str(json.dumps(load_history()))
    return {
        "model": "LLaMA_CPP",
        "messages": [
            {
                "role": "system",
                "content": prompt_template.format(video_transcript=video_transcript, history=history)
            },
            {
                "role": "user",
                "content": user_query
            }
        ],
        "stream": False 
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
     
#@app.before_request
#def require_api_key():
#    api_key = request.headers.get('X-API-Key')
#    if not api_key:
#        return jsonify({"error": "API Key is required"}), 401

#    api_keys = get_api_keys_from_db()
#    if api_key not in api_keys:
#        return jsonify({"error": "Unauthorized - Invalid API Key"}), 401

@app.route('/youtube', methods=['POST'])
def get_answer():
    data = request.form
    youtube_url = data.get('youtube_url')
    user_query = data.get('user_query')
    #video_file = request.files.get('video_file')
    
    if not user_query:
        return jsonify({"error": "User query is required"}), 400

    if youtube_url:
        loader = YoutubeLoader.from_youtube_url(youtube_url, add_video_info=False)
        docs = loader.load()
        video_transcript = docs[0].page_content
    #elif video_file and (video_file.filename.endswith('.mp4') or video_file.filename.endswith('.mp3')):
    #    file_path = f"temp_upload.{video_file.filename.split('.')[-1]}"
    #    video_file.save(file_path)
    #    try:
    #        video_transcript = transcribe_audio(file_path)
    #    finally:
    #        os.remove(file_path)
    else:
        return jsonify({"error": "You must provide a YouTube URL or a .mp4/.mp3 file"}), 400

    payload = get_payload(video_transcript, user_query)
    response = requests.post(LLAMA_API_URL, headers=HEADERS, json=payload)
    return response.json()['choices'][0]['message']['content']

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)