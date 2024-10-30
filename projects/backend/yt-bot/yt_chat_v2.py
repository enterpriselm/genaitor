from flask import Flask, request, jsonify
import sqlite3
import requests
from langchain_community.document_loaders import YoutubeLoader
import json
import speech_recognition as sr
from pydub import AudioSegment
from moviepy.editor import VideoFileClip
import os
from gpt4all import GPT4All

app = Flask(__name__)

model = GPT4All("Meta-Llama-3-8B-Instruct.Q4_0.gguf")

prompt_template = """
You are a helpful assistant that explains YT videos. Given the following video transcript:
{video_transcript}
Help the user with the following request:
{user_query}
"""

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
     
@app.route('/youtube', methods=['POST'])
def get_answer():
    data = request.json
    youtube_url = data.get('youtube_url')
    user_query = data.get('user_query')
    video_file = request.files.get('video_file')
    
    if not user_query:
        return jsonify({"error": "User query is required"}), 400

    if youtube_url:
        loader = YoutubeLoader.from_youtube_url(youtube_url, add_video_info=False)
        docs = loader.load()
        video_transcript = docs[0].page_content
    elif video_file and (video_file.filename.endswith('.mp4') or video_file.filename.endswith('.mp3')):
        file_path = f"temp_upload.{video_file.filename.split('.')[-1]}"
        video_file.save(file_path)
        try:
            video_transcript = transcribe_audio(file_path)
        finally:
            os.remove(file_path)
    else:
        return jsonify({"error": "You must provide a YouTube URL or a .mp4/.mp3 file"}), 400

    question = prompt_template.format(video_transcript=video_transcript, user_query=user_query)
    with model.chat_session():
        return jsonify({"answer":model.generate(question)}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)