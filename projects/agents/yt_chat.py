from flask import Blueprint, request, jsonify
from config import config
from utils.video_utils import transcribe_audio
from langchain_community.document_loaders import YoutubeLoader
import os

video_explainer_bp = Blueprint('video_explainer', __name__)

prompt_template = """
You are a helpful assistant that explains YT videos. Given the following video transcript:
{video_transcript}
Help the user with their needs.
"""

@video_explainer_bp.route('/explain-video', methods=['POST'])
def explain_video():
    """
    Endpoint to transcribe a YouTube video or audio file and generate an AI-based response.
    Expects JSON data with 'video_url' or 'file_path'.
    """
    data = request.get_json()
    video_url = data.get('video_url')
    file_path = data.get('file_path')

    if video_url:
        # Load transcript from YouTube video
        loader = YoutubeLoader(video_url)
        video_transcript = loader.load()
    elif file_path:
        # Transcribe audio from local file
        try:
            video_transcript = transcribe_audio(file_path)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
    else:
        return jsonify({"error": "A video URL or file path is required"}), 400

    # Format the prompt with the video transcript
    prompt = prompt_template.format(video_transcript=video_transcript)
    
    # Assuming a call to an AI API or some local model
    # For example: response = make_llama_request(prompt)
    response = {"content": prompt}  # Placeholder response

    return jsonify({"ai_response": response["content"]})
