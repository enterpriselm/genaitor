from flask import Blueprint, request, jsonify
from config import config
from utils.video_utils import transcribe_audio
from langchain_community.document_loaders import YoutubeLoader
from utils.agents import Agent

video_explainer_bp = Blueprint('video_explainer', __name__)

# Define individual agents for different tasks
video_loader_agent = Agent(
    role='Video Loader Agent',
    system_message=(
        "You are an AI agent specialized in loading and transcribing YouTube videos or local audio files. "
        "Your task is to fetch and transcribe video content or audio content into text format. "
        "Provide the transcript of the video or audio file."
    )
)

ai_explainer_agent = Agent(
    role='AI Explainer Agent',
    system_message=(
        "You are an AI agent specialized in explaining video content. "
        "Your task is to generate a helpful and informative explanation of the video transcript, "
        "based on the user's needs. Please provide the necessary explanation based on the given transcript."
    )
)

prompt_template = """
You are a helpful assistant that explains YT videos. Given the following video transcript:
{video_transcript}
Help the user with their needs.
"""

# Define the main route for Video Explainer
@video_explainer_bp.route('/explain-video', methods=['POST'])
def explain_video():
    """
    Endpoint to transcribe a YouTube video or audio file and generate an AI-based response.
    Expects JSON data with 'video_url' or 'file_path'.
    """
    data = request.get_json()
    video_url = data.get('video_url')
    file_path = data.get('file_path')

    if not video_url and not file_path:
        return jsonify({"error": "A video URL or file path is required"}), 400

    # Step 1: Load and Transcribe the Video (via agents)
    video_transcript = None
    if video_url:
        # Load transcript from YouTube video
        try:
            loader = YoutubeLoader(video_url)
            video_transcript = loader.load()
        except Exception as e:
            return jsonify({"error": f"Failed to load and transcribe the YouTube video: {str(e)}"}), 500
    elif file_path:
        # Transcribe audio from local file
        try:
            video_transcript = transcribe_audio(file_path)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

    if not video_transcript:
        return jsonify({"error": "Failed to obtain video transcript"}), 500

    # Step 2: Generate an Explanation (via AI explainer agent)
    prompt = prompt_template.format(video_transcript=video_transcript)
    explanation = ai_explainer_agent.perform_task({"video_transcript": video_transcript})
    
    if "error" in explanation:
        return jsonify({"error": "Failed to generate explanation", "details": explanation}), 500

    # Return the final response with the explanation
    return jsonify({"ai_response": explanation["content"]})