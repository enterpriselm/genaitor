from flask import Blueprint, request, jsonify
from genaitor.config import config
from genaitor.video_utils import transcribe_audio
from langchain_community.document_loaders import YoutubeLoader
from genaitor.utils.agents import Agent, Orchestrator, Task

video_explainer_bp = Blueprint('video_explainer', __name__)

# Define individual agents for different tasks
agents = {
    'ai_explainer_agent' : Agent(
        role='AI Explainer Agent',
        system_message=(
            """You are an AI agent specializing in summarizing long texts of any complexity."""
     ),
        temperature=0.8,
        max_tokens=2000,
        max_iterations=1
    )
}

class VideoTasks():

    def summary_task(self, agent, text):
        return Task(
            description=f"""
            This task involves extract the key points and provide a concise, clear, and coherent summary. Focus on the main ideas, arguments, and conclusions while excluding unnecessary details, examples, or elaborations. The summary should be comprehensive yet brief, capturing the essence of the original text in no more than 1-3 paragraphs, depending on the length and complexity. Ensure that the summary maintains the core message and intent of the original content, without adding personal interpretations or opinions..
            
            Text: {text}
            """,
            expected_output=f"""
            A brief summary of the gathered information, in a concise text format.
            """,
            agent=agent,
            output_file='trip_summary.txt',
        )
    
    def needs_based_response_task(self, agent, user_input, review_details):
        return Task(
            description=f"""
            This task requires the agent to respond to the user's specific needs by considering both the user input and the review previously conducted by the agent. The agent should craft a tailored response that directly addresses the user's requirements and provides useful, actionable information based on the prior review.
            
            User Input: {user_input}
            Previous Review Details: {review_details}
            """,
            expected_output=f"""
            A clear and personalized response that answers the user's specific question or request, incorporating information from the previous review.
            """,
            agent=agent,
            output_file='needs_based_response.txt',
        )


video_explainer_orchestrator = Orchestrator(agents=agents, tasks=tasks, process='sequential', cumulative=False)

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
    result = video_explainer_orchestrator.kickoff(prompt)
    
    if "error" in result:
        return jsonify(result), 500

    # Format the output from each task in the pipeline
    response_data = {task['description']: res for task, res in zip(tasks, result["output"])}

    return jsonify(response_data), 200
