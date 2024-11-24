from genaitor.config import config
from genaitor.video_utils import transcribe_audio
from langchain_community.document_loaders import YoutubeLoader
from genaitor.utils.agents import Agent, Orchestrator, Task

# Define individual agents for different tasks
agents = {
    'ai_explainer_agent': Agent(
        role='AI Explainer Agent',
        system_message=(
            """You are an AI agent specializing in summarizing and explaining long texts of any complexity."""
        ),
        temperature=0.8,
        max_tokens=2000,
        max_iterations=1,
    ),
    'ai_answer_agent': Agent(
        role='AI Answer Agent',
        system_message=(
            """You are an AI agent specializing in answering questions from texts."""
        ),
        temperature=0.8,
        max_tokens=2000,
        max_iterations=1,
    )
}

class VideoTasks():

    def summary_task(self, agent):
        return Task(
            description=f"""
            Your task involves extracting the key points and providing a concise, clear, and coherent summary. 
            Focus on the main ideas, arguments, and conclusions while excluding unnecessary details, examples, or elaborations. 
            The summary should be comprehensive yet brief, capturing the essence of the original text in no more than 1-3 paragraphs.
            """,
            expected_output=f"""
            A brief summary of the gathered information, in a concise text format.
            """,
            agent=agent,
            output_file='trip_summary.txt',
            goal="""Based on the following text:
            {review}
            
            Generate a complete review of two paragraphs.
            """
        )

    def needs_based_response_task(self, agent):
        return Task(
            description=f"""
            Your task requires the to respond to the user's specific needs by considering both the user input and the review previously conducted by the agent. 
            You should craft a tailored response that directly addresses the user's requirements and provides useful, actionable information based on the prior review.
            """,
            expected_output=f"""
            A clear and personalized response that answers the user's specific question or request.
            """,
            agent=agent,
            output_file='needs_based_response.txt',
            goal="""
            Based on this text:
            {review}
            
            Based on this prompt i send you right now, please do the follow for me:
            {user_query}
            """
        )

video_tasks = VideoTasks()

# Placeholder for video id and file path
video_id = '5rqZyKeErkY'
file_path = ''

# Check if video ID or file path is provided
if not video_id and not file_path:
    print("Need video")
    
# Step 1: Load and Transcribe the Video (via agents)
video_transcript = None
if video_id != '':
    # Load transcript from YouTube video
    try:
        loader = YoutubeLoader(video_id)
        video_transcript = loader.load()[0]
    except Exception as e:
        print(f"Error loading video: {e}")
elif file_path != '':
    # Transcribe audio from local file
    try:
        video_transcript = transcribe_audio(file_path)
    except ValueError as e:
        print(f"Error transcribing audio: {e}")

# If transcription failed
if not video_transcript:
    print("Failed to load or transcribe video")

summary_task = video_tasks.summary_task(agents['ai_explainer_agent'])
needs_based_response_task = video_tasks.needs_based_response_task(agents['ai_answer_agent'])

# Step 2: Create the Orchestrator and add tasks
video_explainer_orchestrator = Orchestrator(agents=agents, tasks=[summary_task, needs_based_response_task], process='sequential', cumulative=False)

# Step 3: Execute the Orchestrator's tasks
user_query = "List for me all the pokemon that appear on the text"
result = video_explainer_orchestrator.kickoff(user_query=user_query, review=video_transcript.page_content)