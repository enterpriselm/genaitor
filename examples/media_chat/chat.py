from genaitor.core import Agent, Task, Orchestrator
from genaitor.llm import get_random_api_key
from genaitor.utils.media import process_media_files
from typing import List, Dict, Any

class MediaChatTasks:
    @staticmethod
    def create_extraction_task():
        return Task(
            description="Extract relevant information from provided media based on user query",
            goal="Find and extract specific information that answers the user's question",
            output_format="Relevant excerpt or information that directly addresses the query"
        )

    @staticmethod
    def create_validation_task():
        return Task(
            description="Validate the extracted information against the user's query",
            goal="Ensure the extracted information properly answers the question",
            output_format="Validation result with explanation"
        )

    @staticmethod
    def create_refinement_task():
        return Task(
            description="Refine and improve the answer",
            goal="Make the response more precise and helpful",
            output_format="Refined answer in clear, natural language"
        )

def create_media_agents():
    tasks = MediaChatTasks()
    return {
        'extractor': Agent(
            role='Media Content Extractor',
            task=tasks.create_extraction_task()
        ),
        'validator': Agent(
            role='Content Validator',
            task=tasks.create_validation_task()
        ),
        'refiner': Agent(
            role='Response Refiner',
            task=tasks.create_refinement_task()
        )
    }

def execute_media_query(
    media_files: List[str],
    user_query: str = "What is this content about?"
) -> Dict[str, Any]:
    # Process all media files into text
    text = process_media_files(media_files)
    
    if not text:
        raise Exception("No content could be extracted from the provided media files")

    # Create and run agents
    agents = create_media_agents()
    manager = Orchestrator(get_random_api_key())
    answer, history = manager.main_pipeline(
        user_query=f"From this content:\n\n{text}\n\nAnswer: {user_query}",
        agents=agents
    )

    return {
        "explanation": answer,
        "text": text,
        "history": history
    } 