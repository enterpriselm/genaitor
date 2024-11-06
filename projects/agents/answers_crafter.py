from flask import Blueprint, request, jsonify
from multiprocessing import Manager, Pool
from utils.agents import Agent, Orchestrator

# Initialize Flask Blueprint
answer_crafter_bp = Blueprint('answer_crafter', __name__)

# Define individual agents with perform_task methods
agents = {
    'validator': Agent(
        role='Validator',
        goal='Validate and correct input for clarity and relevance.',
        system_message=(
            "You are an AI specializing in response validation and correction. "
            "When given an input, assess the accuracy and relevance. If unclear or incorrect, "
            "provide a revised response that aligns with the input's requirements."
        )
    ),
    'researcher': Agent(
        role='Researcher',
        goal='Conduct AI research to find the latest advancements and trends.',
        system_message=(
            "You are an expert researcher in AI advancements. Given validated input, "
            "identify and summarize the latest trends and breakthroughs in AI, especially those from 2024."
        )
    ),
    'writer': Agent(
        role='Content Writer',
        goal='Craft engaging content based on research findings.',
        system_message=(
            "You are a skilled writer who transforms research insights into engaging content. "
            "Take the summarized findings and create a compelling article suitable for a tech-savvy audience."
        )
    )
}

# Define the sequence of tasks for the Orchestrator
tasks = [
    {"description": "Validate input", "agent": agents['validator']},
    {"description": "Conduct research", "agent": agents['researcher']},
    {"description": "Write content", "agent": agents['writer']}
]

# Initialize Orchestrator with agents and tasks for sequential processing
project_orchestrator = Orchestrator(agents=agents, tasks=tasks, process='sequential', cumulative=False)

@answer_crafter_bp.route('/answers-crafter', methods=['POST'])
def generate_answer():
    data = request.get_json()
    
    user_query = data.get('user_query')
    if not user_query:
        return jsonify({"error": "User query is required"}), 400

    # Use Orchestrator to process the pipeline
    result = project_orchestrator.kickoff(user_query)

    if "error" in result:
        return jsonify(result), 500

    # Extract pipeline details for structured response
    pipeline_details = {task['description']: res for task, res in zip(tasks, result["output"])}
    
    return jsonify({
        "ai_agent_prompt": pipeline_details["Write content"],
        "pipeline_details": {
            "validated_input": pipeline_details["Validate input"],
            "research_summary": pipeline_details["Conduct research"]
        }
    }), 200
