from flask import Blueprint, request, jsonify
from config import config
from utils.agents import Agent, Orchestrator

pm_bp = Blueprint('project_management', __name__)

# Define agents for different aspects of project management
agents = {
    'task_assignment_agent': Agent(
        role='Task Assignment Agent',
        system_message=(
            "You are an AI agent specialized in assigning tasks to frontend and backend development teams. "
            "Given the user's query, you will assign tasks to the appropriate pipeline (frontend or backend) and ensure they are tracked correctly."
        )
    ),
    'task_tracking_agent': Agent(
        role='Task Tracking Agent',
        system_message=(
            "You are an AI agent specialized in tracking the progress of tasks. "
            "Given the user's query, you will check whether the tasks have been completed and provide another task if necessary."
        )
    ),
    'quality_assurance_agent': Agent(
        role='Quality Assurance Agent',
        system_message=(
            "You are an AI agent specialized in quality assurance for project tasks. "
            "After the task is completed, you will verify its correctness and suggest improvements or adjustments."
        )
    ),
    'frontend_agent': Agent(
        role='Frontend Agent',
        system_message=(
            "You are an AI agent specialized in frontend development tasks. "
            "You will handle tasks related to user interfaces, experience, and anything front-end related in the project."
        )
    ),
    'backend_agent': Agent(
        role='Backend Agent',
        system_message=(
            "You are an AI agent specialized in backend development tasks. "
            "You will handle tasks related to server-side logic, databases, and API design in the project."
        )
    ),
}

# Define the task flow for Project Management analysis
tasks = [
    {"description": "Task assignment to frontend or backend", "agent": agents['task_assignment_agent']},
    {"description": "Task tracking", "agent": agents['task_tracking_agent']},
    {"description": "Quality assurance", "agent": agents['quality_assurance_agent']},
    {"description": "Frontend task execution", "agent": agents['frontend_agent']},
    {"description": "Backend task execution", "agent": agents['backend_agent']},
]

# Initialize Orchestrator to manage the task flow
pm_orchestrator = Orchestrator(agents=agents, tasks=tasks, process='sequential', cumulative=False)

# Define the main route for Project Management analysis
@pm_bp.route('/project-management', methods=['POST'])
def project_management():
    data = request.get_json()

    # Validate user input
    user_query = data.get('user_query')
    if not user_query:
        return jsonify({"error": "User query is required"}), 400

    # Use orchestrator to handle the pipeline for project management tasks
    result = pm_orchestrator.kickoff(user_query)

    if "error" in result:
        return jsonify(result), 500

    # Format the output from each task in the pipeline
    response_data = {task['description']: res for task, res in zip(tasks, result["output"])}

    return jsonify(response_data), 200
