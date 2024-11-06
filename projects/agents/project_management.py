from flask import Blueprint, request, jsonify
from config import config
from utils.agents import Agent

pm_bp = Blueprint('project_management', __name__)

# Define individual agents for different aspects of project management
task_assignment_agent = Agent(
    role='Task Assignment Agent',
    system_message=(
        "You are an AI agent specialized in assigning tasks to frontend and backend development teams. "
        "Given the user's query, you will assign tasks to the appropriate pipeline (frontend or backend) and ensure they are tracked correctly."
    )
)

task_tracking_agent = Agent(
    role='Task Tracking Agent',
    system_message=(
        "You are an AI agent specialized in tracking the progress of tasks. "
        "Given the user's query, you will check whether the tasks have been completed and provide another task if necessary."
    )
)

quality_assurance_agent = Agent(
    role='Quality Assurance Agent',
    system_message=(
        "You are an AI agent specialized in quality assurance for project tasks. "
        "After the task is completed, you will verify its correctness and suggest improvements or adjustments."
    )
)

frontend_agent = Agent(
    role='Frontend Agent',
    system_message=(
        "You are an AI agent specialized in frontend development tasks. "
        "You will handle tasks related to user interfaces, experience, and anything front-end related in the project."
    )
)

backend_agent = Agent(
    role='Backend Agent',
    system_message=(
        "You are an AI agent specialized in backend development tasks. "
        "You will handle tasks related to server-side logic, databases, and API design in the project."
    )
)

# Define the main route for Project Management analysis
@pm_bp.route('/project-management', methods=['POST'])
def project_management():
    data = request.get_json()

    # Validate user input
    user_query = data.get('user_query')
    if not user_query:
        return jsonify({"error": "User query is required"}), 400

    # Step 1: Assign tasks to frontend or backend
    task_assignment_output = task_assignment_agent.perform_task(user_query)
    if "error" in task_assignment_output:
        return jsonify({"error": "Task assignment failed.", "details": task_assignment_output}), 500

    # Step 2: Track task progress
    task_tracking_output = task_tracking_agent.perform_task(user_query)
    if "error" in task_tracking_output:
        return jsonify({"error": "Task tracking failed.", "details": task_tracking_output}), 500

    # Step 3: Quality assurance after task completion
    quality_assurance_output = quality_assurance_agent.perform_task(user_query)
    if "error" in quality_assurance_output:
        return jsonify({"error": "Quality assurance failed.", "details": quality_assurance_output}), 500

    # Step 4: Assign tasks to frontend and backend agents
    frontend_task_output = frontend_agent.perform_task(user_query)
    backend_task_output = backend_agent.perform_task(user_query)

    if "error" in frontend_task_output or "error" in backend_task_output:
        return jsonify({"error": "Frontend or backend task failed.", "details": {"frontend": frontend_task_output, "backend": backend_task_output}}), 500

    # Format the final response with each agent's output
    response_data = {
        "task_assignment": task_assignment_output,
        "task_tracking": task_tracking_output,
        "quality_assurance": quality_assurance_output,
        "frontend_task": frontend_task_output,
        "backend_task": backend_task_output
    }

    return jsonify(response_data)