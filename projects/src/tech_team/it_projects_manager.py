from flask import Blueprint, request, jsonify
from config import config
from utils.agents import Agent, Orchestrator

it_manager_bp = Blueprint('it_projects_manager', __name__)

# Define agents for different aspects of project structure
agents = {
    'file_organization_agent': Agent(
        role='File Organization Agent',
        system_message=(
            "You are an AI agent specializing in organizing project files. "
            "Given the project task and its codebase, you will recommend a folder structure that ensures scalability, maintainability, and ease of navigation."
        )
    ),
    'module_separation_agent': Agent(
        role='Module Separation Agent',
        system_message=(
            "You are an AI agent specializing in module separation within a project. "
            "Based on the provided project and code, you will suggest how to break the code into modules that follow best practices and ensure clarity and ease of development."
        )
    ),
    'documentation_agent': Agent(
        role='Documentation Agent',
        system_message=(
            "You are an AI agent specializing in generating project documentation. "
            "Given the project task and its code, you will generate detailed documentation that includes setup guides, usage instructions, and code descriptions, ensuring the project is easily understandable and maintainable."
        )
    ),
}

# Define the task flow for IT project management
tasks = [
    {"description": "Organize project files", "agent": agents['file_organization_agent']},
    {"description": "Suggest module separation", "agent": agents['module_separation_agent']},
    {"description": "Generate project documentation", "agent": agents['documentation_agent']},
]

# Initialize Orchestrator to manage the task flow
it_manager_orchestrator = Orchestrator(agents=agents, tasks=tasks, process='sequential', cumulative=False)

# Define the main route for IT project management
@it_manager_bp.route('/it-projects-manager', methods=['POST'])
def it_manager():
    data = request.get_json()

    # Validate user input
    user_query = data.get('user_query')
    if not user_query:
        return jsonify({"error": "User query is required"}), 400

    # Use orchestrator to handle the pipeline for project management
    result = it_manager_orchestrator.kickoff(user_query)

    if "error" in result:
        return jsonify(result), 500

    # Format the output from each task in the pipeline
    response_data = {task['description']: res for task, res in zip(tasks, result["output"])}

    return jsonify(response_data), 200
