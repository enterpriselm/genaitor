from flask import Blueprint, request, jsonify
from utils.agents import Agent, Orchestrator

# Initialize Flask Blueprint
automatize_task_bp = Blueprint('automatic_pc', __name__)

# Define agents with their roles, system messages
agents = {
    'project_structure_analyst': Agent(
        role='Project Structure Analyst',
        system_message=(
            "You are a skilled Project Structure Analyst. Given the user's query about automating project setup, "
            "analyze the project structure and identify all components, directories, and configurations that "
            "need to be automated for a smooth setup and deployment."
        )
    ),
    'dependencies_manager': Agent(
        role='Dependencies and Configuration Manager',
        system_message=(
            "You are an expert Dependencies and Configuration Manager. Based on the project structure, "
            "identify all necessary dependencies, configuration files, and setup instructions that need to be "
            "included for the project to run successfully."
        )
    ),
    'deployment_planner': Agent(
        role='Deployment Strategy Planner',
        system_message=(
            "You are a Deployment Strategy Planner. Based on the project structure and dependencies, "
            "generate detailed deployment instructions, including installation steps, configurations, and "
            "any scripts necessary to deploy and run the project."
        )
    )
}

# Define the task flow for the project automation pipeline
tasks = [
    {"description": "Analyze project structure", "agent": agents['project_structure_analyst']},
    {"description": "Manage dependencies and configurations", "agent": agents['dependencies_manager']},
    {"description": "Plan deployment strategy", "agent": agents['deployment_planner']}
]

# Initialize the Orchestrator with agents, tasks, and process
automation_orchestrator = Orchestrator(agents=agents, tasks=tasks, process='sequential', cumulative=False)

@automatize_task_bp.route('/automatic-pc', methods=['POST'])
def automatize_task():
    data = request.get_json()

    # Validate user input
    user_query = data.get('user_query')
    if not user_query:
        return jsonify({"error": "User query is required"}), 400

    # Initiate the Orchestrator to handle the project automation workflow
    result = automation_orchestrator.kickoff(user_query)

    if "error" in result:
        return jsonify(result), 500

    # Extract outputs for each step in the process
    response_data = {task['description']: res for task, res in zip(tasks, result["output"])}

    return jsonify({
        "ai_agent_prompt": response_data
    }), 200
