from flask import Blueprint, request, jsonify
from config import config
from utils.agents import Agent

automatize_task_bp = Blueprint('automatic_pc', __name__)

project_structure_analyst = Agent(
    role='Project Structure Analyst',
    system_message=(
        "You are a skilled Project Structure Analyst. Given the user's query about automating project setup, "
        "analyze the project structure and identify all components, directories, and configurations that "
        "need to be automated for a smooth setup and deployment."
    )
)

dependencies_manager = Agent(
    role='Dependencies and Configuration Manager',
    system_message=(
        "You are an expert Dependencies and Configuration Manager. Based on the project structure, "
        "identify all necessary dependencies, configuration files, and setup instructions that need to be "
        "included for the project to run successfully."
    )
)

deployment_planner = Agent(
    role='Deployment Strategy Planner',
    system_message=(
        "You are a Deployment Strategy Planner. Based on the project structure and dependencies, "
        "generate detailed deployment instructions, including installation steps, configurations, and "
        "any scripts necessary to deploy and run the project."
    )
)

# Define the main route that processes the user's query
@automatize_task_bp.route('/automatic-pc', methods=['POST'])
def automatize_task():
    data = request.get_json()

    # Validate user input
    user_query = data.get('user_query')
    if not user_query:
        return jsonify({"error": "User query is required"}), 400

    # Pipeline process
    structure_output = project_structure_analyst.perform_task(user_query)
    if "error" in structure_output:
        return jsonify({"error": "Project structure analysis failed.", "details": structure_output}), 500

    dependencies_output = dependencies_manager.perform_task(structure_output)
    if "error" in dependencies_output:
        return jsonify({"error": "Dependencies and configuration management failed.", "details": dependencies_output}), 500

    deployment_output = deployment_planner.perform_task(dependencies_output)
    if "error" in deployment_output:
        return jsonify({"error": "Deployment strategy planning failed.", "details": deployment_output}), 500

    # Format the final response with each step's output
    response_data = {
        "project_structure_analysis": structure_output,
        "dependencies_and_configurations": dependencies_output,
        "deployment_instructions": deployment_output
    }

    return jsonify({"ai_agent_prompt": response_data})