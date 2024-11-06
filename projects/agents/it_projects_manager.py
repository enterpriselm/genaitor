from flask import Blueprint, request, jsonify
from config import config
from utils.agents import Agent

it_manager_bp = Blueprint('it_projects_manager', __name__)

# Define individual agents for different aspects of project structure
file_organization_agent = Agent(
    role='File Organization Agent',
    system_message=(
        "You are an AI agent specializing in organizing project files. "
        "Given the project task and its codebase, you will recommend a folder structure that ensures scalability, maintainability, and ease of navigation."
    )
)

module_separation_agent = Agent(
    role='Module Separation Agent',
    system_message=(
        "You are an AI agent specializing in module separation within a project. "
        "Based on the provided project and code, you will suggest how to break the code into modules that follow best practices and ensure clarity and ease of development."
    )
)

documentation_agent = Agent(
    role='Documentation Agent',
    system_message=(
        "You are an AI agent specializing in generating project documentation. "
        "Given the project task and its code, you will generate detailed documentation that includes setup guides, usage instructions, and code descriptions, ensuring the project is easily understandable and maintainable."
    )
)

# Define the main route for IT project management
@it_manager_bp.route('/it-projects-manager', methods=['POST'])
def it_manager():
    data = request.get_json()

    # Validate user input
    user_query = data.get('user_query')
    if not user_query:
        return jsonify({"error": "User query is required"}), 400

    # Step 1: Organize project files
    file_organization_output = file_organization_agent.perform_task(user_query)
    if "error" in file_organization_output:
        return jsonify({"error": "File organization recommendation failed.", "details": file_organization_output}), 500

    # Step 2: Suggest module separation
    module_separation_output = module_separation_agent.perform_task(user_query)
    if "error" in module_separation_output:
        return jsonify({"error": "Module separation recommendation failed.", "details": module_separation_output}), 500

    # Step 3: Generate project documentation
    documentation_output = documentation_agent.perform_task(user_query)
    if "error" in documentation_output:
        return jsonify({"error": "Documentation generation failed.", "details": documentation_output}), 500

    # Format the final response with each agent's output
    response_data = {
        "file_organization": file_organization_output,
        "module_separation": module_separation_output,
        "documentation": documentation_output
    }

    return jsonify(response_data)