from flask import Blueprint, request, jsonify
from utils.agents import Agent, Orchestrator

# Initialize Flask Blueprint
backend_bp = Blueprint('backend_developer', __name__)

# Define agents with their roles, system messages
agents = {
    'api_spec_generator': Agent(
        role='API Specification Generator',
        system_message=(
            "You are an expert API specification generator. Given the user's query, generate a detailed specification "
            "for the Flask API. This specification should include endpoints, request and response formats, error handling, "
            "and any necessary details that are required to implement the API."
        )
    ),
    'flask_api_developer': Agent(
        role='Flask API Developer',
        system_message=(
            "You are a skilled backend developer. Using the API specification provided, create the corresponding Flask "
            "API. Ensure that the code is clean, follows best practices, and implements all necessary endpoints and functionalities."
        )
    ),
    'testing_and_debugging_assistant': Agent(
        role='Testing and Debugging Assistant',
        system_message=(
            "You are an expert in testing and debugging backend applications. Review the Flask API created, generate "
            "test cases to validate its functionality, and ensure it is working properly. Provide any necessary debugging tips."
        )
    )
}

# Define the task flow for the backend development pipeline
tasks = [
    {"description": "Generate API specification", "agent": agents['api_spec_generator']},
    {"description": "Develop Flask API", "agent": agents['flask_api_developer']},
    {"description": "Test and debug Flask API", "agent": agents['testing_and_debugging_assistant']}
]

# Initialize the Orchestrator with agents, tasks, and process
backend_orchestrator = Orchestrator(agents=agents, tasks=tasks, process='sequential', cumulative=False)

@backend_bp.route('/backend-developer', methods=['POST'])
def develop_backend():
    data = request.get_json()

    # Validate user input
    user_query = data.get('user_query')
    if not user_query:
        return jsonify({"error": "User query is required"}), 400

    # Initiate the Orchestrator to handle the backend development workflow
    result = backend_orchestrator.kickoff(user_query)

    if "error" in result:
        return jsonify(result), 500

    # Extract outputs for each step in the process
    response_data = {task['description']: res for task, res in zip(tasks, result["output"])}

    return jsonify({
        "ai_agent_prompt": response_data
    }), 200
