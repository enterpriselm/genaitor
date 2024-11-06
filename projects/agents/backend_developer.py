from flask import Blueprint, request, jsonify
from config import config
from utils.agents import Agent

backend_bp = Blueprint('backend_developer', __name__)

api_spec_generator = Agent(
    role='API Specification Generator',
    system_message=(
        "You are an expert API specification generator. Given the user's query, generate a detailed specification "
        "for the Flask API. This specification should include endpoints, request and response formats, error handling, "
        "and any necessary details that are required to implement the API."
    )
)

flask_api_developer = Agent(
    role='Flask API Developer',
    system_message=(
        "You are a skilled backend developer. Using the API specification provided, create the corresponding Flask "
        "API. Ensure that the code is clean, follows best practices, and implements all necessary endpoints and functionalities."
    )
)

testing_and_debugging_assistant = Agent(
    role='Testing and Debugging Assistant',
    system_message=(
        "You are an expert in testing and debugging backend applications. Review the Flask API created, generate "
        "test cases to validate its functionality, and ensure it is working properly. Provide any necessary debugging tips."
    )
)

# Define the main route that processes the user's query
@backend_bp.route('/backend-developer', methods=['POST'])
def develop_backend():
    data = request.get_json()

    # Validate user input
    user_query = data.get('user_query')
    if not user_query:
        return jsonify({"error": "User query is required"}), 400

    # Pipeline process
    # Step 1: Generate API Specification
    specification_output = api_spec_generator.perform_task(user_query)
    if "error" in specification_output:
        return jsonify({"error": "API specification generation failed.", "details": specification_output}), 500

    # Step 2: Develop Flask API
    api_code_output = flask_api_developer.perform_task(specification_output)
    if "error" in api_code_output:
        return jsonify({"error": "Flask API development failed.", "details": api_code_output}), 500

    # Step 3: Test and Debug
    test_cases_output = testing_and_debugging_assistant.perform_task(api_code_output)
    if "error" in test_cases_output:
        return jsonify({"error": "Testing and debugging failed.", "details": test_cases_output}), 500

    # Format the final response with each step's output
    response_data = {
        "api_specification": specification_output,
        "flask_api_code": api_code_output,
        "testing_and_debugging": test_cases_output
    }

    return jsonify({"ai_agent_prompt": response_data})
