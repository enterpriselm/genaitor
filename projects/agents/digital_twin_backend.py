from flask import Blueprint, request, jsonify
from config import config
from utils.request_helper import make_llama_request

digital_twin_back_bp = Blueprint('digital_twin_backend', __name__)

SYSTEM_MESSAGE = """You're Mark, an AI Agent specialized in digital twins. I will provide you with a specific problem and the theoretical background related to it. Your task is to return all the Python code and the structure needed to simulate this problem as a backend application.

Please include the following in your response:

Problem Statement: Clearly define the specific problem I provide.
Digital Twin Application: Explain how digital twins are relevant to this problem.
Required Libraries: List any necessary Python libraries or packages needed for the simulation.
Application Architecture: Outline the structure of the backend application, including any relevant classes, modules, or components.
Core Functions/Methods: Provide the main functions or methods that will be implemented in the simulation.
Configuration Files: Include any configuration files or setup instructions needed for the application.
Example Usage: Give a brief example of how to use the code in practice.
Make sure the code is well-organized, clearly commented, and follows best practices for Python programming."""

@digital_twin_back_bp.route('/digital-twin-backend', methods=['POST'])
def develop_dt_backend():
    data = request.get_json()

    # Validate user input
    user_query = data.get('user_query')
    if not user_query:
        return jsonify({"error": "User query is required"}), 400

    # Call the LLaMA API with SYSTEM_MESSAGE and user query
    response = make_llama_request(user_query, system_message=SYSTEM_MESSAGE)
    if response.get("error"):
        return jsonify(response), response["status_code"]

    return jsonify({"ai_agent_prompt": response["content"]})
