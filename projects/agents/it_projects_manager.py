from flask import Blueprint, request, jsonify
from config import config
from utils.request_helper import make_llama_request

it_manager_bp = Blueprint('it_projects_manager', __name__)

SYSTEM_MESSAGE = "You are a project structure architect. Based on a received task and related code, generate an organized project repository structure. Include file organization, module separation, and documentation that aligns with best practices, ensuring scalability and maintainability of the project."

@it_manager_bp.route('/it-projects-manager', methods=['POST'])
def it_manager():
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