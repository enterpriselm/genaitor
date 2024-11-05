from flask import Blueprint, request, jsonify
from config import config
from utils.request_helper import make_llama_request

automatize_task_bp = Blueprint('automatic_pc', __name__)

SYSTEM_MESSAGE = "You are a project automation assistant. Given a project structure and corresponding code, generate Python scripts that automate the creation and configuration of the entire project. Include all required dependencies, configurations, and any setup instructions for easy deployment."

@automatize_task_bp.route('/automatic-pc', methods=['POST'])
def automatize_task():
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
