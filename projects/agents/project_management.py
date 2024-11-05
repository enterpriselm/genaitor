from flask import Blueprint, request, jsonify
from config import config
from utils.request_helper import make_llama_request

pm_bp = Blueprint('project_management', __name__)

SYSTEM_MESSAGE = "You are an AI agent specialized in project management. The user requires an AI agent to work as a project management specialist. You will receive a task and pass the task for frontend and backend pipelines. You will also check if the tasks have been completed and provide another task if necessary."

@pm_bp.route('/project-management', methods=['POST'])
def project_management():
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
