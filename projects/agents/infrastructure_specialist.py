from flask import Blueprint, request, jsonify
from config import config
from utils.request_helper import make_llama_request

infra_bp = Blueprint('infrastructure_specialist', __name__)

SYSTEM_MESSAGE = "You are an IT infrastructure expert. Upon receiving a project with its codebase, design a robust, scalable, and secure infrastructure setup. This should include cloud services, server architecture, load balancing, database setup, and necessary security configurations. Provide code snippets, configuration files, and setup guides for ease of deployment."

@infra_bp.route('/infrastructure_specialist', methods=['POST'])
def get_infra():
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
