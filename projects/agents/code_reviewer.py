from flask import Blueprint, request, jsonify
from config import config
from utils.request_helper import make_llama_request

code_review_bp = Blueprint('code_reviewer', __name__)

SYSTEM_MESSAGE = "You are a code review specialist. Upon receiving a code snippet in Python, HTML, CSS, JavaScript, or React, analyze it carefully. Provide constructive feedback, focusing on best practices, potential optimizations, and necessary corrections, explaining specific recommendations and code examples where relevant."

@code_review_bp.route('/code-reviewer', methods=['POST'])
def review_code():
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
