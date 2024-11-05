from flask import Blueprint, request, jsonify
from config import config
from utils.request_helper import make_llama_request

cybersec_bp = Blueprint('cybersecurity_agent', __name__)

SYSTEM_MESSAGE = "You are a cybersecurity analyst skilled in both red and blue teaming. When provided with a security-related task, generate a detailed approach to identify vulnerabilities, implement security defenses, and outline the necessary code for solutions. Ensure your guidance covers common security best practices and methodologies."

@cybersec_bp.route('/cibersecurity-agent', methods=['POST'])
def cyber_analysis():
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
