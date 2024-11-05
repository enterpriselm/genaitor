from flask import Blueprint, request, jsonify
from config import config
from utils.request_helper import make_llama_request

backend_bp = Blueprint('backend_developer', __name__)

SYSTEM_MESSAGE = "You are a highly skilled backend developer proficient in Python. You have experience creating Flask APIs and can easily follow project guidelines. Please create a Flask API for the task provided."

@backend_bp.route('/backend-developer', methods=['POST'])
def develop_backend():
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
