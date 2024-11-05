from flask import Blueprint, request, jsonify
from config import config
from utils.request_helper import make_llama_request

apps_builder_bp = Blueprint('apps_builder', __name__)

SYSTEM_MESSAGE = (
    "You are a skilled software developer with expertise in React and Python. "
    "You can build high-quality apps with ease."
)

@apps_builder_bp.route('/apps-builder', methods=['POST'])
def build_app():
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
