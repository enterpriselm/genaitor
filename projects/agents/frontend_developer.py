from flask import Blueprint, request, jsonify
from config import config
from utils.request_helper import make_llama_request

frontend_bp = Blueprint('frontend_developer', __name__)

SYSTEM_MESSAGE = "You are a skilled frontend developer proficient in HTML, CSS, and JavaScript. You have experience working with React and are adept at receiving tasks and references and producing high-quality code."

@frontend_bp.route('/frontend-developer', methods=['POST'])
def develop_frontend():
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
