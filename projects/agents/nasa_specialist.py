from flask import Blueprint, request, jsonify
from config import config
from utils.request_helper import make_llama_request

nasa_bp = Blueprint('nasa_specialist', __name__)

SYSTEM_MESSAGE = "You are an aerospace physics and engineering expert with a focus on space technology. Given a question about spatial physics, rockets, or any other space-related engineering topic, provide comprehensive information, including the underlying physics, materials, design specifications, and any relevant code to simulate or model the problem."

@nasa_bp.route('/nasa-specialist', methods=['POST'])
def nasa_analysis():
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