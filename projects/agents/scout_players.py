from flask import Blueprint, request, jsonify
from config import config
from utils.request_helper import make_llama_request

scout_bp = Blueprint('scout_players', __name__)

SYSTEM_MESSAGE = "You are a highly skilled AI agent specialized in generating the best possible suggestions for players in a football game. You have access to a vast amount of data, including players' stats, market values, and team stats. You are able to analyze this data and provide the best possible suggestions for your user."

@scout_bp.route('/scout-players', methods=['POST'])
def scout_players():
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
