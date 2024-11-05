from flask import Blueprint, request, jsonify
from config import config
from utils.request_helper import make_llama_request

answer_crafter_bp = Blueprint('answer_crafter', __name__)

SYSTEM_MESSAGE = (
    "You are an AI specializing in response validation and correction. "
    "When given an input and an output from a language model, assess the accuracy "
    "and relevance of the output. If the response is inaccurate or unclear, "
    "generate a revised, correct response that fully aligns with the input's requirements."
)

@answer_crafter_bp.route('/answers-crafter', methods=['POST'])
def generate_answer():
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
