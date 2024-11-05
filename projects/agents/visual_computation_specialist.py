from flask import Blueprint, request, jsonify
from config import config
from utils.request_helper import make_llama_request

visual_computing_bp = Blueprint('visual_computation_specialist', __name__)

SYSTEM_MESSAGE = "You are an AI agent specialized in visual computation. You can solve problems related to image processing, computer vision, and deep learning. Please provide the problem you'd like to solve, and I'll provide the Python code to help you."

@visual_computing_bp.route('/visual-computation-specialist', methods=['POST'])
def seeing():
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
1