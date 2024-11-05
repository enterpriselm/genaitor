from flask import Blueprint, request, jsonify
from config import config
from utils.request_helper import make_llama_request

digital_twin_front_bp = Blueprint('digital_twin_frontend', __name__)

SYSTEM_MESSAGE = "You are a digital twins visualization expert. Given a problem and its backend solution, suggest an advanced visualization approach. Provide code for APIs, 3D visualizations, and any relevant data mapping. Your visualizations should be interactive, accurate to the model's physics, and use cutting-edge 3D visualization libraries. Additionally, integrate physics-informed neural networks if relevant to the model."

@digital_twin_front_bp.route('/digital-twin-frontend', methods=['POST'])
def develop_dt_frontend():
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
