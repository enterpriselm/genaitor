from flask import Blueprint, request, jsonify
from config import config
from utils.request_helper import make_llama_request

pinn_bp = Blueprint('pinn_agent', __name__)

SYSTEM_MESSAGE = "From now on, you are an AI agent specialized in Physics Informed Neural Networks. I will present a physical problem, and you will show me the PyTorch modeling to solve it."

@pinn_bp.route('/pinn-agent', methods=['POST'])
def imagine_pinn():
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
