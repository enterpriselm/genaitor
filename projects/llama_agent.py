from flask import Blueprint, request, jsonify
from config import config
from utils.request_helper import make_llama_request

llama_agent_bp = Blueprint('llama_agent', __name__)

@llama_agent_bp.route('/generate-agent', methods=['POST'])
def generate_agent():
    """Generate an AI agent response based on the user query."""
    data = request.get_json()
    user_query = data.get('user_query')
    if not user_query:
        return jsonify({"error": "User query is required"}), 400

    response = make_llama_request(user_query)
    if response.get("error"):
        return jsonify(response), response["status_code"]
    
    return jsonify({"ai_agent_prompt": response["content"]})
