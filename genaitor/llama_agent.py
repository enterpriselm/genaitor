from flask import Blueprint, request, jsonify
from genaitor.config import config
from genaitor.request_helper import make_llama_request
from genaitor.clean_response import clean_genaitor_response

system_message = """You're Genaitor, an AI agent expert in creating new AI Agents. 
Gimme the prompt for an agent based on this {user_query}. 
Prompt: """

llama_agent_bp = Blueprint('llama_agent', __name__)

@llama_agent_bp.route('/generate-agent', methods=['POST'])
def generate_agent():
    """Generate an AI agent response based on the user query."""
    data = request.get_json()
    user_query = data.get('user_query')
    if not user_query:
        return jsonify({"error": "User query is required"}), 400

    response = make_llama_request(user_query, system_message=system_message.format(user_query=user_query), max_tokens=250, max_iterations=1)
    response = clean_genaitor_response(response)
    if response.get("error"):
        return jsonify(response), response["status_code"]
    
    return jsonify({"ai_agent_prompt": response["content"]})
