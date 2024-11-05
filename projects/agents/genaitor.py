from flask import Blueprint, request, jsonify
from config import config
from utils.request_helper import make_llama_request

genaitor_bp = Blueprint('genaitor', __name__)

SYSTEM_MESSAGE = """You are an AI agent specialized in generating new AI agents. 
The user should require something, and you should pass the best prompt for passing through the LLM 
to attend the necessities as an AI agent. Example: {example}"""

EXAMPLE = {
    "model": "LLaMA_CPP",
    "messages": [
        {
            "role": "system",
            "content": "You are the best cooking chef in the World. You know all the recipes and how to make the best food."
        }
    ]
}

SYSTEM_MESSAGE.format(example=EXAMPLE)

@genaitor_bp.route('/genaitor', methods=['POST'])
def imaginate():
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
