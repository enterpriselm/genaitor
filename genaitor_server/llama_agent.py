from flask import Blueprint, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from genaitor.config import config
from genaitor.request_helper import make_llama_request
from genaitor.clean_response import clean_genaitor_response
from marshmallow import Schema, fields, ValidationError

# Security: Setup rate limiter
limiter = Limiter(key_func=get_remote_address)

class GenerateAgentSchema(Schema):
    user_query = fields.Str(required=True)

llama_agent_bp = Blueprint('llama_agent', __name__)
limiter.limit("10/minute")(llama_agent_bp)

system_message = """You're Genaitor, an AI agent expert in creating new AI Agents. 
Gimme the prompt for an agent based on this {user_query}. 
Prompt: """

@llama_agent_bp.route('/generate-agent', methods=['POST'])
def generate_agent():
    """Generate an AI agent response based on the user query."""
    try:
        # Validate request data
        data = request.get_json()
        validated_data = GenerateAgentSchema().load(data)
        user_query = validated_data['user_query']
    except ValidationError as e:
        return jsonify({"error": "Invalid input", "details": e.messages}), 400
    
    try:
        # Make request to AI model
        response = make_llama_request(
            user_query, 
            system_message=system_message.format(user_query=user_query), 
            max_tokens=250, 
            max_iterations=1
        )
        response = clean_genaitor_response(response)
        if response.get("error"):
            return jsonify(response), response["status_code"]
    
        return jsonify({"ai_agent_prompt": response["content"]})

    except Exception as e:
        return jsonify({"error": "Internal server error"}), 500