from flask import Blueprint, request, jsonify
from config import config
from utils.request_helper import make_llama_request

paper_summarize_bp = Blueprint('papers_analyst', __name__)

SYSTEM_MESSAGE = "You are a highly skilled AI agent trained to summarize scientific papers. You have expertise in Physics Informed Neural Networks and can provide a concise summary of a paper on this topic."

@paper_summarize_bp.route('/papers_analyst', methods=['POST'])
def papers_summarize():
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
