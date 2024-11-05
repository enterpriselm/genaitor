from flask import Blueprint, request, jsonify
from config import config
from utils.request_helper import make_llama_request

scraper_bp = Blueprint('scraper_agent', __name__)

SYSTEM_MESSAGE = "You are a web scraping expert. Upon receiving an HTML page and a data extraction task, develop an efficient scraping strategy. Provide the necessary code, troubleshoot any issues with your implementation, and ensure compliance with web scraping best practices and legal considerations."

@scraper_bp.route('/scraper-agent', methods=['POST'])
def generate_scraper():
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
