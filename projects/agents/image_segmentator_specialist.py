from flask import Blueprint, request, jsonify
from config import config
from utils.request_helper import make_llama_request

segmentation_bp = Blueprint('image_segmentator_specialist', __name__)

SYSTEM_MESSAGE = "You're an AI agent specialized in image segmentation using Python and OpenCV. I will receive an image array and return the segmented image based on your requirements.'"

@segmentation_bp.route('/image-segmentator-specialist', methods=['POST'])
def segmentate():
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
