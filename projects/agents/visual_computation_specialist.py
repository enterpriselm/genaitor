from flask import Blueprint, request, jsonify
from config import config
from utils.agents import Agent

visual_computing_bp = Blueprint('visual_computation_specialist', __name__)

# Define individual agents for different visual computation tasks
image_processing_agent = Agent(
    role='Image Processing Agent',
    system_message=(
        "You are an AI agent specialized in image processing. "
        "Your task is to analyze and process images to solve problems like noise reduction, enhancement, segmentation, etc. "
        "Please provide the necessary image processing code."
    )
)

computer_vision_agent = Agent(
    role='Computer Vision Agent',
    system_message=(
        "You are an AI agent specialized in computer vision tasks. "
        "Your task is to analyze visual data and apply techniques like object detection, face recognition, and feature extraction. "
        "Provide the required computer vision solution using the relevant deep learning models."
    )
)

deep_learning_agent = Agent(
    role='Deep Learning Agent',
    system_message=(
        "You are an AI agent specializing in deep learning for visual computation. "
        "Your task is to develop and implement deep learning models for image classification, object detection, semantic segmentation, etc. "
        "Provide the necessary deep learning model code, including training and evaluation."
    )
)

# Define the main route for Visual Computation Specialist
@visual_computing_bp.route('/visual-computation-specialist', methods=['POST'])
def seeing():
    data = request.get_json()

    # Validate user input
    user_query = data.get('user_query')
    if not user_query:
        return jsonify({"error": "User query is required"}), 400

    # Step 1: Process the image (if relevant)
    image_processing_output = image_processing_agent.perform_task(user_query)
    if "error" in image_processing_output:
        return jsonify({"error": "Image processing failed.", "details": image_processing_output}), 500

    # Step 2: Apply computer vision techniques (if relevant)
    computer_vision_output = computer_vision_agent.perform_task(user_query)
    if "error" in computer_vision_output:
        return jsonify({"error": "Computer vision task failed.", "details": computer_vision_output}), 500

    # Step 3: Implement deep learning model (if relevant)
    deep_learning_output = deep_learning_agent.perform_task(user_query)
    if "error" in deep_learning_output:
        return jsonify({"error": "Deep learning model failed.", "details": deep_learning_output}), 500

    # Format the final response with each agent's output
    response_data = {
        "image_processing": image_processing_output,
        "computer_vision": computer_vision_output,
        "deep_learning": deep_learning_output
    }

    return jsonify(response_data)