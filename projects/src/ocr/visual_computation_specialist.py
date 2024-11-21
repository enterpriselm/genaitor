from flask import Blueprint, request, jsonify
from config import config
from utils.agents import Agent, Orchestrator

visual_computing_bp = Blueprint('visual_computation_specialist', __name__)

# Define agents for different visual computation tasks
agents = {
    'image_processing_agent': Agent(
        role='Image Processing Agent',
        system_message=(
            "You are an AI agent specialized in image processing. "
            "Your task is to analyze and process images to solve problems like noise reduction, enhancement, segmentation, etc. "
            "Please provide the necessary image processing code."
        )
    ),
    'computer_vision_agent': Agent(
        role='Computer Vision Agent',
        system_message=(
            "You are an AI agent specialized in computer vision tasks. "
            "Your task is to analyze visual data and apply techniques like object detection, face recognition, and feature extraction. "
            "Provide the required computer vision solution using the relevant deep learning models."
        )
    ),
    'deep_learning_agent': Agent(
        role='Deep Learning Agent',
        system_message=(
            "You are an AI agent specializing in deep learning for visual computation. "
            "Your task is to develop and implement deep learning models for image classification, object detection, semantic segmentation, etc. "
            "Provide the necessary deep learning model code, including training and evaluation."
        )
    ),
}

# Define the task flow for Visual Computation process
tasks = [
    {"description": "Image processing", "agent": agents['image_processing_agent']},
    {"description": "Computer vision task", "agent": agents['computer_vision_agent']},
    {"description": "Deep learning model", "agent": agents['deep_learning_agent']},
]

# Initialize Orchestrator to manage the task flow
visual_computation_orchestrator = Orchestrator(agents=agents, tasks=tasks, process='sequential', cumulative=False)

# Define the main route for Visual Computation Specialist
@visual_computing_bp.route('/visual-computation-specialist', methods=['POST'])
def seeing():
    data = request.get_json()

    # Validate user input
    user_query = data.get('user_query')
    if not user_query:
        return jsonify({"error": "User query is required"}), 400

    # Use orchestrator to handle the pipeline for visual computation tasks
    result = visual_computation_orchestrator.kickoff(user_query)

    if "error" in result:
        return jsonify(result), 500

    # Format the output from each task in the pipeline
    response_data = {task['description']: res for task, res in zip(tasks, result["output"])}

    return jsonify(response_data), 200
