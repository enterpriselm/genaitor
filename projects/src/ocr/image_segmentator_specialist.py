from flask import Blueprint, request, jsonify
from config import config
from utils.agents import Agent, Orchestrator

segmentation_bp = Blueprint('image_segmentator_specialist', __name__)

# Define agents for image segmentation tasks
agents = {
    'image_preprocessing_agent': Agent(
        role='Image Preprocessing Agent',
        system_message=(
            "You are an AI agent specialized in preprocessing images for segmentation tasks. "
            "Given an image, you will handle operations like resizing, normalization, and noise reduction to ensure it is ready for segmentation."
        )
    ),
    'segmentation_algorithm_agent': Agent(
        role='Segmentation Algorithm Agent',
        system_message=(
            "You are an AI agent specialized in performing image segmentation. "
            "Using libraries like OpenCV and Python, you will apply algorithms like thresholding, contour detection, and deep learning-based methods to segment the given image."
        )
    ),
    'postprocessing_agent': Agent(
        role='Postprocessing Agent',
        system_message=(
            "You are an AI agent specialized in postprocessing segmented images. "
            "You will handle tasks such as smoothing, contour extraction, or labeling regions based on segmentation results."
        )
    ),
}

# Define the task flow for image segmentation
tasks = [
    {"description": "Preprocess the image for segmentation", "agent": agents['image_preprocessing_agent']},
    {"description": "Apply segmentation algorithm", "agent": agents['segmentation_algorithm_agent']},
    {"description": "Postprocess the segmented image", "agent": agents['postprocessing_agent']}
]

# Initialize Orchestrator to manage the task flow
segmentation_orchestrator = Orchestrator(agents=agents, tasks=tasks, process='sequential', cumulative=False)

# Define the main route for image segmentation tasks
@segmentation_bp.route('/image-segmentator-specialist', methods=['POST'])
def segmentate():
    data = request.get_json()

    # Validate user input
    user_query = data.get('user_query')
    if not user_query:
        return jsonify({"error": "User query is required"}), 400

    # Use orchestrator to handle the pipeline for image segmentation
    result = segmentation_orchestrator.kickoff(user_query)

    if "error" in result:
        return jsonify(result), 500

    # Format the output from each task in the pipeline
    response_data = {task['description']: res for task, res in zip(tasks, result["output"])}

    return jsonify(response_data), 200
