from flask import Blueprint, request, jsonify
from config import config
from utils.agents import Agent

segmentation_bp = Blueprint('image_segmentator_specialist', __name__)

# Define individual agents for image segmentation tasks
image_preprocessing_agent = Agent(
    role='Image Preprocessing Agent',
    system_message=(
        "You are an AI agent specialized in preprocessing images for segmentation tasks. "
        "Given an image, you will handle operations like resizing, normalization, and noise reduction to ensure it is ready for segmentation."
    )
)

segmentation_algorithm_agent = Agent(
    role='Segmentation Algorithm Agent',
    system_message=(
        "You are an AI agent specialized in performing image segmentation. "
        "Using libraries like OpenCV and Python, you will apply algorithms like thresholding, contour detection, and deep learning-based methods to segment the given image."
    )
)

postprocessing_agent = Agent(
    role='Postprocessing Agent',
    system_message=(
        "You are an AI agent specialized in postprocessing segmented images. "
        "You will handle tasks such as smoothing, contour extraction, or labeling regions based on segmentation results."
    )
)

# Define the main route for image segmentation tasks
@segmentation_bp.route('/image-segmentator-specialist', methods=['POST'])
def segmentate():
    data = request.get_json()

    # Validate user input
    user_query = data.get('user_query')
    if not user_query:
        return jsonify({"error": "User query is required"}), 400

    # Pipeline process
    # Step 1: Preprocess the image for segmentation
    preprocessing_output = image_preprocessing_agent.perform_task(user_query)
    if "error" in preprocessing_output:
        return jsonify({"error": "Image preprocessing failed.", "details": preprocessing_output}), 500

    # Step 2: Apply segmentation algorithm
    segmentation_output = segmentation_algorithm_agent.perform_task(user_query)
    if "error" in segmentation_output:
        return jsonify({"error": "Segmentation algorithm failed.", "details": segmentation_output}), 500

    # Step 3: Postprocess the segmented image
    postprocessing_output = postprocessing_agent.perform_task(user_query)
    if "error" in postprocessing_output:
        return jsonify({"error": "Postprocessing failed.", "details": postprocessing_output}), 500

    # Format the final response with each agent's output
    response_data = {
        "preprocessed_image": preprocessing_output,
        "segmented_image": segmentation_output,
        "postprocessed_image": postprocessing_output
    }

    return jsonify(response_data)