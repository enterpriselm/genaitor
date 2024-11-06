from flask import Blueprint, request, jsonify
from config import config
from utils.agents import Agent

digital_twin_front_bp = Blueprint('digital_twin_frontend', __name__)

visualization_expert = Agent(
    role='Visualization Expert',
    system_message=(
        "You are an expert in digital twin visualization. Given a problem and its backend solution, suggest the most suitable "
        "visualization approach. Focus on interactive, physics-accurate, 3D visualizations. Include detailed code examples for "
        "APIs, 3D visualizations, and data mapping. Use cutting-edge libraries such as Three.js or similar, and if applicable, "
        "integrate physics-informed neural networks into the visualizations."
    )
)

interactivity_specialist = Agent(
    role='Interactivity Specialist',
    system_message=(
        "You specialize in interactive visualizations. Given the backend solution and problem context, propose interactive features "
        "that users can engage with. Your goal is to design intuitive and fluid interactions for visualizing the model. "
        "Focus on user-friendly API integration and frontend interactivity, especially for 3D environments."
    )
)

physics_integration_expert = Agent(
    role='Physics Integration Expert',
    system_message=(
        "You are a physics integration expert for digital twin visualizations. Given the problem and backend solution, suggest "
        "ways to incorporate physics-informed neural networks (PINNs) or similar methods into the visualization. Ensure that the "
        "visualizations reflect the physics of the system, and that the 3D environment can simulate real-world physics accurately."
    )
)

# Define the main route that processes the user's query
@digital_twin_front_bp.route('/digital-twin-frontend', methods=['POST'])
def develop_dt_frontend():
    data = request.get_json()

    # Validate user input
    user_query = data.get('user_query')
    if not user_query:
        return jsonify({"error": "User query is required"}), 400

    # Pipeline process
    # Step 1: Visualization Expert suggests a suitable visualization approach
    visualization_output = visualization_expert.perform_task(user_query)
    if "error" in visualization_output:
        return jsonify({"error": "Visualization suggestion failed.", "details": visualization_output}), 500

    # Step 2: Interactivity Specialist proposes interactive features
    interactivity_output = interactivity_specialist.perform_task(user_query)
    if "error" in interactivity_output:
        return jsonify({"error": "Interactivity design failed.", "details": interactivity_output}), 500

    # Step 3: Physics Integration Expert suggests physics-informed approaches
    physics_output = physics_integration_expert.perform_task(user_query)
    if "error" in physics_output:
        return jsonify({"error": "Physics integration failed.", "details": physics_output}), 500

    # Format the final response with each agent's output
    response_data = {
        "visualization_suggestions": visualization_output,
        "interactivity_suggestions": interactivity_output,
        "physics_integration": physics_output
    }

    return jsonify({"ai_agent_prompt": response_data})