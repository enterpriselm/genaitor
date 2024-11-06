from flask import Blueprint, request, jsonify
from utils.agents import Agent, Orchestrator

# Initialize Flask Blueprint for digital twin frontend
digital_twin_front_bp = Blueprint('digital_twin_frontend', __name__)

# Define agents with their roles and system messages
agents = {
    'visualization_expert': Agent(
        role='Visualization Expert',
        system_message=(
            "You are an expert in digital twin visualization. Given a problem and its backend solution, suggest the most suitable "
            "visualization approach. Focus on interactive, physics-accurate, 3D visualizations. Include detailed code examples for "
            "APIs, 3D visualizations, and data mapping. Use cutting-edge libraries such as Three.js or similar, and if applicable, "
            "integrate physics-informed neural networks into the visualizations."
        )
    ),
    'interactivity_specialist': Agent(
        role='Interactivity Specialist',
        system_message=(
            "You specialize in interactive visualizations. Given the backend solution and problem context, propose interactive features "
            "that users can engage with. Your goal is to design intuitive and fluid interactions for visualizing the model. "
            "Focus on user-friendly API integration and frontend interactivity, especially for 3D environments."
        )
    ),
    'physics_integration_expert': Agent(
        role='Physics Integration Expert',
        system_message=(
            "You are a physics integration expert for digital twin visualizations. Given the problem and backend solution, suggest "
            "ways to incorporate physics-informed neural networks (PINNs) or similar methods into the visualization. Ensure that the "
            "visualizations reflect the physics of the system, and that the 3D environment can simulate real-world physics accurately."
        )
    ),
}

# Define the task flow for the digital twin frontend application
tasks = [
    {"description": "Suggest suitable visualization approach", "agent": agents['visualization_expert']},
    {"description": "Propose interactive features", "agent": agents['interactivity_specialist']},
    {"description": "Suggest physics-informed approaches", "agent": agents['physics_integration_expert']},
]

# Initialize Orchestrator to handle task execution
digital_twin_orchestrator = Orchestrator(agents=agents, tasks=tasks, process='sequential', cumulative=False)

# Define the main route that processes the user's query
@digital_twin_front_bp.route('/digital-twin-frontend', methods=['POST'])
def develop_dt_frontend():
    data = request.get_json()

    # Validate user input
    user_query = data.get('user_query')
    if not user_query:
        return jsonify({"error": "User query is required"}), 400

    # Use orchestrator to handle the workflow for the digital twin frontend development
    result = digital_twin_orchestrator.kickoff(user_query)

    if "error" in result:
        return jsonify(result), 500

    # Format the output from each task in the pipeline
    response_data = {task['description']: res for task, res in zip(tasks, result["output"])}

    return jsonify({
        "ai_agent_prompt": response_data
    }), 200
