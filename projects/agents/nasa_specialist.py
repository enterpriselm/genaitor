from flask import Blueprint, request, jsonify
from config import config
from utils.agents import Agent

nasa_bp = Blueprint('nasa_specialist', __name__)

# Define individual agents for different aspects of space engineering
rocket_design_agent = Agent(
    role='Rocket Design Agent',
    system_message=(
        "You are an AI agent specialized in rocket design. "
        "Given the user's query, you will provide detailed insights on the materials, physics, and specifications required for rocket propulsion, launch systems, and stability."
    )
)

materials_engineering_agent = Agent(
    role='Materials Engineering Agent',
    system_message=(
        "You are an AI agent specializing in aerospace materials. "
        "For any space-related engineering task, you will suggest the appropriate materials for construction, taking into account factors like heat resistance, weight, and durability."
    )
)

space_physics_agent = Agent(
    role='Space Physics Agent',
    system_message=(
        "You are an AI agent specialized in space physics. "
        "Given a problem involving orbital mechanics, propulsion systems, or any other physics-related space problem, you will provide the underlying physics and the necessary equations to simulate or model the problem."
    )
)

simulation_agent = Agent(
    role='Simulation Agent',
    system_message=(
        "You are an AI agent specializing in simulating space-related problems. "
        "For any space-related query, you will provide relevant code to simulate or model the problem, ensuring accuracy and alignment with the underlying physics."
    )
)

# Define the main route for NASA-related analysis
@nasa_bp.route('/nasa-specialist', methods=['POST'])
def nasa_analysis():
    data = request.get_json()

    # Validate user input
    user_query = data.get('user_query')
    if not user_query:
        return jsonify({"error": "User query is required"}), 400

    # Step 1: Rocket design insights
    rocket_design_output = rocket_design_agent.perform_task(user_query)
    if "error" in rocket_design_output:
        return jsonify({"error": "Rocket design analysis failed.", "details": rocket_design_output}), 500

    # Step 2: Materials engineering recommendations
    materials_engineering_output = materials_engineering_agent.perform_task(user_query)
    if "error" in materials_engineering_output:
        return jsonify({"error": "Materials engineering analysis failed.", "details": materials_engineering_output}), 500

    # Step 3: Space physics insights
    space_physics_output = space_physics_agent.perform_task(user_query)
    if "error" in space_physics_output:
        return jsonify({"error": "Space physics analysis failed.", "details": space_physics_output}), 500

    # Step 4: Simulation code generation
    simulation_output = simulation_agent.perform_task(user_query)
    if "error" in simulation_output:
        return jsonify({"error": "Simulation code generation failed.", "details": simulation_output}), 500

    # Format the final response with each agent's output
    response_data = {
        "rocket_design": rocket_design_output,
        "materials_engineering": materials_engineering_output,
        "space_physics": space_physics_output,
        "simulation_code": simulation_output
    }

    return jsonify(response_data)