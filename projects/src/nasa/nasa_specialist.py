from flask import Blueprint, request, jsonify
from config import config
from utils.agents import Agent, Orchestrator

nasa_bp = Blueprint('nasa_specialist', __name__)

# Define agents for different aspects of space engineering
agents = {
    'rocket_design_agent': Agent(
        role='Rocket Design Agent',
        system_message=(
            "You are an AI agent specialized in rocket design. "
            "Given the user's query, you will provide detailed insights on the materials, physics, and specifications required for rocket propulsion, launch systems, and stability."
        )
    ),
    'materials_engineering_agent': Agent(
        role='Materials Engineering Agent',
        system_message=(
            "You are an AI agent specializing in aerospace materials. "
            "For any space-related engineering task, you will suggest the appropriate materials for construction, taking into account factors like heat resistance, weight, and durability."
        )
    ),
    'space_physics_agent': Agent(
        role='Space Physics Agent',
        system_message=(
            "You are an AI agent specialized in space physics. "
            "Given a problem involving orbital mechanics, propulsion systems, or any other physics-related space problem, you will provide the underlying physics and the necessary equations to simulate or model the problem."
        )
    ),
    'simulation_agent': Agent(
        role='Simulation Agent',
        system_message=(
            "You are an AI agent specializing in simulating space-related problems. "
            "For any space-related query, you will provide relevant code to simulate or model the problem, ensuring accuracy and alignment with the underlying physics."
        )
    ),
}

# Define the task flow for NASA-related analysis
tasks = [
    {"description": "Rocket design insights", "agent": agents['rocket_design_agent']},
    {"description": "Materials engineering recommendations", "agent": agents['materials_engineering_agent']},
    {"description": "Space physics insights", "agent": agents['space_physics_agent']},
    {"description": "Simulation code generation", "agent": agents['simulation_agent']},
]

# Initialize Orchestrator to manage the task flow
nasa_orchestrator = Orchestrator(agents=agents, tasks=tasks, process='sequential', cumulative=False)

# Define the main route for NASA-related analysis
@nasa_bp.route('/nasa-specialist', methods=['POST'])
def nasa_analysis():
    data = request.get_json()

    # Validate user input
    user_query = data.get('user_query')
    if not user_query:
        return jsonify({"error": "User query is required"}), 400

    # Use orchestrator to handle the pipeline for NASA-related analysis
    result = nasa_orchestrator.kickoff(user_query)

    if "error" in result:
        return jsonify(result), 500

    # Format the output from each task in the pipeline
    response_data = {task['description']: res for task, res in zip(tasks, result["output"])}

    return jsonify(response_data), 200
