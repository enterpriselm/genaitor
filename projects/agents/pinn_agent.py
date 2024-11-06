from flask import Blueprint, request, jsonify
from config import config
from utils.agents import Agent, Orchestrator

pinn_bp = Blueprint('pinn_agent', __name__)

# Define agents for different aspects of PINN modeling
agents = {
    'model_definition_agent': Agent(
        role='Model Definition Agent',
        system_message=(
            "You are an AI agent specialized in defining neural network architectures for Physics Informed Neural Networks (PINNs). "
            "Given the user's query, you will suggest the appropriate neural network architecture for solving the given physical problem."
        )
    ),
    'physics_equation_agent': Agent(
        role='Physics Equation Agent',
        system_message=(
            "You are an AI agent specialized in formulating physics-based equations for PINNs. "
            "Given the user's query, you will extract the necessary physical equations that need to be embedded in the neural network."
        )
    ),
    'loss_function_agent': Agent(
        role='Loss Function Agent',
        system_message=(
            "You are an AI agent specialized in defining loss functions for Physics Informed Neural Networks (PINNs). "
            "For the user's query, you will design the appropriate loss function to incorporate the physical constraints into the training process."
        )
    ),
    'training_agent': Agent(
        role='Training Agent',
        system_message=(
            "You are an AI agent specialized in setting up and executing the training process for PINNs. "
            "Given the user's query, you will define the necessary training loop, optimization, and performance evaluation metrics for the neural network."
        )
    ),
}

# Define the task flow for PINN analysis
tasks = [
    {"description": "Neural network architecture definition", "agent": agents['model_definition_agent']},
    {"description": "Physics equations extraction", "agent": agents['physics_equation_agent']},
    {"description": "Loss function definition", "agent": agents['loss_function_agent']},
    {"description": "Training setup for the PINN", "agent": agents['training_agent']},
]

# Initialize Orchestrator to manage the task flow
pinn_orchestrator = Orchestrator(agents=agents, tasks=tasks, process='sequential', cumulative=False)

# Define the main route for PINN analysis
@pinn_bp.route('/pinn-agent', methods=['POST'])
def pinn_analysis():
    data = request.get_json()

    # Validate user input
    user_query = data.get('user_query')
    if not user_query:
        return jsonify({"error": "User query is required"}), 400

    # Use orchestrator to handle the pipeline for PINN analysis
    result = pinn_orchestrator.kickoff(user_query)

    if "error" in result:
        return jsonify(result), 500

    # Format the output from each task in the pipeline
    response_data = {task['description']: res for task, res in zip(tasks, result["output"])}

    return jsonify(response_data), 200
