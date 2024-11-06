from flask import Blueprint, request, jsonify
from config import config
from utils.agents import Agent

pinn_bp = Blueprint('pinn_agent', __name__)

# Define individual agents for different aspects of PINN modeling
model_definition_agent = Agent(
    role='Model Definition Agent',
    system_message=(
        "You are an AI agent specialized in defining neural network architectures for Physics Informed Neural Networks (PINNs). "
        "Given the user's query, you will suggest the appropriate neural network architecture for solving the given physical problem."
    )
)

physics_equation_agent = Agent(
    role='Physics Equation Agent',
    system_message=(
        "You are an AI agent specialized in formulating physics-based equations for PINNs. "
        "Given the user's query, you will extract the necessary physical equations that need to be embedded in the neural network."
    )
)

loss_function_agent = Agent(
    role='Loss Function Agent',
    system_message=(
        "You are an AI agent specialized in defining loss functions for Physics Informed Neural Networks (PINNs). "
        "For the user's query, you will design the appropriate loss function to incorporate the physical constraints into the training process."
    )
)

training_agent = Agent(
    role='Training Agent',
    system_message=(
        "You are an AI agent specialized in setting up and executing the training process for PINNs. "
        "Given the user's query, you will define the necessary training loop, optimization, and performance evaluation metrics for the neural network."
    )
)

# Define the main route for PINN analysis
@pinn_bp.route('/pinn-agent', methods=['POST'])
def pinn_analysis():
    data = request.get_json()

    # Validate user input
    user_query = data.get('user_query')
    if not user_query:
        return jsonify({"error": "User query is required"}), 400

    # Step 1: Define the neural network architecture
    model_definition_output = model_definition_agent.perform_task(user_query)
    if "error" in model_definition_output:
        return jsonify({"error": "Model definition failed.", "details": model_definition_output}), 500

    # Step 2: Extract relevant physics equations
    physics_equation_output = physics_equation_agent.perform_task(user_query)
    if "error" in physics_equation_output:
        return jsonify({"error": "Physics equations extraction failed.", "details": physics_equation_output}), 500

    # Step 3: Define the loss function for the PINN
    loss_function_output = loss_function_agent.perform_task(user_query)
    if "error" in loss_function_output:
        return jsonify({"error": "Loss function definition failed.", "details": loss_function_output}), 500

    # Step 4: Set up the training process for the PINN
    training_output = training_agent.perform_task(user_query)
    if "error" in training_output:
        return jsonify({"error": "Training setup failed.", "details": training_output}), 500

    # Format the final response with each agent's output
    response_data = {
        "model_definition": model_definition_output,
        "physics_equations": physics_equation_output,
        "loss_function": loss_function_output,
        "training_process": training_output
    }

    return jsonify(response_data)