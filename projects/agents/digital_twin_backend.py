from flask import Blueprint, request, jsonify
from config import config
from utils.agents import Agent

digital_twin_back_bp = Blueprint('digital_twin_backend', __name__)

system_message = """You're Mark, an AI Agent specialized in digital twins. I will provide you with a specific problem and the theoretical background related to it. Your task is to return all the Python code and the structure needed to simulate this problem as a backend application.

Please include the following in your response:

Problem Statement: Clearly define the specific problem I provide.
Digital Twin Application: Explain how digital twins are relevant to this problem.
Required Libraries: List any necessary Python libraries or packages needed for the simulation.
Application Architecture: Outline the structure of the backend application, including any relevant classes, modules, or components.
Core Functions/Methods: Provide the main functions or methods that will be implemented in the simulation.
Configuration Files: Include any configuration files or setup instructions needed for the application.
Example Usage: Give a brief example of how to use the code in practice.
Make sure the code is well-organized, clearly commented, and follows best practices for Python programming."""

# Instantiate agents with specific roles and goals
problem_definition_agent = Agent(
    role='Problem Definition Specialist',
    system_message=(
        "You are a specialist in defining clear and concise problem statements. Given the provided query, define the "
        "problem clearly and structure the problem statement."
    )
)

digital_twin_relevance_agent = Agent(
    role='Digital Twin Relevance Specialist',
    system_message=(
        "You are a specialist in the application of digital twins. Based on the provided problem, explain how digital "
        "twins can be applied to simulate and solve the given problem."
    )
)

required_libraries_agent = Agent(
    role='Required Libraries Expert',
    system_message=(
        "You are a Python expert. Based on the given problem, list all necessary Python libraries and packages that will "
        "be used in the simulation, ensuring the libraries are the most appropriate for the task."
    )
)

application_architecture_agent = Agent(
    role='Application Architecture Expert',
    system_message=(
        "You are an expert in designing backend applications. Given the problem and the context of the digital twin, "
        "outline the application architecture, including modules, classes, components, and how they interact."
    )
)

core_functions_agent = Agent(
    role='Core Functions Expert',
    system_message=(
        "You are a Python developer. Given the provided problem, outline and provide the core functions and methods "
        "that will be implemented to simulate the problem. Be sure to follow best practices in Python development."
    )
)

configuration_files_agent = Agent(
    role='Configuration Files Specialist',
    system_message=(
        "You are an expert in Python application setup. Given the simulation problem, provide the necessary "
        "configuration files and setup instructions for the backend application."
    )
)

# Define the main route that processes the user's query
@digital_twin_back_bp.route('/digital-twin-backend', methods=['POST'])
def develop_dt_backend():
    data = request.get_json()

    # Validate user input
    user_query = data.get('user_query')
    if not user_query:
        return jsonify({"error": "User query is required"}), 400

    # Pipeline process
    # Step 1: Problem Definition
    problem_output = problem_definition_agent.perform_task(user_query)
    if "error" in problem_output:
        return jsonify({"error": "Problem definition failed.", "details": problem_output}), 500

    # Step 2: Digital Twin Relevance
    relevance_output = digital_twin_relevance_agent.perform_task(user_query)
    if "error" in relevance_output:
        return jsonify({"error": "Digital twin relevance explanation failed.", "details": relevance_output}), 500

    # Step 3: Required Libraries
    libraries_output = required_libraries_agent.perform_task(user_query)
    if "error" in libraries_output:
        return jsonify({"error": "Library list generation failed.", "details": libraries_output}), 500

    # Step 4: Application Architecture
    architecture_output = application_architecture_agent.perform_task(user_query)
    if "error" in architecture_output:
        return jsonify({"error": "Application architecture definition failed.", "details": architecture_output}), 500

    # Step 5: Core Functions and Methods
    core_functions_output = core_functions_agent.perform_task(user_query)
    if "error" in core_functions_output:
        return jsonify({"error": "Core functions generation failed.", "details": core_functions_output}), 500

    # Step 6: Configuration Files and Setup
    config_files_output = configuration_files_agent.perform_task(user_query)
    if "error" in config_files_output:
        return jsonify({"error": "Configuration files generation failed.", "details": config_files_output}), 500

    # Format the final response with each step's output
    response_data = {
        "problem_definition": problem_output,
        "digital_twin_relevance": relevance_output,
        "required_libraries": libraries_output,
        "application_architecture": architecture_output,
        "core_functions": core_functions_output,
        "configuration_files": config_files_output
    }

    return jsonify({"ai_agent_prompt": response_data})