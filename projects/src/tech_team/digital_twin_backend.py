from flask import Blueprint, request, jsonify
from utils.agents import Agent, Orchestrator

# Initialize Flask Blueprint for digital twin backend
digital_twin_back_bp = Blueprint('digital_twin_backend', __name__)

# Define agents with their roles and system messages
agents = {
    'problem_definition': Agent(
        role='Problem Definition Specialist',
        system_message=(
            "You are a specialist in defining clear and concise problem statements. Given the provided query, define the "
            "problem clearly and structure the problem statement."
        )
    ),
    'digital_twin_relevance': Agent(
        role='Digital Twin Relevance Specialist',
        system_message=(
            "You are a specialist in the application of digital twins. Based on the provided problem, explain how digital "
            "twins can be applied to simulate and solve the given problem."
        )
    ),
    'required_libraries': Agent(
        role='Required Libraries Expert',
        system_message=(
            "You are a Python expert. Based on the given problem, list all necessary Python libraries and packages that will "
            "be used in the simulation, ensuring the libraries are the most appropriate for the task."
        )
    ),
    'application_architecture': Agent(
        role='Application Architecture Expert',
        system_message=(
            "You are an expert in designing backend applications. Given the problem and the context of the digital twin, "
            "outline the application architecture, including modules, classes, components, and how they interact."
        )
    ),
    'core_functions': Agent(
        role='Core Functions Expert',
        system_message=(
            "You are a Python developer. Given the provided problem, outline and provide the core functions and methods "
            "that will be implemented to simulate the problem. Be sure to follow best practices in Python development."
        )
    ),
    'configuration_files': Agent(
        role='Configuration Files Specialist',
        system_message=(
            "You are an expert in Python application setup. Given the simulation problem, provide the necessary "
            "configuration files and setup instructions for the backend application."
        )
    )
}

# Define the task flow for the digital twin backend application
tasks = [
    {"description": "Define the problem statement", "agent": agents['problem_definition']},
    {"description": "Explain the relevance of digital twins", "agent": agents['digital_twin_relevance']},
    {"description": "List required libraries", "agent": agents['required_libraries']},
    {"description": "Outline application architecture", "agent": agents['application_architecture']},
    {"description": "Provide core functions and methods", "agent": agents['core_functions']},
    {"description": "Generate configuration files", "agent": agents['configuration_files']}
]

# Initialize Orchestrator to handle task execution
digital_twin_orchestrator = Orchestrator(agents=agents, tasks=tasks, process='sequential', cumulative=False)

# Define the main route that processes the user's query
@digital_twin_back_bp.route('/digital-twin-backend', methods=['POST'])
def develop_dt_backend():
    data = request.get_json()

    # Validate user input
    user_query = data.get('user_query')
    if not user_query:
        return jsonify({"error": "User query is required"}), 400

    # Use orchestrator to handle the workflow for the digital twin backend development
    result = digital_twin_orchestrator.kickoff(user_query)

    if "error" in result:
        return jsonify(result), 500

    # Format the output from each task in the pipeline
    response_data = {task['description']: res for task, res in zip(tasks, result["output"])}

    return jsonify({
        "ai_agent_prompt": response_data
    }), 200
