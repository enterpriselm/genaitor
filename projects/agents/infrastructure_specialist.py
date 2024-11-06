from flask import Blueprint, request, jsonify
from config import config
from utils.agents import Agent, Orchestrator

infra_bp = Blueprint('infrastructure_specialist', __name__)

# Define agents for infrastructure components
agents = {
    'cloud_services_agent': Agent(
        role='Cloud Services Agent',
        system_message=(
            "You are an AI agent specialized in recommending cloud services. "
            "Given a project and its requirements, you will suggest the appropriate cloud provider, services, and configurations."
        )
    ),
    'server_architecture_agent': Agent(
        role='Server Architecture Agent',
        system_message=(
            "You are an AI agent specializing in designing server architectures. "
            "You will analyze the project codebase and requirements to suggest an appropriate server setup, including load balancing, scaling, and resource management."
        )
    ),
    'database_setup_agent': Agent(
        role='Database Setup Agent',
        system_message=(
            "You are an AI agent specializing in database setup. "
            "You will recommend the best database system, configurations, and strategies for ensuring scalability, availability, and security."
        )
    ),
    'security_configuration_agent': Agent(
        role='Security Configuration Agent',
        system_message=(
            "You are an AI agent specialized in security configurations. "
            "Given the project codebase and infrastructure requirements, you will suggest the best practices for securing the infrastructure, including firewall settings, encryption, and authentication mechanisms."
        )
    ),
}

# Define the task flow for IT infrastructure design
tasks = [
    {"description": "Recommend cloud services", "agent": agents['cloud_services_agent']},
    {"description": "Design server architecture", "agent": agents['server_architecture_agent']},
    {"description": "Recommend database setup", "agent": agents['database_setup_agent']},
    {"description": "Suggest security configurations", "agent": agents['security_configuration_agent']},
]

# Initialize Orchestrator to manage the task flow
infra_orchestrator = Orchestrator(agents=agents, tasks=tasks, process='sequential', cumulative=False)

# Define the main route for IT infrastructure design
@infra_bp.route('/infrastructure_specialist', methods=['POST'])
def get_infra():
    data = request.get_json()

    # Validate user input
    user_query = data.get('user_query')
    if not user_query:
        return jsonify({"error": "User query is required"}), 400

    # Use orchestrator to handle the pipeline for infrastructure design
    result = infra_orchestrator.kickoff(user_query)

    if "error" in result:
        return jsonify(result), 500

    # Format the output from each task in the pipeline
    response_data = {task['description']: res for task, res in zip(tasks, result["output"])}

    return jsonify(response_data), 200
