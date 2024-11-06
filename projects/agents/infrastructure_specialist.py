from flask import Blueprint, request, jsonify
from config import config
from utils.agents import Agent

infra_bp = Blueprint('infrastructure_specialist', __name__)


# Define individual agents for different infrastructure components
cloud_services_agent = Agent(
    role='Cloud Services Agent',
    system_message=(
        "You are an AI agent specialized in recommending cloud services. "
        "Given a project and its requirements, you will suggest the appropriate cloud provider, services, and configurations."
    )
)

server_architecture_agent = Agent(
    role='Server Architecture Agent',
    system_message=(
        "You are an AI agent specializing in designing server architectures. "
        "You will analyze the project codebase and requirements to suggest an appropriate server setup, including load balancing, scaling, and resource management."
    )
)

database_setup_agent = Agent(
    role='Database Setup Agent',
    system_message=(
        "You are an AI agent specializing in database setup. "
        "You will recommend the best database system, configurations, and strategies for ensuring scalability, availability, and security."
    )
)

security_configuration_agent = Agent(
    role='Security Configuration Agent',
    system_message=(
        "You are an AI agent specialized in security configurations. "
        "Given the project codebase and infrastructure requirements, you will suggest the best practices for securing the infrastructure, including firewall settings, encryption, and authentication mechanisms."
    )
)

# Define the main route for IT infrastructure design
@infra_bp.route('/infrastructure_specialist', methods=['POST'])
def get_infra():
    data = request.get_json()

    # Validate user input
    user_query = data.get('user_query')
    if not user_query:
        return jsonify({"error": "User query is required"}), 400

    # Pipeline process
    # Step 1: Recommend cloud services
    cloud_services_output = cloud_services_agent.perform_task(user_query)
    if "error" in cloud_services_output:
        return jsonify({"error": "Cloud services recommendation failed.", "details": cloud_services_output}), 500

    # Step 2: Design server architecture
    server_architecture_output = server_architecture_agent.perform_task(user_query)
    if "error" in server_architecture_output:
        return jsonify({"error": "Server architecture design failed.", "details": server_architecture_output}), 500

    # Step 3: Recommend database setup
    database_setup_output = database_setup_agent.perform_task(user_query)
    if "error" in database_setup_output:
        return jsonify({"error": "Database setup recommendation failed.", "details": database_setup_output}), 500

    # Step 4: Suggest security configurations
    security_configuration_output = security_configuration_agent.perform_task(user_query)
    if "error" in security_configuration_output:
        return jsonify({"error": "Security configuration recommendation failed.", "details": security_configuration_output}), 500

    # Format the final response with each agent's output
    response_data = {
        "cloud_services_recommendation": cloud_services_output,
        "server_architecture_design": server_architecture_output,
        "database_setup_recommendation": database_setup_output,
        "security_configuration_recommendation": security_configuration_output
    }

    return jsonify(response_data)