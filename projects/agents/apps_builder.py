from flask import Blueprint, request, jsonify
from config import config
from utils.agents import Agent

apps_builder_bp = Blueprint('apps_builder', __name__)

requirement_analyst = Agent(
    role='Requirement Analyst',
    goal='Analyze user requirements for the application.',
    system_message=(
        "You are a skilled Requirement Analyst. Given an app-building query, identify and clarify "
        "the user’s main requirements and desired features. Summarize the app’s core functions."
    )
)

system_architect = Agent(
    role='System Architect',
    goal='Design a system architecture based on the analyzed requirements.',
    system_message=(
        "You are an experienced System Architect. Based on the identified requirements, "
        "propose an architecture that includes the necessary components, databases, and integrations."
    )
)

frontend_developer = Agent(
    role='Frontend Developer',
    goal='Create the frontend plan using React.',
    system_message=(
        "You are a skilled Frontend Developer with expertise in React. Based on the app’s architecture, "
        "design the frontend structure, components, and main user interactions in React."
    )
)

backend_developer = Agent(
    role='Backend Developer',
    goal='Develop the backend plan using Python.',
    system_message=(
        "You are a Python Backend Developer. Given the app’s architecture, design the backend "
        "services, APIs, and database interactions necessary to support the frontend and app features."
    )
)

# Define the main route that processes the user's query
@apps_builder_bp.route('/apps-builder', methods=['POST'])
def build_app():
    data = request.get_json()

    # Validate user input
    user_query = data.get('user_query')
    if not user_query:
        return jsonify({"error": "User query is required"}), 400

    # Pipeline process
    requirements_output = requirement_analyst.perform_task(user_query)
    if "error" in requirements_output:
        return jsonify({"error": "Requirements analysis failed.", "details": requirements_output}), 500

    architecture_output = system_architect.perform_task(requirements_output)
    if "error" in architecture_output:
        return jsonify({"error": "System architecture design failed.", "details": architecture_output}), 500

    frontend_plan = frontend_developer.perform_task(architecture_output)
    if "error" in frontend_plan:
        return jsonify({"error": "Frontend development plan failed.", "details": frontend_plan}), 500

    backend_plan = backend_developer.perform_task(architecture_output)
    if "error" in backend_plan:
        return jsonify({"error": "Backend development plan failed.", "details": backend_plan}), 500

    # Format the final response with each step's output
    response_data = {
        "requirements_analysis": requirements_output,
        "system_architecture": architecture_output,
        "frontend_development_plan": frontend_plan,
        "backend_development_plan": backend_plan
    }

    return jsonify({"ai_agent_prompt": response_data})