from flask import Blueprint, request, jsonify
from utils.agents import Agent, Orchestrator

# Initialize Flask Blueprint
apps_builder_bp = Blueprint('apps_builder', __name__)

# Define agents with their roles, goals, and system messages
agents = {
    'requirement_analyst': Agent(
        role='Requirement Analyst',
        goal='Analyze user requirements for the application.',
        system_message=(
            "You are a skilled Requirement Analyst. Given an app-building query, identify and clarify "
            "the user’s main requirements and desired features. Summarize the app’s core functions."
        )
    ),
    'system_architect': Agent(
        role='System Architect',
        goal='Design a system architecture based on the analyzed requirements.',
        system_message=(
            "You are an experienced System Architect. Based on the identified requirements, "
            "propose an architecture that includes the necessary components, databases, and integrations."
        )
    ),
    'frontend_developer': Agent(
        role='Frontend Developer',
        goal='Create the frontend plan using React.',
        system_message=(
            "You are a skilled Frontend Developer with expertise in React. Based on the app’s architecture, "
            "design the frontend structure, components, and main user interactions in React."
        )
    ),
    'backend_developer': Agent(
        role='Backend Developer',
        goal='Develop the backend plan using Python.',
        system_message=(
            "You are a Python Backend Developer. Given the app’s architecture, design the backend "
            "services, APIs, and database interactions necessary to support the frontend and app features."
        )
    )
}

# Define the task flow for the app-building pipeline
tasks = [
    {"description": "Analyze requirements", "agent": agents['requirement_analyst']},
    {"description": "Design system architecture", "agent": agents['system_architect']},
    {"description": "Plan frontend in React", "agent": agents['frontend_developer']},
    {"description": "Plan backend in Python", "agent": agents['backend_developer']}
]

# Initialize the Orchestrator with agents, tasks, and process
app_builder_orchestrator = Orchestrator(agents=agents, tasks=tasks, process='sequential', cumulative=False)

@apps_builder_bp.route('/apps-builder', methods=['POST'])
def build_app():
    data = request.get_json()
    
    # Validate user input
    user_query = data.get('user_query')
    if not user_query:
        return jsonify({"error": "User query is required"}), 400

    # Initiate the Orchestrator to handle the app-building workflow
    result = app_builder_orchestrator.kickoff(user_query)

    if "error" in result:
        return jsonify(result), 500

    # Extract outputs for each step in the process
    response_data = {task['description']: res for task, res in zip(tasks, result["output"])}

    return jsonify({
        "ai_agent_prompt": response_data
    }), 200