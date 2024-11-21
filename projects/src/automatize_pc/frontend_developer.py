from flask import Blueprint, request, jsonify
from config import config
from utils.agents import Agent, Orchestrator

frontend_bp = Blueprint('frontend_developer', __name__)

# Define agents for frontend development tasks
agents = {
    'html_css_expert': Agent(
        role='HTML/CSS Expert',
        system_message=(
            "You are an expert in HTML and CSS. Given a task and its reference, provide high-quality HTML and CSS code that is "
            "responsive, clean, and optimized for performance. Ensure the design aligns with the user's specifications, and use the "
            "best practices for web standards."
        )
    ),
    'javascript_react_expert': Agent(
        role='JavaScript/React Expert',
        system_message=(
            "You are a JavaScript and React expert. Given a task and its reference, provide clean and efficient JavaScript code. "
            "If the task involves React, create reusable and maintainable components, ensuring the use of hooks, state management, "
            "and optimal rendering techniques. You are also skilled in integrating APIs and managing state effectively in React."
        )
    ),
    'frontend_design_expert': Agent(
        role='Frontend Design Expert',
        system_message=(
            "You are a frontend design expert. Based on the task and its context, propose a user-friendly, aesthetically pleasing "
            "design for the interface. Ensure the design enhances the user experience and aligns with modern design trends, including "
            "color schemes, typography, and layout considerations."
        )
    ),
}

# Define the task flow for frontend development
tasks = [
    {"description": "Generate HTML/CSS code", "agent": agents['html_css_expert']},
    {"description": "Generate JavaScript/React components", "agent": agents['javascript_react_expert']},
    {"description": "Propose frontend design", "agent": agents['frontend_design_expert']}
]

# Initialize Orchestrator to handle task execution
frontend_orchestrator = Orchestrator(agents=agents, tasks=tasks, process='sequential', cumulative=False)

# Define the main route for frontend development
@frontend_bp.route('/frontend-developer', methods=['POST'])
def develop_frontend():
    data = request.get_json()

    # Validate user input
    user_query = data.get('user_query')
    if not user_query:
        return jsonify({"error": "User query is required"}), 400

    # Use orchestrator to handle the workflow for frontend development
    result = frontend_orchestrator.kickoff(user_query)

    if "error" in result:
        return jsonify(result), 500

    # Format the output from each task in the pipeline
    response_data = {task['description']: res for task, res in zip(tasks, result["output"])}

    return jsonify({
        "ai_agent_prompt": response_data
    }), 200
