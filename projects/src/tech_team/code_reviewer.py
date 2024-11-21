from flask import Blueprint, request, jsonify
from utils.agents import Agent, Orchestrator

# Initialize Flask Blueprint
code_review_bp = Blueprint('code_reviewer', __name__)

# Define agents with their roles, system messages
agents = {
    'code_style_checker': Agent(
        role='Code Style Checker',
        system_message=(
            "You are an expert in code style and best practices. Review the given code snippet and check if it adheres "
            "to best practices for style (e.g., PEP 8 for Python). Provide suggestions for improving readability and maintainability."
        )
    ),
    'code_optimization_specialist': Agent(
        role='Code Optimization Specialist',
        system_message=(
            "You are an expert in code optimization. Review the given code snippet and identify any areas where performance "
            "can be improved. Suggest optimizations where relevant."
        )
    ),
    'bug_and_error_finder': Agent(
        role='Bug and Error Finder',
        system_message=(
            "You are a code debugger. Review the provided code and identify any potential bugs, logical errors, or edge cases "
            "that could cause the code to break or behave unexpectedly."
        )
    ),
    'documentation_assistant': Agent(
        role='Documentation and Explanation Assistant',
        system_message=(
            "You are an expert in explaining code. After reviewing the code, provide a detailed explanation of the feedback, "
            "including what was changed, why it was necessary, and any specific code examples where applicable."
        )
    )
}

# Define the task flow for the code review pipeline
tasks = [
    {"description": "Check code style", "agent": agents['code_style_checker']},
    {"description": "Review code optimization", "agent": agents['code_optimization_specialist']},
    {"description": "Detect bugs and errors", "agent": agents['bug_and_error_finder']},
    {"description": "Generate documentation and explanation", "agent": agents['documentation_assistant']}
]

# Initialize the Orchestrator with agents, tasks, and process
code_review_orchestrator = Orchestrator(agents=agents, tasks=tasks, process='sequential', cumulative=False)

@code_review_bp.route('/code-reviewer', methods=['POST'])
def review_code():
    data = request.get_json()

    # Validate user input
    user_query = data.get('user_query')
    if not user_query:
        return jsonify({"error": "User query is required"}), 400

    # Initiate the Orchestrator to handle the code review workflow
    result = code_review_orchestrator.kickoff(user_query)

    if "error" in result:
        return jsonify(result), 500

    # Extract outputs for each step in the process
    response_data = {task['description']: res for task, res in zip(tasks, result["output"])}

    return jsonify({
        "ai_agent_prompt": response_data
    }), 200
