from flask import Blueprint, request, jsonify
from config import config
from utils.agents import Agent

code_review_bp = Blueprint('code_reviewer', __name__)

code_style_checker = Agent(
    role='Code Style Checker',
    system_message=(
        "You are an expert in code style and best practices. Review the given code snippet and check if it adheres "
        "to best practices for style (e.g., PEP 8 for Python). Provide suggestions for improving readability and maintainability."
    )
)

code_optimization_specialist = Agent(
    role='Code Optimization Specialist',
    system_message=(
        "You are an expert in code optimization. Review the given code snippet and identify any areas where performance "
        "can be improved. Suggest optimizations where relevant."
    )
)

bug_and_error_finder = Agent(
    role='Bug and Error Finder',
    system_message=(
        "You are a code debugger. Review the provided code and identify any potential bugs, logical errors, or edge cases "
        "that could cause the code to break or behave unexpectedly."
    )
)

documentation_assistant = Agent(
    role='Documentation and Explanation Assistant',
    system_message=(
        "You are an expert in explaining code. After reviewing the code, provide a detailed explanation of the feedback, "
        "including what was changed, why it was necessary, and any specific code examples where applicable."
    )
)

# Define the main route that processes the user's query
@code_review_bp.route('/code-reviewer', methods=['POST'])
def review_code():
    data = request.get_json()

    # Validate user input
    user_query = data.get('user_query')
    if not user_query:
        return jsonify({"error": "User query is required"}), 400

    # Pipeline process
    # Step 1: Code Style Check
    style_output = code_style_checker.perform_task(user_query)
    if "error" in style_output:
        return jsonify({"error": "Code style review failed.", "details": style_output}), 500

    # Step 2: Code Optimization Review
    optimization_output = code_optimization_specialist.perform_task(user_query)
    if "error" in optimization_output:
        return jsonify({"error": "Code optimization review failed.", "details": optimization_output}), 500

    # Step 3: Bug and Error Detection
    bug_output = bug_and_error_finder.perform_task(user_query)
    if "error" in bug_output:
        return jsonify({"error": "Bug and error detection failed.", "details": bug_output}), 500

    # Step 4: Documentation and Explanation
    documentation_output = documentation_assistant.perform_task(user_query)
    if "error" in documentation_output:
        return jsonify({"error": "Documentation and explanation generation failed.", "details": documentation_output}), 500

    # Format the final response with each step's output
    response_data = {
        "code_style_feedback": style_output,
        "optimization_suggestions": optimization_output,
        "bug_and_error_feedback": bug_output,
        "documentation_and_explanation": documentation_output
    }

    return jsonify({"ai_agent_prompt": response_data})
