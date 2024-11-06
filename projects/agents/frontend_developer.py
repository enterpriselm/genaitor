from flask import Blueprint, request, jsonify
from config import config
from utils.agents import Agent

frontend_bp = Blueprint('frontend_developer', __name__)

# Define individual agents for different frontend development tasks
html_css_expert = Agent(
    role='HTML/CSS Expert',
    system_message=(
        "You are an expert in HTML and CSS. Given a task and its reference, provide high-quality HTML and CSS code that is "
        "responsive, clean, and optimized for performance. Ensure the design aligns with the user's specifications, and use the "
        "best practices for web standards."
    )
)

javascript_react_expert = Agent(
    role='JavaScript/React Expert',
    system_message=(
        "You are a JavaScript and React expert. Given a task and its reference, provide clean and efficient JavaScript code. "
        "If the task involves React, create reusable and maintainable components, ensuring the use of hooks, state management, "
        "and optimal rendering techniques. You are also skilled in integrating APIs and managing state effectively in React."
    )
)

frontend_design_expert = Agent(
    role='Frontend Design Expert',
    system_message=(
        "You are a frontend design expert. Based on the task and its context, propose a user-friendly, aesthetically pleasing "
        "design for the interface. Ensure the design enhances the user experience and aligns with modern design trends, including "
        "color schemes, typography, and layout considerations."
    )
)

# Define the main route for frontend development
@frontend_bp.route('/frontend-developer', methods=['POST'])
def develop_frontend():
    data = request.get_json()

    # Validate user input
    user_query = data.get('user_query')
    if not user_query:
        return jsonify({"error": "User query is required"}), 400

    # Pipeline process
    # Step 1: HTML/CSS Expert generates HTML and CSS code
    html_css_output = html_css_expert.perform_task(user_query)
    if "error" in html_css_output:
        return jsonify({"error": "HTML/CSS generation failed.", "details": html_css_output}), 500

    # Step 2: JavaScript/React Expert generates JavaScript or React components
    javascript_react_output = javascript_react_expert.perform_task(user_query)
    if "error" in javascript_react_output:
        return jsonify({"error": "JavaScript/React generation failed.", "details": javascript_react_output}), 500

    # Step 3: Frontend Design Expert proposes a design for the interface
    design_output = frontend_design_expert.perform_task(user_query)
    if "error" in design_output:
        return jsonify({"error": "Frontend design failed.", "details": design_output}), 500

    # Format the final response with each agent's output
    response_data = {
        "html_css_code": html_css_output,
        "javascript_react_code": javascript_react_output,
        "design_proposal": design_output
    }

    return jsonify({"ai_agent_prompt": response_data})