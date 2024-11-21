from flask import Blueprint, request, jsonify
from utils.agents import Agent, Orchestrator

# Initialize Flask Blueprint for cybersecurity
cybersec_bp = Blueprint('cybersecurity_agent', __name__)

# Define agents with their roles, system messages, and goals
agents = {
    'vulnerability_assessor': Agent(
        role='Vulnerability Assessor',
        system_message=(
            "You are a skilled cybersecurity analyst. Given a security-related task, identify potential vulnerabilities in "
            "the provided system or code. Focus on common weaknesses and areas where attackers might exploit flaws."
        )
    ),
    'defensive_measures_expert': Agent(
        role='Defensive Measures Expert',
        system_message=(
            "You are a cybersecurity expert specializing in defensive measures. Given a system or code, suggest security best practices, "
            "defensive programming techniques, and implementations of firewalls, encryption, or other protective measures."
        )
    ),
    'threat_detection_specialist': Agent(
        role='Threat Detection Specialist',
        system_message=(
            "You are a cybersecurity analyst with expertise in threat detection. Based on the provided security task, identify "
            "methods for detecting potential threats, including code injection, malware, DDoS attacks, or other malicious activities."
        )
    ),
    'security_documentation_assistant': Agent(
        role='Security Documentation Assistant',
        system_message=(
            "You are a security documentation expert. After analyzing the security task, provide a detailed summary of the vulnerabilities, "
            "defensive measures, and threat detection techniques. Include code examples and configuration steps where necessary."
        )
    )
}

# Define the task flow for the cybersecurity analysis pipeline
tasks = [
    {"description": "Assess potential vulnerabilities", "agent": agents['vulnerability_assessor']},
    {"description": "Review defensive measures", "agent": agents['defensive_measures_expert']},
    {"description": "Detect and mitigate threats", "agent": agents['threat_detection_specialist']},
    {"description": "Generate security documentation", "agent": agents['security_documentation_assistant']}
]

# Initialize Orchestrator to handle task execution
cybersec_orchestrator = Orchestrator(agents=agents, tasks=tasks, process='sequential', cumulative=False)

@cybersec_bp.route('/cybersecurity-agent', methods=['POST'])
def cyber_analysis():
    data = request.get_json()

    # Validate user input
    user_query = data.get('user_query')
    if not user_query:
        return jsonify({"error": "User query is required"}), 400

    # Use orchestrator to handle the workflow for cybersecurity analysis
    result = cybersec_orchestrator.kickoff(user_query)

    if "error" in result:
        return jsonify(result), 500

    # Format the output from each task in the pipeline
    response_data = {task['description']: res for task, res in zip(tasks, result["output"])}

    return jsonify({
        "ai_agent_prompt": response_data
    }), 200
