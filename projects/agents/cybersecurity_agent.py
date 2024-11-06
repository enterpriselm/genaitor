from flask import Blueprint, request, jsonify
from config import config
from utils.agents import Agent

cybersec_bp = Blueprint('cybersecurity_agent', __name__)

# Instantiate agents with specific roles and goals
vulnerability_assessor = Agent(
    role='Vulnerability Assessor',
    system_message=(
        "You are a skilled cybersecurity analyst. Given a security-related task, identify potential vulnerabilities in "
        "the provided system or code. Focus on common weaknesses and areas where attackers might exploit flaws."
    )
)

defensive_measures_expert = Agent(
    role='Defensive Measures Expert',
    system_message=(
        "You are a cybersecurity expert specializing in defensive measures. Given a system or code, suggest security best practices, "
        "defensive programming techniques, and implementations of firewalls, encryption, or other protective measures."
    )
)

threat_detection_specialist = Agent(
    role='Threat Detection Specialist',
    system_message=(
        "You are a cybersecurity analyst with expertise in threat detection. Based on the provided security task, identify "
        "methods for detecting potential threats, including code injection, malware, DDoS attacks, or other malicious activities."
    )
)

security_documentation_assistant = Agent(
    role='Security Documentation Assistant',
    system_message=(
        "You are a security documentation expert. After analyzing the security task, provide a detailed summary of the vulnerabilities, "
        "defensive measures, and threat detection techniques. Include code examples and configuration steps where necessary."
    )
)

# Define the main route that processes the user's query
@cybersec_bp.route('/cybersecurity-agent', methods=['POST'])
def cyber_analysis():
    data = request.get_json()

    # Validate user input
    user_query = data.get('user_query')
    if not user_query:
        return jsonify({"error": "User query is required"}), 400

    # Pipeline process
    # Step 1: Vulnerability Assessment
    vulnerability_output = vulnerability_assessor.perform_task(user_query)
    if "error" in vulnerability_output:
        return jsonify({"error": "Vulnerability assessment failed.", "details": vulnerability_output}), 500

    # Step 2: Defensive Measures Review
    defense_output = defensive_measures_expert.perform_task(user_query)
    if "error" in defense_output:
        return jsonify({"error": "Defensive measures review failed.", "details": defense_output}), 500

    # Step 3: Threat Detection and Mitigation
    threat_detection_output = threat_detection_specialist.perform_task(user_query)
    if "error" in threat_detection_output:
        return jsonify({"error": "Threat detection failed.", "details": threat_detection_output}), 500

    # Step 4: Security Documentation and Recommendations
    documentation_output = security_documentation_assistant.perform_task(user_query)
    if "error" in documentation_output:
        return jsonify({"error": "Security documentation generation failed.", "details": documentation_output}), 500

    # Format the final response with each step's output
    response_data = {
        "vulnerability_assessment": vulnerability_output,
        "defensive_measures": defense_output,
        "threat_detection_and_mitigation": threat_detection_output,
        "security_documentation": documentation_output
    }

    return jsonify({"ai_agent_prompt": response_data})