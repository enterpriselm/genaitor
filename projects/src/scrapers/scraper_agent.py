from flask import Blueprint, request, jsonify
from config import config
from utils.agents import Agent, Orchestrator

scraper_bp = Blueprint('scraper_agent', __name__)

# Define agents for different scraping tasks
agents = {
    'html_analysis_agent': Agent(
        role='HTML Analysis Agent',
        system_message=(
            "You are an AI agent specialized in analyzing the HTML structure of a webpage. "
            "Your task is to understand the structure and identify where the desired data is located within the HTML."
        )
    ),
    'data_extraction_agent': Agent(
        role='Data Extraction Agent',
        system_message=(
            "You are an AI agent specialized in extracting data from an HTML page. "
            "You will develop strategies to collect the requested data efficiently, ensuring all desired elements are extracted."
        )
    ),
    'scraper_code_agent': Agent(
        role='Scraper Code Generation Agent',
        system_message=(
            "You are an AI agent specialized in generating web scraping code. "
            "Your task is to write efficient and optimized code for extracting the specified data from a webpage, considering factors like pagination, AJAX, and dynamic content."
        )
    ),
    'best_practices_agent': Agent(
        role='Scraping Best Practices Agent',
        system_message=(
            "You are an AI agent specialized in ensuring compliance with web scraping best practices. "
            "You will provide guidance on rate limiting, handling CAPTCHA challenges, respecting robots.txt files, and avoiding IP blocking."
        )
    ),
    'legal_compliance_agent': Agent(
        role='Legal Compliance Agent',
        system_message=(
            "You are an AI agent specialized in web scraping legal considerations. "
            "You will analyze the legality of scraping the target website, ensuring that the scraping strategy complies with relevant laws and regulations such as GDPR or copyright laws."
        )
    ),
}

# Define the task flow for Scraping Agent process
tasks = [
    {"description": "HTML structure analysis", "agent": agents['html_analysis_agent']},
    {"description": "Data extraction", "agent": agents['data_extraction_agent']},
    {"description": "Scraper code generation", "agent": agents['scraper_code_agent']},
    {"description": "Scraping best practices", "agent": agents['best_practices_agent']},
    {"description": "Legal compliance check", "agent": agents['legal_compliance_agent']},
]

# Initialize Orchestrator to manage the task flow
scraper_orchestrator = Orchestrator(agents=agents, tasks=tasks, process='sequential', cumulative=False)

# Define the main route for Scraping Agent
@scraper_bp.route('/scraper-agent', methods=['POST'])
def generate_scraper():
    data = request.get_json()

    # Validate user input
    user_query = data.get('user_query')
    if not user_query:
        return jsonify({"error": "User query is required"}), 400

    # Use orchestrator to handle the pipeline for scraping tasks
    result = scraper_orchestrator.kickoff(user_query)

    if "error" in result:
        return jsonify(result), 500

    # Format the output from each task in the pipeline
    response_data = {task['description']: res for task, res in zip(tasks, result["output"])}

    return jsonify(response_data), 200
