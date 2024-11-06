from flask import Blueprint, request, jsonify
from config import config
from utils.agents import Agent

scraper_bp = Blueprint('scraper_agent', __name__)

# Define individual agents for different scraping tasks
html_analysis_agent = Agent(
    role='HTML Analysis Agent',
    system_message=(
        "You are an AI agent specialized in analyzing the HTML structure of a webpage. "
        "Your task is to understand the structure and identify where the desired data is located within the HTML."
    )
)

data_extraction_agent = Agent(
    role='Data Extraction Agent',
    system_message=(
        "You are an AI agent specialized in extracting data from an HTML page. "
        "You will develop strategies to collect the requested data efficiently, ensuring all desired elements are extracted."
    )
)

scraper_code_agent = Agent(
    role='Scraper Code Generation Agent',
    system_message=(
        "You are an AI agent specialized in generating web scraping code. "
        "Your task is to write efficient and optimized code for extracting the specified data from a webpage, considering factors like pagination, AJAX, and dynamic content."
    )
)

best_practices_agent = Agent(
    role='Scraping Best Practices Agent',
    system_message=(
        "You are an AI agent specialized in ensuring compliance with web scraping best practices. "
        "You will provide guidance on rate limiting, handling CAPTCHA challenges, respecting robots.txt files, and avoiding IP blocking."
    )
)

legal_compliance_agent = Agent(
    role='Legal Compliance Agent',
    system_message=(
        "You are an AI agent specialized in web scraping legal considerations. "
        "You will analyze the legality of scraping the target website, ensuring that the scraping strategy complies with relevant laws and regulations such as GDPR or copyright laws."
    )
)

# Define the main route for Scraping Agent
@scraper_bp.route('/scraper-agent', methods=['POST'])
def generate_scraper():
    data = request.get_json()

    # Validate user input
    user_query = data.get('user_query')
    if not user_query:
        return jsonify({"error": "User query is required"}), 400

    # Step 1: Analyze the HTML structure
    html_analysis_output = html_analysis_agent.perform_task(user_query)
    if "error" in html_analysis_output:
        return jsonify({"error": "HTML analysis failed.", "details": html_analysis_output}), 500

    # Step 2: Extract the required data from the HTML
    data_extraction_output = data_extraction_agent.perform_task(user_query)
    if "error" in data_extraction_output:
        return jsonify({"error": "Data extraction failed.", "details": data_extraction_output}), 500

    # Step 3: Generate scraper code for the task
    scraper_code_output = scraper_code_agent.perform_task(user_query)
    if "error" in scraper_code_output:
        return jsonify({"error": "Scraper code generation failed.", "details": scraper_code_output}), 500

    # Step 4: Provide best practices for web scraping
    best_practices_output = best_practices_agent.perform_task(user_query)
    if "error" in best_practices_output:
        return jsonify({"error": "Best practices advice failed.", "details": best_practices_output}), 500

    # Step 5: Ensure legal compliance for scraping
    legal_compliance_output = legal_compliance_agent.perform_task(user_query)
    if "error" in legal_compliance_output:
        return jsonify({"error": "Legal compliance check failed.", "details": legal_compliance_output}), 500

    # Format the final response with each agent's output
    response_data = {
        "html_analysis": html_analysis_output,
        "data_extraction": data_extraction_output,
        "scraper_code": scraper_code_output,
        "best_practices": best_practices_output,
        "legal_compliance": legal_compliance_output
    }

    return jsonify(response_data)