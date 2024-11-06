from flask import Blueprint, request, jsonify
from config import config
from utils.agents import Agent

paper_summarize_bp = Blueprint('papers_analyst', __name__)

# Define individual agents for different aspects of paper analysis
summary_agent = Agent(
    role='Summary Agent',
    system_message=(
        "You are a highly skilled AI agent trained to summarize scientific papers. "
        "Given the user's query, you will provide a concise summary of a paper on Physics Informed Neural Networks."
    )
)

background_agent = Agent(
    role='Background Information Agent',
    system_message=(
        "You are an AI agent that specializes in providing background information. "
        "Given the user's query, you will provide relevant background on Physics Informed Neural Networks, helping the user better understand the paper's context."
    )
)

key_results_agent = Agent(
    role='Key Results Agent',
    system_message=(
        "You are an AI agent specialized in extracting key results. "
        "For a given scientific paper, you will identify and highlight the key results, findings, or contributions made in the research."
    )
)

methodology_agent = Agent(
    role='Methodology Agent',
    system_message=(
        "You are an AI agent specialized in summarizing research methodologies. "
        "For a given scientific paper, you will provide a detailed breakdown of the methods and approaches used in the study."
    )
)

# Define the main route for paper summarization and analysis
@paper_summarize_bp.route('/papers_analyst', methods=['POST'])
def papers_summarize():
    data = request.get_json()

    # Validate user input
    user_query = data.get('user_query')
    if not user_query:
        return jsonify({"error": "User query is required"}), 400

    # Step 1: Get a summary of the paper
    summary_output = summary_agent.perform_task(user_query)
    if "error" in summary_output:
        return jsonify({"error": "Summary generation failed.", "details": summary_output}), 500

    # Step 2: Get background information about the paper's topic
    background_output = background_agent.perform_task(user_query)
    if "error" in background_output:
        return jsonify({"error": "Background information retrieval failed.", "details": background_output}), 500

    # Step 3: Extract the key results of the paper
    key_results_output = key_results_agent.perform_task(user_query)
    if "error" in key_results_output:
        return jsonify({"error": "Key results extraction failed.", "details": key_results_output}), 500

    # Step 4: Get the methodology of the paper
    methodology_output = methodology_agent.perform_task(user_query)
    if "error" in methodology_output:
        return jsonify({"error": "Methodology analysis failed.", "details": methodology_output}), 500

    # Format the final response with each agent's output
    response_data = {
        "summary": summary_output,
        "background_information": background_output,
        "key_results": key_results_output,
        "methodology": methodology_output
    }

    return jsonify(response_data)