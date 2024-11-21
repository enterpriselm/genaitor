from flask import Blueprint, request, jsonify
from config import config
from utils.agents import Agent, Orchestrator

paper_summarize_bp = Blueprint('papers_analyst', __name__)

# Define agents for different aspects of paper analysis
agents = {
    'summary_agent': Agent(
        role='Summary Agent',
        system_message=(
            "You are a highly skilled AI agent trained to summarize scientific papers. "
            "Given the user's query, you will provide a concise summary of a paper on Physics Informed Neural Networks."
        )
    ),
    'background_agent': Agent(
        role='Background Information Agent',
        system_message=(
            "You are an AI agent that specializes in providing background information. "
            "Given the user's query, you will provide relevant background on Physics Informed Neural Networks, helping the user better understand the paper's context."
        )
    ),
    'key_results_agent': Agent(
        role='Key Results Agent',
        system_message=(
            "You are an AI agent specialized in extracting key results. "
            "For a given scientific paper, you will identify and highlight the key results, findings, or contributions made in the research."
        )
    ),
    'methodology_agent': Agent(
        role='Methodology Agent',
        system_message=(
            "You are an AI agent specialized in summarizing research methodologies. "
            "For a given scientific paper, you will provide a detailed breakdown of the methods and approaches used in the study."
        )
    ),
}

# Define the task flow for paper analysis
tasks = [
    {"description": "Summary of the paper", "agent": agents['summary_agent']},
    {"description": "Background information on the paper's topic", "agent": agents['background_agent']},
    {"description": "Key results of the paper", "agent": agents['key_results_agent']},
    {"description": "Methodology of the paper", "agent": agents['methodology_agent']},
]

# Initialize Orchestrator to manage the task flow
paper_orchestrator = Orchestrator(agents=agents, tasks=tasks, process='sequential', cumulative=False)

# Define the main route for paper summarization and analysis
@paper_summarize_bp.route('/papers_analyst', methods=['POST'])
def papers_summarize():
    data = request.get_json()

    # Validate user input
    user_query = data.get('user_query')
    if not user_query:
        return jsonify({"error": "User query is required"}), 400

    # Use orchestrator to handle the pipeline for paper analysis
    result = paper_orchestrator.kickoff(user_query)

    if "error" in result:
        return jsonify(result), 500

    # Format the output from each task in the pipeline
    response_data = {task['description']: res for task, res in zip(tasks, result["output"])}

    return jsonify(response_data), 200
