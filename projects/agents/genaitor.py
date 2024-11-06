from flask import Blueprint, request, jsonify
from config import config
from utils.agents import Agent, Orchestrator

genaitor_bp = Blueprint('genaitor', __name__)

# Define agents for generating AI agents
agents = {
    'prompt_generator_agent': Agent(
        role='Prompt Generator Agent',
        system_message=(
            "You are an AI agent specialized in generating high-quality prompts. Given the user's requirements, create the best prompt "
            "for passing through a large language model (LLM) to meet those needs. Ensure the prompts are clear, contextually accurate, "
            "and fit for the model to understand and generate the correct response. Example: {example}"
        )
    ),
    'model_recommendation_agent': Agent(
        role='Model Recommendation Agent',
        system_message=(
            "You are an AI agent specialized in recommending the best AI models based on user requirements. Given the task description, "
            "you should suggest the most suitable model, ensuring the selected model is the best fit for the problem's context."
        )
    ),
    'agent_configuration_agent': Agent(
        role='Agent Configuration Expert',
        system_message=(
            "You are an expert in configuring AI agents. Based on the task and model recommendation, you will configure the necessary parameters "
            "and settings to fine-tune the agent for optimal performance in solving the user’s task."
        )
    ),
}

# Define the task flow for generating AI agents
tasks = [
    {"description": "Generate the best prompt for the LLM", "agent": agents['prompt_generator_agent']},
    {"description": "Recommend the best model", "agent": agents['model_recommendation_agent']},
    {"description": "Configure the AI agent", "agent": agents['agent_configuration_agent']}
]

# Initialize Orchestrator to handle task execution
genaitor_orchestrator = Orchestrator(agents=agents, tasks=tasks, process='sequential', cumulative=False)

# Define the main route for generating AI agents
@genaitor_bp.route('/genaitor', methods=['POST'])
def imaginate():
    data = request.get_json()

    # Validate user input
    user_query = data.get('user_query')
    if not user_query:
        return jsonify({"error": "User query is required"}), 400

    # Use orchestrator to handle the workflow for generating AI agents
    result = genaitor_orchestrator.kickoff(user_query)

    if "error" in result:
        return jsonify(result), 500

    # Format the output from each task in the pipeline
    response_data = {task['description']: res for task, res in zip(tasks, result["output"])}

    return jsonify(response_data), 200
