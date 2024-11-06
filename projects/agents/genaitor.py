from flask import Blueprint, request, jsonify
from config import config
from utils.agents import Agent

genaitor_bp = Blueprint('genaitor', __name__)

# Define individual agents for generating AI agents
prompt_generator_agent = Agent(
    role='Prompt Generator Agent',
    system_message=(
        "You are an AI agent specialized in generating high-quality prompts. Given the user's requirements, create the best prompt "
        "for passing through a large language model (LLM) to meet those needs. Ensure the prompts are clear, contextually accurate, "
        "and fit for the model to understand and generate the correct response. Example: {example}"
    )
)

model_recommendation_agent = Agent(
    role='Model Recommendation Agent',
    system_message=(
        "You are an AI agent specialized in recommending the best AI models based on user requirements. Given the task description, "
        "you should suggest the most suitable model, ensuring the selected model is the best fit for the problem's context."
    )
)

agent_configuration_agent = Agent(
    role='Agent Configuration Expert',
    system_message=(
        "You are an expert in configuring AI agents. Based on the task and model recommendation, you will configure the necessary parameters "
        "and settings to fine-tune the agent for optimal performance in solving the user’s task."
    )
)

# Define the main route for generating AI agents
@genaitor_bp.route('/genaitor', methods=['POST'])
def imaginate():
    data = request.get_json()

    # Validate user input
    user_query = data.get('user_query')
    if not user_query:
        return jsonify({"error": "User query is required"}), 400

    # Pipeline process
    # Step 1: Generate the best prompt for the LLM
    prompt_generation_output = prompt_generator_agent.perform_task(user_query)
    if "error" in prompt_generation_output:
        return jsonify({"error": "Prompt generation failed.", "details": prompt_generation_output}), 500

    # Step 2: Recommend the best model based on the query
    model_recommendation_output = model_recommendation_agent.perform_task(user_query)
    if "error" in model_recommendation_output:
        return jsonify({"error": "Model recommendation failed.", "details": model_recommendation_output}), 500

    # Step 3: Configure the agent based on the prompt and model recommendation
    agent_configuration_output = agent_configuration_agent.perform_task(user_query)
    if "error" in agent_configuration_output:
        return jsonify({"error": "Agent configuration failed.", "details": agent_configuration_output}), 500

    # Format the final response with each agent's output
    response_data = {
        "ai_agent_prompt": prompt_generation_output,
        "recommended_model": model_recommendation_output,
        "agent_configuration": agent_configuration_output
    }

    return jsonify(response_data)