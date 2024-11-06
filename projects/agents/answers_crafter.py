
from flask import Blueprint, request, jsonify
from config import config
from utils.agents import Agent

answer_crafter_bp = Blueprint('answer_crafter', __name__)

validator = Agent(
    role='Validator',
    goal='Validate and correct input for clarity and relevance.',
    system_message=(
        "You are an AI specializing in response validation and correction. "
        "When given an input, assess the accuracy and relevance. If unclear or incorrect, "
        "provide a revised response that aligns with the input's requirements."
    )
)

researcher = Agent(
    role='Researcher',
    goal='Conduct AI research to find the latest advancements and trends.',
    system_message=(
        "You are an expert researcher in AI advancements. Given validated input, "
        "identify and summarize the latest trends and breakthroughs in AI, especially those from 2024."
    )
)

writer = Agent(
    role='Content Writer',
    goal='Craft engaging content based on research findings.',
    system_message=(
        "You are a skilled writer who transforms research insights into engaging content. "
        "Take the summarized findings and create a compelling article suitable for a tech-savvy audience."
    )
)

def process_pipeline(user_query):
    # Step 1: Validate the input
    validated_output = validator.perform_task(user_query)
    if "error" in validated_output:
        return {"error": "Validation step failed. Check your input and try again.", "details": validated_output}

    # Step 2: Conduct research based on validated output
    research_output = researcher.perform_task(validated_output)
    if "error" in research_output:
        return {"error": "Research step failed. Unable to gather insights.", "details": research_output}

    # Step 3: Write the article based on research findings
    final_content = writer.perform_task(research_output)
    if "error" in final_content:
        return {"error": "Content writing step failed. Unable to generate final article.", "details": final_content}

    return {
        "validated_input": validated_output,
        "research_summary": research_output,
        "final_article": final_content
    }

@answer_crafter_bp.route('/answers-crafter', methods=['POST'])
def generate_answer():
    data = request.get_json()
    
    user_query = data.get('user_query')
    if not user_query:
        return jsonify({"error": "User query is required"}), 400

    result = process_pipeline(user_query)
    
    if "error" in result:
        return jsonify(result), 500

    return jsonify({
        "ai_agent_prompt": result["final_article"],
        "pipeline_details": {
            "validated_input": result["validated_input"],
            "research_summary": result["research_summary"]
        }
    }), 200