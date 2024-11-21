from flask import Blueprint, request, jsonify
from config import config
from utils.agents import Agent, Orchestrator

scout_bp = Blueprint('scout_players', __name__)

# Define agents for different scouting aspects
agents = {
    'player_analysis_agent': Agent(
        role='Player Analysis Agent',
        system_message=(
            "You are an AI agent specialized in analyzing football players' stats. "
            "You will assess the players' performance, including goals scored, assists, pass accuracy, and more, to make an informed recommendation."
        )
    ),
    'market_value_analysis_agent': Agent(
        role='Market Value Analysis Agent',
        system_message=(
            "You are an AI agent specialized in analyzing the market value of football players. "
            "You will analyze the players' transfer market values, considering age, performance, and club demand."
        )
    ),
    'team_compatibility_agent': Agent(
        role='Team Compatibility Agent',
        system_message=(
            "You are an AI agent specialized in analyzing the compatibility of football players with a specific team. "
            "You will assess factors like playing style, team requirements, and squad balance to recommend players that fit the team needs."
        )
    ),
    'performance_prediction_agent': Agent(
        role='Performance Prediction Agent',
        system_message=(
            "You are an AI agent specialized in predicting football players' future performances. "
            "You will analyze historical data and project how well players are likely to perform in the next season based on their stats and trends."
        )
    ),
}

# Define the task flow for Scouting Players analysis
tasks = [
    {"description": "Player stats analysis", "agent": agents['player_analysis_agent']},
    {"description": "Market value analysis", "agent": agents['market_value_analysis_agent']},
    {"description": "Team compatibility analysis", "agent": agents['team_compatibility_agent']},
    {"description": "Performance prediction", "agent": agents['performance_prediction_agent']},
]

# Initialize Orchestrator to manage the task flow
scout_orchestrator = Orchestrator(agents=agents, tasks=tasks, process='sequential', cumulative=False)

# Define the main route for Scouting Players analysis
@scout_bp.route('/scout-players', methods=['POST'])
def scout_players():
    data = request.get_json()

    # Validate user input
    user_query = data.get('user_query')
    if not user_query:
        return jsonify({"error": "User query is required"}), 400

    # Use orchestrator to handle the pipeline for scouting tasks
    result = scout_orchestrator.kickoff(user_query)

    if "error" in result:
        return jsonify(result), 500

    # Format the output from each task in the pipeline
    response_data = {task['description']: res for task, res in zip(tasks, result["output"])}

    return jsonify(response_data), 200
