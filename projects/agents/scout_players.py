from flask import Blueprint, request, jsonify
from config import config
from utils.agents import Agent

scout_bp = Blueprint('scout_players', __name__)

# Define individual agents for different scouting aspects
player_analysis_agent = Agent(
    role='Player Analysis Agent',
    system_message=(
        "You are an AI agent specialized in analyzing football players' stats. "
        "You will assess the players' performance, including goals scored, assists, pass accuracy, and more, to make an informed recommendation."
    )
)

market_value_analysis_agent = Agent(
    role='Market Value Analysis Agent',
    system_message=(
        "You are an AI agent specialized in analyzing the market value of football players. "
        "You will analyze the players' transfer market values, considering age, performance, and club demand."
    )
)

team_compatibility_agent = Agent(
    role='Team Compatibility Agent',
    system_message=(
        "You are an AI agent specialized in analyzing the compatibility of football players with a specific team. "
        "You will assess factors like playing style, team requirements, and squad balance to recommend players that fit the team needs."
    )
)

performance_prediction_agent = Agent(
    role='Performance Prediction Agent',
    system_message=(
        "You are an AI agent specialized in predicting football players' future performances. "
        "You will analyze historical data and project how well players are likely to perform in the next season based on their stats and trends."
    )
)

# Define the main route for Scouting Players analysis
@scout_bp.route('/scout-players', methods=['POST'])
def scout_players():
    data = request.get_json()

    # Validate user input
    user_query = data.get('user_query')
    if not user_query:
        return jsonify({"error": "User query is required"}), 400

    # Step 1: Analyze the player's stats
    player_analysis_output = player_analysis_agent.perform_task(user_query)
    if "error" in player_analysis_output:
        return jsonify({"error": "Player analysis failed.", "details": player_analysis_output}), 500

    # Step 2: Analyze the player's market value
    market_value_output = market_value_analysis_agent.perform_task(user_query)
    if "error" in market_value_output:
        return jsonify({"error": "Market value analysis failed.", "details": market_value_output}), 500

    # Step 3: Analyze team compatibility
    team_compatibility_output = team_compatibility_agent.perform_task(user_query)
    if "error" in team_compatibility_output:
        return jsonify({"error": "Team compatibility analysis failed.", "details": team_compatibility_output}), 500

    # Step 4: Predict the player's performance
    performance_prediction_output = performance_prediction_agent.perform_task(user_query)
    if "error" in performance_prediction_output:
        return jsonify({"error": "Performance prediction failed.", "details": performance_prediction_output}), 500

    # Format the final response with each agent's output
    response_data = {
        "player_analysis": player_analysis_output,
        "market_value_analysis": market_value_output,
        "team_compatibility": team_compatibility_output,
        "performance_prediction": performance_prediction_output
    }

    return jsonify(response_data)