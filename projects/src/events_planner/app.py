from flask import Flask, render_template, request, jsonify
from datetime import datetime, timedelta
from genaitor.config import config
from genaitor.utils.agents import Agent, Orchestrator, Task

# Initialize Flask app
app = Flask(__name__)

# Define your agents
agents = {
    'local_expert': Agent(
        role='Local Expert at this city',
        system_message=(
            "You're a knowledgeable local guide with extensive information about the city, it's attractions and customs"
            "Provide the BEST insights about the inputed city"
        )
    ),
    'city_selection_agent': Agent(
        role='City Selection Expert',
        system_message=(
            "You are an AI agent expert in analyzing travel data to pick ideal destinations"
            "Given the user's query, you will select the best city based on weather, season and prices."
        )
    ),
    'travel_concierge': Agent(
        role='Travel Concierge',
        system_message=(
            "You are an AI agent Specialist in travel planning and logistics with decades of experience"
            "You will create the most amazing travel itineraries with budget and packing suggestions for the city."
        )
    )
}

# TravelTasks class as defined in the backend code
class TravelTasks:
    def location_task(self, agent, from_city, destination_city, date_from, date_to):
        return Task(
            description=f"""
            This task involves a comprehensive data collection process to provide the traveler with essential information about their destination. It includes researching and compiling details on various accommodations, ranging from budget-friendly hostels to luxury hotels, as well as estimating the cost of living in the area. The task also covers transportation options, visa requirements, and any travel advisories that may be relevant.
            Traveling from : {from_city}
            Destination city : {destination_city}
            Arrival Date : {date_from}
            Departure Date : {date_to}
            """,
            expected_output=f"""
            In markdown format : A detailed markdown report that includes a curated list of recommended places to stay, a breakdown of daily living expenses, and practical travel tips.
            """,
            agent=agent,
            output_file='city_report.md',
        )

    def guide_task(self, agent, destination_city, interests, date_from, date_to):    
        return Task(
            description=f"""
            Tailored to the traveler's personal {interests}, this task focuses on creating an engaging and informative guide to the city's attractions. It involves identifying cultural landmarks, historical spots, entertainment venues, dining experiences, and outdoor activities that align with the user's preferences.
            Destination city : {destination_city}
            interests : {interests}
            Arrival Date : {date_from}
            Departure Date : {date_to}
            """,
            expected_output=f"""
            An interactive markdown report that presents a personalized itinerary of activities and attractions, complete with descriptions, locations, and any necessary reservations or tickets.
            """,
            agent=agent,
            output_file='guide_report.md',
        )

    def planner_task(self, context, agent, destination_city, interests, date_from, date_to):
        return Task(
            description=f"""
            This task synthesizes all collected information into a detailed travel plan that includes a curated introduction to the city, a breakdown of daily living expenses, and a personalized travel itinerary.
            Destination city : {destination_city}
            interests : {interests}
            Arrival Date : {date_from}
            Departure Date : {date_to}
            """,
            expected_output=f"""
            A rich markdown document with emojis on each title and subtitle, outlining a detailed travel plan, with time allocations and activity details.
            """,
            context=context,
            agent=agent,
            output_file='travel_plan.md',
        )


@app.route('/')
def home():
    return render_template('index.html')  # Frontend template


@app.route('/generate_plan', methods=['POST'])
def generate_plan():
    # Get data from the form
    from_city = request.form['from_city']
    destination_city = request.form['destination_city']
    interests = request.form['interests']
    date_from = request.form['date_from']
    date_to = request.form['date_to']

    # Convert date inputs to datetime
    date_from = datetime.strptime(date_from, '%Y-%m-%d')
    date_to = datetime.strptime(date_to, '%Y-%m-%d')

    # Create travel tasks
    travel_tasks = TravelTasks()
    
    tasks = [
        travel_tasks.location_task(
            agent=agents['local_expert'],
            from_city=from_city,
            destination_city=destination_city,
            date_from=date_from,
            date_to=date_to
        ),
        travel_tasks.guide_task(
            agent=agents['city_selection_agent'],
            destination_city=destination_city,
            interests=interests,
            date_from=date_from,
            date_to=date_to
        ),
        travel_tasks.planner_task(
            context=[],
            agent=agents['travel_concierge'],
            destination_city=destination_city,
            interests=interests,
            date_from=date_from,
            date_to=date_to
        )
    ]

    # Initialize Orchestrator and kickoff the process
    pm_orchestrator = Orchestrator(agents=agents, tasks=tasks, process='sequential', cumulative=False)
    user_query = f"Plan my trip from {from_city} to {destination_city} with details on attractions, costs, and logistics."
    result = pm_orchestrator.kickoff(user_query)

    # Return the result as JSON (or you can render in a template)
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
